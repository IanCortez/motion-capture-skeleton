"""
Utilidades matemáticas para manejo de rotaciones.

Trabajamos internamente con cuaterniones (estables, sin gimbal lock) y solo
convertimos a Euler en el momento de exportar a BVH. La descomposición
swing-twist se usa en constraints.py para clampear rangos articulares.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


# -----------------------------------------------------------------------------
# Conversiones básicas
# -----------------------------------------------------------------------------
def quat_from_vectors(v_from: np.ndarray, v_to: np.ndarray,
                      preferred_axis: np.ndarray | None = None) -> np.ndarray:
    """
    Cuaternión que rota v_from hasta v_to (componente swing, sin twist).

    Parameters
    ----------
    v_from, v_to : vectores de dirección (no necesitan ser unitarios).
    preferred_axis : eje preferido para resolver la ambigüedad cuando los
        vectores son antiparalelos (rotación de 180°). Si es None se usa
        el eje Y del mundo, que corresponde al giro vertical natural del
        cuerpo humano. Pasar el eje Y del frame padre mejora la consistencia
        en joints de la columna y pelvis.

    Returns
    -------
    quat : (4,) array en formato (x, y, z, w) compatible con scipy.

    Notas
    -----
    Cuando dot(v_from, v_to) ≈ −1 existe un plano entero de ejes válidos.
    La elección arbitraria del eje produce saltos discontinuos en la animación
    cuando el cuerpo gira ~180° (de frente a de espaldas). Usar un eje
    preferido consistente (Y mundial) elimina estos artefactos.
    """
    v_from = np.asarray(v_from, dtype=float)
    v_to = np.asarray(v_to, dtype=float)
    # Guarda NaN/inf: una dirección no finita (marcador ocluido) no define una
    # rotación; devolvemos identidad en vez de propagar NaN al cuaternión.
    if not (np.all(np.isfinite(v_from)) and np.all(np.isfinite(v_to))):
        return np.array([0.0, 0.0, 0.0, 1.0])

    v_from = v_from / (np.linalg.norm(v_from) + 1e-12)
    v_to   = v_to   / (np.linalg.norm(v_to)   + 1e-12)

    dot = np.clip(np.dot(v_from, v_to), -1.0, 1.0)

    if dot > 0.9999:
        return np.array([0.0, 0.0, 0.0, 1.0])  # identidad

    if dot < -0.9999:
        # Rotación de 180°: hay infinitos ejes válidos. Usamos un eje preferido
        # para obtener continuidad temporal al girar el cuerpo.
        # Orden de candidatos: preferred_axis → Y mundial → X mundial → Z mundial
        candidates = []
        if preferred_axis is not None:
            candidates.append(np.asarray(preferred_axis, dtype=float))
        candidates += [
            np.array([0.0, 1.0, 0.0]),  # Y mundo: giro vertical natural
            np.array([1.0, 0.0, 0.0]),  # X mundo: fallback
            np.array([0.0, 0.0, 1.0]),  # Z mundo: último recurso
        ]
        for candidate in candidates:
            axis = np.cross(v_from, candidate)
            n = np.linalg.norm(axis)
            if n > 1e-6:
                axis = axis / n
                return R.from_rotvec(axis * np.pi).as_quat()
        # Caso degenerado extremo: devolver identidad
        return np.array([0.0, 0.0, 0.0, 1.0])

    axis = np.cross(v_from, v_to)
    angle = np.arccos(dot)
    return R.from_rotvec(axis / np.linalg.norm(axis) * angle).as_quat()


def quat_to_euler(quat: np.ndarray, order: str = "ZXY", degrees: bool = True) -> np.ndarray:
    """
    Convierte cuaternión a ángulos de Euler en el orden especificado.

    El orden debe coincidir con CHANNELS en el BVH writer.
    """
    return R.from_quat(quat).as_euler(order, degrees=degrees)


def euler_to_quat(euler: np.ndarray, order: str = "ZXY", degrees: bool = True) -> np.ndarray:
    return R.from_euler(order, euler, degrees=degrees).as_quat()


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Producto de cuaterniones (composición de rotaciones)."""
    return (R.from_quat(q1) * R.from_quat(q2)).as_quat()


def quat_inverse(q: np.ndarray) -> np.ndarray:
    return R.from_quat(q).inv().as_quat()


def quat_relative(parent_world: np.ndarray, child_world: np.ndarray) -> np.ndarray:
    """
    Rotación local del hijo en el frame del padre:
        R_local = R_parent^-1 * R_child
    """
    return quat_mul(quat_inverse(parent_world), child_world)


# -----------------------------------------------------------------------------
# Descomposición swing-twist (para clampear restricciones biomecánicas)
# -----------------------------------------------------------------------------
def swing_twist_decompose(quat: np.ndarray, twist_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Descompone una rotación en swing (perpendicular al eje) + twist (sobre el eje).

    Parameters
    ----------
    quat : cuaternión (x, y, z, w)
    twist_axis : eje del hueso (normalizado), p.ej. (1, 0, 0) si X apunta al hijo.

    Returns
    -------
    (swing_quat, twist_quat)
    """
    twist_axis = twist_axis / (np.linalg.norm(twist_axis) + 1e-12)
    qv = quat[:3]  # parte vectorial
    proj = np.dot(qv, twist_axis) * twist_axis

    twist = np.array([proj[0], proj[1], proj[2], quat[3]])
    twist_norm = np.linalg.norm(twist)
    if twist_norm < 1e-9:
        twist = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        twist = twist / twist_norm

    swing = quat_mul(quat, quat_inverse(twist))
    return swing, twist


# -----------------------------------------------------------------------------
# Slerp para blending temporal
# -----------------------------------------------------------------------------
def slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Interpolación esférica entre dos cuaterniones (t ∈ [0, 1])."""
    key_times = [0.0, 1.0]
    key_rots = R.from_quat([q1, q2])
    from scipy.spatial.transform import Slerp
    slerp_obj = Slerp(key_times, key_rots)
    return slerp_obj([t]).as_quat()[0]


# -----------------------------------------------------------------------------
# Frame de orientación de la pelvis (3 DoF completos)
# -----------------------------------------------------------------------------
def build_pelvis_frame(
    lasi: np.ndarray,
    rasi: np.ndarray,
    lpsi: np.ndarray,
    rpsi: np.ndarray,
) -> np.ndarray:
    """
    Calcula la orientación completa (3 DoF) de la pelvis a partir de los
    cuatro marcadores estándar del cinturón pélvico.

    A diferencia de ``quat_from_vectors``, que solo recupera el swing de
    un hueso (2 DoF), este frame captura también el twist/yaw, por lo que
    funciona correctamente cuando el sujeto gira 180° (de frente a de espaldas).

    Sistema de ejes del frame pélvico (convención ISB / Y-up):
    - X  →  lateral derecho  (RASI - LASI)
    - Y  →  superior         (producto cruzado de X y Z)
    - Z  →  anterior         (de PSI a ASI)

    Parameters
    ----------
    lasi, rasi : marcadores espinas ilíacas antero-superiores izq./der.
    lpsi, rpsi : marcadores espinas ilíacas postero-superiores izq./der.

    Returns
    -------
    quat : (4,) cuaternión (x, y, z, w) que representa la orientación
           global de la pelvis. Multiplícalo por el cuaternión de T-pose
           invertido para obtener la rotación relativa al reposo.

    Notas
    -----
    Si alguno de los marcadores viene como NaN (oclusión), la función
    degrada con gracia: intenta usar los que estén disponibles. Si todos
    faltan, devuelve identidad.
    """
    # Guarda NaN: si algún marcador está ocluido en este frame, devolver identidad.
    if (np.any(np.isnan(lasi)) or np.any(np.isnan(rasi))
            or np.any(np.isnan(lpsi)) or np.any(np.isnan(rpsi))):
        return np.array([0.0, 0.0, 0.0, 1.0])

    # Centro ASI y PSI para el eje anterior-posterior
    asi_center = (lasi + rasi) / 2.0
    psi_center = (lpsi + rpsi) / 2.0

    # Eje Z: anterior (de sacro a pubis, es decir PSI → ASI)
    # Se calcula primero porque define la dirección "hacia adelante".
    fwd = asi_center - psi_center
    fwd_norm = np.linalg.norm(fwd)
    if fwd_norm < 1e-9:
        fwd = np.array([0.0, 0.0, 1.0])
    else:
        fwd = fwd / fwd_norm

    # Eje temporal "hacia arriba" usando el Y mundial
    up_approx = np.array([0.0, 1.0, 0.0])

    # Eje X: lateral derecho del paciente = cross(fwd, up_approx)
    # Cuando el paciente mira +Z, su derecha es +X si fwd=[0,0,1] y up=[0,1,0]:
    #   cross([0,0,1], [0,1,0]) = [-1, 0, 0]  ← necesitamos negarlo
    # En cambio cross(up_approx, fwd) = [1, 0, 0] ✓
    # Convención ISB: X apunta al eje derecho del paciente
    right = np.cross(up_approx, fwd)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-9:
        # El paciente está tumbado horizontalmente; usar RASI-LASI como fallback
        right = rasi - lasi
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0])
    right = right / right_norm

    # Ortogonalizar fwd respecto a right (Gram-Schmidt) para garantizar
    # perpendicularidad estricta.
    fwd = fwd - np.dot(fwd, right) * right
    fwd_norm = np.linalg.norm(fwd)
    if fwd_norm < 1e-9:
        fwd = np.array([0.0, 0.0, 1.0])
    else:
        fwd = fwd / fwd_norm

    # Eje Y: superior. En sistema dextrogiro con X=right, Z=fwd: Y = Z × X = fwd × right
    up = np.cross(fwd, right)
    up_norm = np.linalg.norm(up)
    if up_norm < 1e-9:
        up = np.array([0.0, 1.0, 0.0])
    else:
        up = up / up_norm

    # Matriz de rotacion 3x3: columnas = ejes del frame pelvico
    rot_matrix = np.column_stack([right, up, fwd])

    # Red de seguridad: si algo se volvió no finito, no alimentar el SVD de
    # from_matrix (lanzaría "SVD did not converge"); devolver identidad.
    if not np.all(np.isfinite(rot_matrix)):
        return np.array([0.0, 0.0, 0.0, 1.0])

    # Garantizar que sea una rotacion propia (det = +1)
    if np.linalg.det(rot_matrix) < 0:
        rot_matrix[:, 2] = -rot_matrix[:, 2]

    return R.from_matrix(rot_matrix).as_quat()

# -----------------------------------------------------------------------------
# Frame de orientación de la pelvis a partir de POSICIONES DE JOINTS (3 DoF)
# -----------------------------------------------------------------------------
def build_pelvis_frame_from_joints(
    hips: np.ndarray,
    spine: np.ndarray,
    left_hip: np.ndarray,
    right_hip: np.ndarray,
) -> np.ndarray:
    """
    Orientación completa (3 DoF) de la pelvis a partir de las posiciones de los
    joints Hips, Spine, LeftHip y RightHip.

    Es la alternativa a ``build_pelvis_frame`` (que usa marcadores crudos): aquí
    sólo se necesitan las posiciones de joints que ``extract_rotations`` ya
    recibe tras FABRIK. Recupera el yaw porque usa DOS direcciones independientes
    (vertical de la columna + eje lateral de las caderas), no un único hueso.

    Ventajas frente a los marcadores:
    - No depende de marcadores que se ocluyen cuando el sujeto da la espalda.
    - Los centros de cadera y la base de la columna se mueven rígidamente con la
      pelvis, así que el frame gira de forma estable con el cuerpo.

    Ejes (sistema dextrogiro, Y-up por convención, det = +1):
    - Y (up)      : Hips → Spine
    - X (lateral) : LeftHip → RightHip (ortogonalizado)
    - Z (forward) : X × Y

    Notas
    -----
    La convención absoluta de ejes no es crítica: ``extract_rotations`` usa este
    frame SIEMPRE de forma relativa a la T-pose (A_root = frame_actual · Q0⁻¹),
    de modo que cualquier offset constante se cancela. Lo esencial es que sea un
    frame ortonormal propio, continuo y que gire rígidamente con la pelvis.

    Si los joints están degenerados (coincidentes), devuelve identidad.
    """
    # Guarda NaN/inf: si alguna posición de joint está ocluida en este frame,
    # no se puede definir el frame; devolver identidad en vez de propagar NaN
    # hasta R.from_matrix (cuyo SVD interno fallaría con "SVD did not converge").
    for p in (hips, spine, left_hip, right_hip):
        if not np.all(np.isfinite(p)):
            return np.array([0.0, 0.0, 0.0, 1.0])

    # Eje vertical: a lo largo de la columna
    up = spine - hips
    up_norm = np.linalg.norm(up)
    if up_norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    up = up / up_norm

    # Eje lateral: línea de caderas (izq → der)
    lateral = right_hip - left_hip
    lateral_norm = np.linalg.norm(lateral)
    if lateral_norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    lateral = lateral / lateral_norm

    # Ortogonalizar lateral respecto a up (Gram-Schmidt)
    right = lateral - np.dot(lateral, up) * up
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    right = right / right_norm

    # Forward = right × up  (sistema dextrogiro)
    fwd = np.cross(right, up)
    fwd_norm = np.linalg.norm(fwd)
    if fwd_norm < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0])
    fwd = fwd / fwd_norm

    rot_matrix = np.column_stack([right, up, fwd])
    if not np.all(np.isfinite(rot_matrix)):
        return np.array([0.0, 0.0, 0.0, 1.0])
    if np.linalg.det(rot_matrix) < 0:
        rot_matrix[:, 2] = -rot_matrix[:, 2]

    return R.from_matrix(rot_matrix).as_quat()