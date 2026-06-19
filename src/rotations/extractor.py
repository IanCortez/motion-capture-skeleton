"""
Paso 6.4 — Extracción de rotaciones articulares a partir de posiciones.

Estrategia
----------
1. La RAÍZ (Hips) usa ``build_pelvis_frame`` sobre los 4 marcadores pélvicos
   (LASI, RASI, LPSI, RPSI) para recuperar los 3 DoF — incluyendo el yaw —.
   Un vector hueso único no puede recuperar el yaw cuando el hueso es vertical.

2. Los joints NO-RAÍZ calculan su rotación local DIRECTAMENTE en el frame
   local ACTUAL del padre (no en el mundo). Para ello:
     a) Se toma el vector hueso actual en world (child_pos - joint_pos).
     b) Se transforma al frame local del padre con ``R_parent_world.inv()``.
     c) Se calcula el swing en ese frame local: rest_vec → current_in_parent_local.
   Luego ``world_rot(joint) = parent_world ∘ local_rot``.

Por qué esto resuelve el giro de 180°
-------------------------------------
Cuando el cuerpo gira 180° alrededor de Y:
  - Hips.world = 180°Y  (de build_pelvis_frame)
  - Spine: current_vec_world = (0,1,0) (hueso vertical, no cambia).
    Pero en el frame local de Hips: R(180°Y).inv() · (0,1,0) = (0,1,0).
    Swing en frame local = quat_from_vectors((0,1,0), (0,1,0)) = identidad. ✓
    Spine.world = 180°Y * identidad = 180°Y → hereda el yaw del padre.
  - Chest, Neck: ídem, locales = identidad.
  - LeftShoulder: current_vec_world = (-1,0,0) (giró con el cuerpo).
    En el frame local del Chest: R(180°Y).inv() · (-1,0,0) = (1,0,0).
    Swing = quat_from_vectors((1,0,0), (1,0,0)) = identidad. ✓

Bug que existía antes
---------------------
El código viejo calculaba ``world_rot = quat_from_vectors(rest, current)`` en
WORLD. Para huesos verticales esto da identidad incluso si el cuerpo gira,
porque el hueso no cambia de dirección en world. Luego
``local_rot = parent_world.inv() · world_rot = parent_world.inv()``
compensaba el giro del padre, así que el cuerpo se "deshacía" en cascada:
Hips miraba atrás, Spine deshacía el giro, los hombros recibían twist extra.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.skeleton.skeleton import Skeleton
from src.skeleton.joint import Joint
from src.utils.math_utils import (
    quat_from_vectors,
    quat_mul,
    quat_inverse,
    build_pelvis_frame,
    build_pelvis_frame_from_joints,
)
from src.io.markers_reader import MarkerTrajectories


_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])
_PELVIS_MARKERS = ("LASI", "RASI", "LPSI", "RPSI")
_PELVIS_JOINTS = ("Hips", "Spine", "LeftHip", "RightHip")


def extract_rotations(
    skeleton: Skeleton,
    joint_positions_per_frame: np.ndarray,
    joint_name_to_idx: dict[str, int],
    tpose_centers: dict[str, np.ndarray],
    markers: MarkerTrajectories | None = None,
    tpose_markers_avg: dict[str, np.ndarray] | None = None,
) -> None:
    """
    Llena skeleton.local_rotations a partir de posiciones globales por frame.

    Parameters
    ----------
    skeleton : esqueleto con offsets configurados (T-pose).
    joint_positions_per_frame : (T, M, 3) posiciones globales por frame.
    joint_name_to_idx : nombre → índice M.
    tpose_centers : centros articulares en T-pose (para vectores hueso de reposo).
    markers : trayectorias de marcadores; si trae LASI/RASI/LPSI/RPSI se usa
        ``build_pelvis_frame`` para la raíz (necesario para giros de 180°).
    tpose_markers_avg : posiciones promedio de los marcadores en la ventana de
        T-pose (las que produce ``tpose_calibration.average_tpose_markers``).
        Si se provee y contiene LASI/RASI/LPSI/RPSI, define la orientación
        pélvica de reposo Q0 de forma robusta. Si es None, Q0 se estima de los
        primeros frames no ocluidos de ``markers`` (asume que la captura empieza
        en T-pose). Pasar este valor es lo recomendado.
    """
    assert skeleton.local_rotations is not None, "Llamar a allocate_motion() primero."
    num_frames = joint_positions_per_frame.shape[0]

    # Rellena oclusiones (NaN) en las posiciones de joints por interpolación
    # temporal. Los marcadores ocluidos llegan como NaN y, sin rellenar,
    # producirían frames con rotación identidad (saltos) o NaN en el BVH.
    # Interpolar mantiene la animación continua; los joints sin NINGÚN frame
    # válido se quedan en NaN y los guardas de math_utils los vuelven identidad.
    joint_positions_per_frame = _fill_nan_gaps(joint_positions_per_frame)

    rest_bone_vectors = _compute_rest_bone_vectors(skeleton, tpose_centers)

    # ¿Tenemos los joints necesarios para construir el frame pélvico a partir
    # de POSICIONES (Hips, Spine, LeftHip, RightHip)? Es el método primario:
    # recupera el yaw sin depender de marcadores crudos y funciona con la
    # llamada original de 4 argumentos del pipeline.
    use_joint_pelvis_frame = all(
        n in joint_name_to_idx for n in _PELVIS_JOINTS
    ) and all(n in tpose_centers for n in _PELVIS_JOINTS)

    # Frame pélvico de la T-pose (Q0). El actor casi nunca está perfectamente
    # alineado con los ejes del mundo en reposo, así que su pelvis tiene una
    # orientación Q0 != identidad. La rotación de la raíz debe ser RELATIVA a
    # Q0; de lo contrario, esa inclinación de reposo se filtra como flexión
    # falsa en columna y hombros y rompe el rig durante los giros.
    tpose_pelvis_quat = _compute_tpose_pelvis_quat(
        tpose_centers, tpose_markers_avg, markers, use_joint_pelvis_frame
    )

    for frame in range(num_frames):
        positions_world = joint_positions_per_frame[frame]
        # world_rotations almacena la rotación WORLD real de cada joint en
        # este frame, para que los hijos puedan transformar sus vectores
        # al frame local del padre.
        world_rotations: dict[str, np.ndarray] = {}

        for joint in skeleton.traverse():
            if joint.is_end_site:
                continue

            if joint.is_root():
                # Raíz: frame pélvico completo (3 DoF). Sin padre, la rotación
                # local = rotación world.
                local_rot = _compute_root_local_rotation(
                    joint, positions_world, joint_name_to_idx,
                    rest_bone_vectors, markers, frame,
                    use_joint_pelvis_frame, tpose_pelvis_quat,
                )
                world_rot = local_rot
            else:
                # NO-raíz: rotación local en el frame local ACTUAL del padre.
                # Esto es la clave del fix de la cascada.
                parent_world = world_rotations[joint.parent.name]
                local_rot = _compute_local_rotation_in_parent_frame(
                    joint, positions_world, joint_name_to_idx,
                    rest_bone_vectors, parent_world,
                )
                # Componer hacia world para que los hijos lo usen.
                world_rot = quat_mul(parent_world, local_rot)

            world_rotations[joint.name] = world_rot
            skeleton.local_rotations[frame, joint.index] = local_rot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill_nan_gaps(positions: np.ndarray) -> np.ndarray:
    """
    Rellena huecos NaN en (T, M, 3) por interpolación lineal en el tiempo.

    - Cada componente (joint, eje) se interpola usando solo sus frames válidos.
    - Los extremos se sostienen con el primer/último valor válido (np.interp).
    - Un joint sin ningún frame válido se deja en NaN (no hay con qué rellenar);
      los guardas de ``math_utils`` lo tratan como identidad sin romper el SVD.

    No modifica el array de entrada: devuelve una copia.
    """
    positions = np.asarray(positions, dtype=float)
    filled = positions.copy()
    num_frames, num_joints, num_axes = filled.shape
    t = np.arange(num_frames)
    for j in range(num_joints):
        for a in range(num_axes):
            col = filled[:, j, a]
            valid = ~np.isnan(col)
            if valid.all() or not valid.any():
                continue
            filled[:, j, a] = np.interp(t, t[valid], col[valid])
    return filled


def _compute_rest_bone_vectors(
    skeleton: Skeleton, tpose_centers: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """
    Para cada joint con hijos, el vector hueso de reposo apunta al primer
    hijo no-EndSite en T-pose. Suponemos T-pose canónica (cuerpo alineado con
    los ejes del mundo), por lo que el vector en world = vector en frame local
    del padre.
    """
    rest_vectors: dict[str, np.ndarray] = {}
    for joint in skeleton.traverse():
        if joint.is_end_site or joint.name not in tpose_centers:
            continue
        primary_child = next(
            (c for c in joint.children
             if not c.is_end_site and c.name in tpose_centers),
            None,
        )
        if primary_child is None:
            rest_vectors[joint.name] = np.array([0.0, 1.0, 0.0])
            continue
        v = tpose_centers[primary_child.name] - tpose_centers[joint.name]
        n = np.linalg.norm(v)
        rest_vectors[joint.name] = v / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
    return rest_vectors


def _compute_tpose_pelvis_quat(
    tpose_centers: dict[str, np.ndarray],
    tpose_markers_avg: dict[str, np.ndarray] | None,
    markers: MarkerTrajectories | None,
    use_joint_pelvis_frame: bool,
) -> np.ndarray:
    """
    Orientación de la pelvis en la T-pose (Q0). Prioridad:

    1. Posiciones de joints en T-pose (Hips, Spine, LeftHip, RightHip) — MÉTODO
       PRIMARIO, no requiere marcadores y es consistente con el cálculo por
       frame.
    2. ``tpose_markers_avg`` con los 4 marcadores pélvicos — si se proveyó.
    3. Primeros frames no ocluidos de ``markers`` — asume captura inicial en
       T-pose.
    4. Identidad.
    """
    # 1. Joints de la T-pose (preferido)
    if use_joint_pelvis_frame:
        return build_pelvis_frame_from_joints(
            tpose_centers["Hips"], tpose_centers["Spine"],
            tpose_centers["LeftHip"], tpose_centers["RightHip"],
        )

    # 2. Calibración explícita de marcadores de T-pose
    if tpose_markers_avg is not None and all(
        m in tpose_markers_avg for m in _PELVIS_MARKERS
    ):
        lasi = tpose_markers_avg["LASI"]; rasi = tpose_markers_avg["RASI"]
        lpsi = tpose_markers_avg["LPSI"]; rpsi = tpose_markers_avg["RPSI"]
        if not (np.any(np.isnan(lasi)) or np.any(np.isnan(rasi))
                or np.any(np.isnan(lpsi)) or np.any(np.isnan(rpsi))):
            return build_pelvis_frame(lasi, rasi, lpsi, rpsi)

    # 3. Primeros frames de la captura
    if markers is not None and all(m in markers for m in _PELVIS_MARKERS):
        for f in range(min(markers.num_frames, 30)):
            lasi = markers["LASI"][f]; rasi = markers["RASI"][f]
            lpsi = markers["LPSI"][f]; rpsi = markers["RPSI"][f]
            if not (np.any(np.isnan(lasi)) or np.any(np.isnan(rasi))
                    or np.any(np.isnan(lpsi)) or np.any(np.isnan(rpsi))):
                return build_pelvis_frame(lasi, rasi, lpsi, rpsi)

    # 4. Identidad
    return _IDENTITY_QUAT.copy()


def _compute_root_local_rotation(
    joint: Joint,
    positions_world: np.ndarray,
    name_to_idx: dict[str, int],
    rest_bone_vectors: dict[str, np.ndarray],
    markers: MarkerTrajectories | None,
    frame: int,
    use_joint_pelvis_frame: bool,
    tpose_pelvis_quat: np.ndarray,
) -> np.ndarray:
    """
    Raíz: rotación de animación RELATIVA a la T-pose, de 3 DoF.

    A_root = build_pelvis_frame(actual) * Q0^-1
    donde Q0 = orientación pélvica en T-pose. Esto da:
      - En T-pose:      A_root = Q0 * Q0^-1 = identidad  (rest = rotación cero).
      - Giro rígido R:  build_pelvis = R*Q0  →  A_root = R*Q0*Q0^-1 = R.
    Es decir, la raíz recupera exactamente la rotación world del cuerpo, sin
    contaminación de la inclinación de reposo. Al ser identidad en reposo, los
    hijos la heredan correctamente (parent_world = identidad en T-pose).

    El frame pélvico se construye preferentemente a partir de las POSICIONES de
    los joints (Hips, Spine, LeftHip, RightHip), lo que recupera el yaw sin
    necesitar marcadores y funciona con la llamada original del pipeline. Sólo
    si esos joints no están disponibles se usan los marcadores pélvicos, y como
    último recurso el swing (que pierde el yaw).
    """
    # Método primario: frame pélvico a partir de joints (no requiere markers).
    if use_joint_pelvis_frame:
        hips = positions_world[name_to_idx["Hips"]]
        spine = positions_world[name_to_idx["Spine"]]
        lhip = positions_world[name_to_idx["LeftHip"]]
        rhip = positions_world[name_to_idx["RightHip"]]
        world_pelvis = build_pelvis_frame_from_joints(hips, spine, lhip, rhip)
        return quat_mul(world_pelvis, quat_inverse(tpose_pelvis_quat))

    # Alternativa: marcadores pélvicos crudos.
    if markers is not None and all(m in markers for m in _PELVIS_MARKERS):
        lasi = markers["LASI"][frame]; rasi = markers["RASI"][frame]
        lpsi = markers["LPSI"][frame]; rpsi = markers["RPSI"][frame]
        if not (np.any(np.isnan(lasi)) or np.any(np.isnan(rasi))
                or np.any(np.isnan(lpsi)) or np.any(np.isnan(rpsi))):
            world_pelvis = build_pelvis_frame(lasi, rasi, lpsi, rpsi)
            return quat_mul(world_pelvis, quat_inverse(tpose_pelvis_quat))

    # Último recurso: swing en world (pierde yaw). Identidad en reposo.
    return _swing_from_bone(
        joint, positions_world, name_to_idx, rest_bone_vectors,
        preferred_axis=np.array([0.0, 1.0, 0.0]),
    )


def _compute_local_rotation_in_parent_frame(
    joint: Joint,
    positions_world: np.ndarray,
    name_to_idx: dict[str, int],
    rest_bone_vectors: dict[str, np.ndarray],
    parent_world_quat: np.ndarray,
) -> np.ndarray:
    """
    Calcula la rotación local del joint expresando el vector hueso ACTUAL
    en el frame local del padre y comparándolo con rest_vec.

    ESTA ES LA CORRECCIÓN PRINCIPAL del bug de la cascada.

    Cuando el padre gira (p.ej. cuerpo girado 180°), este método identifica
    correctamente que el hueso NO se movió respecto al padre — solo fue
    arrastrado por él. La rotación local resulta identidad y el yaw del padre
    se propaga a los hijos por composición (en lugar de cancelarse).
    """
    if joint.name not in rest_bone_vectors or joint.name not in name_to_idx:
        return _IDENTITY_QUAT.copy()

    primary_child = next(
        (c for c in joint.children if not c.is_end_site and c.name in name_to_idx),
        None,
    )
    if primary_child is None:
        return _IDENTITY_QUAT.copy()

    rest_vec = rest_bone_vectors[joint.name]
    current_vec_world = (
        positions_world[name_to_idx[primary_child.name]]
        - positions_world[name_to_idx[joint.name]]
    )
    norm = np.linalg.norm(current_vec_world)
    if norm < 1e-9:
        return _IDENTITY_QUAT.copy()
    current_vec_world = current_vec_world / norm

    # Transformar current_vec de world al frame local ACTUAL del padre.
    R_parent = R.from_quat(parent_world_quat)
    current_vec_in_parent_local = R_parent.inv().apply(current_vec_world)

    # Swing en el frame local del padre. Eje preferido = Y local del padre
    # (que en el propio frame local es simplemente (0,1,0)).
    return quat_from_vectors(
        rest_vec, current_vec_in_parent_local,
        preferred_axis=np.array([0.0, 1.0, 0.0]),
    )


def _swing_from_bone(
    joint: Joint,
    positions_world: np.ndarray,
    name_to_idx: dict[str, int],
    rest_bone_vectors: dict[str, np.ndarray],
    preferred_axis: np.ndarray | None = None,
) -> np.ndarray:
    """
    Swing en world (T-pose-bone → current-bone). Solo se usa como fallback
    de la raíz cuando los marcadores pélvicos están ocluidos. No recupera
    yaw para huesos verticales.
    """
    if joint.name not in rest_bone_vectors or joint.name not in name_to_idx:
        return _IDENTITY_QUAT.copy()

    primary_child = next(
        (c for c in joint.children if not c.is_end_site and c.name in name_to_idx),
        None,
    )
    if primary_child is None:
        return _IDENTITY_QUAT.copy()

    rest_vec = rest_bone_vectors[joint.name]
    current_vec = (
        positions_world[name_to_idx[primary_child.name]]
        - positions_world[name_to_idx[joint.name]]
    )
    norm = np.linalg.norm(current_vec)
    if norm < 1e-9:
        return _IDENTITY_QUAT.copy()
    current_vec = current_vec / norm

    return quat_from_vectors(rest_vec, current_vec, preferred_axis=preferred_axis)