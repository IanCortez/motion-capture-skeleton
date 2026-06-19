"""
Paso 6.1 (variante) — Calibración SIN período de T-pose.

Cuando la captura no incluye un período estático en T-pose, las longitudes
de hueso y los offsets de referencia se derivan estadísticamente del
movimiento mismo:

  - Longitudes de hueso  → mediana de las distancias entre centros articulares
                           conectados, calculada sobre TODOS los frames.
                           Como el cuerpo es rígido, esta distancia es teóricamente
                           constante; la mediana absorbe el ruido y la deformación
                           de piel.

  - Direcciones de hueso → tomadas de una pose de referencia configurable
                           ('first_frame' o 'mean_pose'). Estas direcciones se
                           re-escalan a las longitudes medianas para garantizar
                           consistencia geométrica.

Limitación heredada (NO peor que con T-pose):
  - El twist (rotación sobre el eje del hueso) sigue sin poder estimarse sin
    marcadores adicionales o triadas. El extractor asume twist = 0.
"""

from __future__ import annotations
import numpy as np

from src.io.markers_reader import MarkerTrajectories
from src.skeleton.skeleton import Skeleton
from src.calibration.tpose_calibration import compute_joint_centers, apply_to_skeleton


# =============================================================================
# Cálculo estadístico de longitudes de hueso
# =============================================================================
def compute_statistical_bone_lengths(
    skeleton: Skeleton,
    markers: MarkerTrajectories,
    percentile: float = 50.0,
) -> dict[str, float]:
    """
    Devuelve un dict { joint_name : longitud } estimado robustamente desde el movimiento.

    Parameters
    ----------
    skeleton : skeleton ya construido (sin offsets aún).
    markers : trayectorias completas.
    percentile : 50 = mediana (default, recomendado).
                 Valores más altos (95) se acercan a la longitud rígida real, pero
                 son más sensibles a outliers. La mediana es el mejor compromiso.
    """
    # Centros articulares por frame para toda la captura
    per_frame_centers = _compute_centers_per_frame(markers)

    bone_lengths: dict[str, float] = {}
    for joint in skeleton.traverse():
        if joint.is_root() or joint.is_end_site:
            continue
        parent_name = joint.parent.name
        if joint.name not in per_frame_centers[0] or parent_name not in per_frame_centers[0]:
            continue

        distances = []
        for centers in per_frame_centers:
            d = np.linalg.norm(centers[joint.name] - centers[parent_name])
            if np.isfinite(d):
                distances.append(d)

        if distances:
            bone_lengths[joint.name] = float(np.percentile(distances, percentile))

    return bone_lengths


# =============================================================================
# Pose de referencia
# =============================================================================
def compute_reference_pose(
    markers: MarkerTrajectories,
    mode: str = "first_frame",
) -> dict[str, np.ndarray]:
    """
    Devuelve un dict { joint_name : (3,) } con la pose de referencia.

    mode : 'first_frame' (rápido, robusto si el frame 0 está completo)
           'mean_pose'   (promedio sobre toda la captura; más estable pero las
                          longitudes inter-articulares quedan distorsionadas —
                          se corrigen luego con _rescale_to_bone_lengths).
           'auto_neutral' (intenta detectar el frame más cercano a una pose
                           neutra: extremidades extendidas, máxima envergadura).
    """
    if mode == "first_frame":
        frame_markers = {name: markers[name][0] for name in markers.marker_names()}
        return compute_joint_centers(frame_markers)

    if mode == "mean_pose":
        mean_markers = {
            name: np.nanmean(markers[name], axis=0)
            for name in markers.marker_names()
        }
        return compute_joint_centers(mean_markers)

    if mode == "auto_neutral":
        ref_frame = _detect_neutral_frame(markers)
        print(f"  [auto_calib] frame neutro detectado: #{ref_frame}")
        frame_markers = {name: markers[name][ref_frame] for name in markers.marker_names()}
        return compute_joint_centers(frame_markers)

    raise ValueError(f"Modo desconocido: '{mode}'. Usar 'first_frame', 'mean_pose' o 'auto_neutral'.")


# =============================================================================
# Re-escalado: forzar las longitudes de hueso estadísticas en la pose de referencia
# =============================================================================
def rescale_to_bone_lengths(
    skeleton: Skeleton,
    ref_centers: dict[str, np.ndarray],
    bone_lengths: dict[str, float],
) -> dict[str, np.ndarray]:
    """
    Recorre el skeleton de raíz a hojas, conservando la DIRECCIÓN del hueso en
    ref_centers pero forzando su LONGITUD a la medida estadísticamente.

    Esto garantiza consistencia con bone_lengths sin perder la pose general.
    """
    rescaled: dict[str, np.ndarray] = {}

    # Raíz: se queda donde está
    root_name = skeleton.root.name
    if root_name in ref_centers:
        rescaled[root_name] = ref_centers[root_name].copy()

    # Recorrido en preorden propaga las posiciones recalculadas a los hijos
    for joint in skeleton.traverse():
        if joint.is_root() or joint.is_end_site:
            continue
        parent_name = joint.parent.name
        if joint.name not in ref_centers or parent_name not in rescaled:
            continue

        direction = ref_centers[joint.name] - ref_centers[parent_name]
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            rescaled[joint.name] = rescaled[parent_name].copy()
            continue

        target_length = bone_lengths.get(joint.name, norm)
        rescaled[joint.name] = rescaled[parent_name] + (direction / norm) * target_length

    return rescaled


# =============================================================================
# API principal — equivalente a apply_to_skeleton sin T-pose
# =============================================================================
def calibrate_auto(
    skeleton: Skeleton,
    markers: MarkerTrajectories,
    reference_mode: str = "first_frame",
    bone_length_percentile: float = 50.0,
) -> dict[str, np.ndarray]:
    """
    Calibra el skeleton sin necesidad de un período de T-pose.

    Devuelve los centros articulares de referencia (equivalente a `tpose_centers`
    en el flujo con T-pose) para que extractor.py los use como reference rest pose.

    Pasos:
      1. Calcula longitudes de hueso por mediana sobre toda la captura.
      2. Construye una pose de referencia (frame 0, pose media o frame neutro auto).
      3. Re-escala las direcciones de hueso de esa pose a las longitudes medianas.
      4. Aplica al skeleton (offsets + bone_lengths).
    """
    print(f"  [auto_calib] computing statistical bone lengths (p={bone_length_percentile})...")
    bone_lengths = compute_statistical_bone_lengths(skeleton, markers, bone_length_percentile)

    print(f"  [auto_calib] reference pose mode: '{reference_mode}'")
    ref_centers = compute_reference_pose(markers, mode=reference_mode)

    print(f"  [auto_calib] rescaling reference pose to median bone lengths...")
    final_centers = rescale_to_bone_lengths(skeleton, ref_centers, bone_lengths)

    apply_to_skeleton(skeleton, final_centers)
    return final_centers


# =============================================================================
# Helpers internos
# =============================================================================
def _compute_centers_per_frame(markers: MarkerTrajectories) -> list[dict[str, np.ndarray]]:
    """Aplica compute_joint_centers a cada frame y devuelve la lista."""
    centers_list = []
    for f in range(markers.num_frames):
        frame_markers = {name: markers[name][f] for name in markers.marker_names()}
        try:
            centers_list.append(compute_joint_centers(frame_markers))
        except KeyError:
            # Algún marcador faltante en este frame — saltar
            continue
    return centers_list


def _detect_neutral_frame(markers: MarkerTrajectories) -> int:
    """
    Heurística simple: busca el frame con MÁXIMA envergadura (distancia mano
    izquierda ↔ mano derecha) AND mínima velocidad. Aproximación a un T-pose
    espontáneo si el sujeto extiende los brazos en algún momento.

    Si no se puede calcular, devuelve 0.
    """
    if "LFIN" not in markers or "RFIN" not in markers:
        return 0

    span = np.linalg.norm(markers["LFIN"] - markers["RFIN"], axis=1)

    # Velocidad: norma de la diferencia entre frames consecutivos
    vel = np.linalg.norm(np.diff(markers["LFIN"], axis=0, prepend=markers["LFIN"][:1]), axis=1)
    vel += np.linalg.norm(np.diff(markers["RFIN"], axis=0, prepend=markers["RFIN"][:1]), axis=1)

    # Score: priorizar gran envergadura + poca velocidad
    span_n = (span - np.nanmin(span)) / (np.nanmax(span) - np.nanmin(span) + 1e-9)
    vel_n = (vel - np.nanmin(vel)) / (np.nanmax(vel) - np.nanmin(vel) + 1e-9)
    score = span_n - vel_n

    return int(np.nanargmax(score))
