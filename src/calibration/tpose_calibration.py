"""
Paso 6.1 — Calibración a partir de la T-pose.

Funciones:
- Promediar marcadores en la ventana estática para reducir ruido.
- Calcular centros articulares aplicando las reglas de marker_set.
- Medir longitudes de hueso y rellenar los OFFSETS del skeleton.

El input es un MarkerTrajectories que contiene (entre otros) los frames
correspondientes a la T-pose estática. El output es un Skeleton con
offsets reales y un dict de centros articulares en la T-pose (que se
usará luego como referencia para extraer rotaciones).
"""

from __future__ import annotations
import numpy as np

from src.io.markers_reader import MarkerTrajectories
from src.skeleton.skeleton import Skeleton
from config.marker_set import JOINT_CENTER_RULES
from config.settings import TPOSE_START_FRAME, TPOSE_DURATION_SEC


# -----------------------------------------------------------------------------
# Promediado en la ventana estática
# -----------------------------------------------------------------------------
def average_tpose_markers(markers: MarkerTrajectories) -> dict[str, np.ndarray]:
    """
    Promedia las posiciones de cada marcador en la ventana de T-pose.

    Returns
    -------
    dict { marker_name : (3,) ndarray }
    """
    num_static_frames = int(TPOSE_DURATION_SEC * markers.fps)
    start = TPOSE_START_FRAME
    end = min(start + num_static_frames, markers.num_frames)

    return {
        name: np.nanmean(traj[start:end], axis=0)
        for name, traj in markers.data.items()
    }


# -----------------------------------------------------------------------------
# Cálculo de centros articulares
# -----------------------------------------------------------------------------
def compute_joint_centers(markers_avg: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Aplica las reglas de marker_set.JOINT_CENTER_RULES para derivar el centro
    de cada articulación a partir de los marcadores superficiales.

    Returns
    -------
    dict { joint_name : (3,) ndarray }
    """
    centers: dict[str, np.ndarray] = {}

    for joint_name, rule in JOINT_CENTER_RULES.items():
        method = rule["method"]

        if method == "single":
            centers[joint_name] = markers_avg[rule["markers"][0]].copy()

        elif method == "midpoint":
            pts = np.stack([markers_avg[m] for m in rule["markers"]])
            centers[joint_name] = pts.mean(axis=0)

        elif method == "harrington":
            centers[joint_name] = _harrington_hip(markers_avg, side=rule["side"])

        else:
            raise ValueError(f"Método desconocido '{method}' para joint {joint_name}")

    return centers


def _harrington_hip(markers: dict[str, np.ndarray], side: str) -> np.ndarray:
    """
    Regresión de Harrington (2007) para estimar el centro de la cadera.

    Aproximación simplificada: a partir del ancho pélvico (LASI–RASI) y
    profundidad (centro ASI a centro PSI), desplaza el ASI homolateral
    hacia el centro articular.

    En una implementación final reemplazar por las ecuaciones completas:
        x = -0.24 * PW - 9.9
        y = -0.30 * PW - 10.9
        z = +0.33 * PW + 7.3        (mm, sistema ISB-pelvis)
    Pero requiere transformar al frame pélvico local primero.
    """
    LASI, RASI = markers["LASI"], markers["RASI"]
    LPSI, RPSI = markers["LPSI"], markers["RPSI"]

    pelvis_center = (LASI + RASI + LPSI + RPSI) / 4
    pw = np.linalg.norm(LASI - RASI)  # pelvic width

    # Desplazamiento hacia medial e inferior desde el ASI
    asi = LASI if side == "left" else RASI
    direction = (pelvis_center - asi)
    direction = direction / (np.linalg.norm(direction) + 1e-9)

    return asi + direction * 0.14 * pw  # aprox 14% del ancho pélvico hacia el centro


# -----------------------------------------------------------------------------
# Aplicar a un Skeleton: offsets + longitudes
# -----------------------------------------------------------------------------
def apply_to_skeleton(skeleton: Skeleton, joint_centers: dict[str, np.ndarray]) -> None:
    """
    Rellena los offsets (locales) y bone_lengths del Skeleton a partir de los
    centros articulares en T-pose.

    El offset de cada joint es su posición relativa a su PADRE.
    """
    for joint in skeleton.traverse():
        if joint.is_root():
            # Hips se posiciona en el mundo, su offset es 0 por convención
            joint.offset = np.zeros(3)
            continue

        if joint.is_end_site:
            # End sites: usar el último hueso visible o un offset pequeño por defecto
            parent_center = joint_centers.get(joint.parent.name)
            if parent_center is not None:
                joint.offset = np.array([0.0, -5.0, 0.0])  # placeholder
            continue

        parent_name = joint.parent.name
        if joint.name not in joint_centers or parent_name not in joint_centers:
            continue

        child_pos = joint_centers[joint.name]
        parent_pos = joint_centers[parent_name]
        offset = child_pos - parent_pos
        joint.offset = offset
        skeleton.set_bone_length(joint.name, float(np.linalg.norm(offset)))
