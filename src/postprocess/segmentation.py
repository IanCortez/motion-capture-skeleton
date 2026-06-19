"""
Paso 7.2 — Segmentación del movimiento.

Detecta puntos de corte naturales (reposos / cambios de actividad) en la
captura. Estrategia simple basada en energía cinética articular:
1. Calcular norma de velocidad angular promediada sobre todos los joints.
2. Suavizar con un filtro Gaussiano.
3. Buscar mínimos locales prolongados → frames de reposo → cortes.
"""

from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d

from src.skeleton.skeleton import Skeleton
from src.utils.math_utils import quat_to_euler
from config.settings import BVH_ROTATION_ORDER


def segment_motion(
    skeleton: Skeleton,
    fps: float,
    rest_threshold: float = 5.0,     # grados/s
    min_rest_duration: float = 0.4,  # segundos
    smooth_sigma_sec: float = 0.2,
) -> list[tuple[int, int]]:
    """
    Devuelve una lista de (start_frame, end_frame_exclusive) para cada segmento
    de movimiento. Si no se detectan cortes, devuelve un único segmento.

    Parameters
    ----------
    skeleton : skeleton animado.
    fps : framerate.
    rest_threshold : umbral de "actividad mínima" para considerar movimiento.
    min_rest_duration : duración mínima del reposo para considerarlo un corte real.
    smooth_sigma_sec : sigma del filtro Gaussiano.
    """
    assert skeleton.local_rotations is not None
    activity = _compute_activity_signal(skeleton, fps)
    activity_smooth = gaussian_filter1d(activity, sigma=smooth_sigma_sec * fps)

    num_frames = len(activity_smooth)
    is_rest = activity_smooth < rest_threshold

    # Encontrar runs de reposo lo suficientemente largos para ser cortes
    cut_points: list[int] = []
    min_rest_frames = int(min_rest_duration * fps)
    i = 0
    while i < num_frames:
        if not is_rest[i]:
            i += 1
            continue
        j = i
        while j < num_frames and is_rest[j]:
            j += 1
        if j - i >= min_rest_frames:
            cut_points.append((i + j) // 2)
        i = j

    # Construir segmentos a partir de los cortes
    if not cut_points:
        return [(0, num_frames)]

    segments = []
    prev = 0
    for cp in cut_points:
        segments.append((prev, cp))
        prev = cp
    segments.append((prev, num_frames))
    return segments


def _compute_activity_signal(skeleton: Skeleton, fps: float) -> np.ndarray:
    """
    Señal de actividad: para cada frame, promedio de magnitudes de velocidad
    angular sobre todos los joints. En grados/segundo.
    """
    rots = skeleton.local_rotations
    num_frames, num_joints, _ = rots.shape

    # Convertimos todo a Euler de una vez para velocidad
    eulers = np.zeros((num_frames, num_joints, 3))
    for j in range(num_joints):
        for f in range(num_frames):
            eulers[f, j] = quat_to_euler(rots[f, j], order=BVH_ROTATION_ORDER, degrees=True)

    # Velocidad angular: diferencia entre frames, escalada por fps
    diff = np.diff(eulers, axis=0, prepend=eulers[:1])
    speeds = np.linalg.norm(diff, axis=2) * fps  # (T, N_joints)
    return speeds.mean(axis=1)
