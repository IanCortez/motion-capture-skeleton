"""
Paso 6.5 — Restricciones biomecánicas.

Clampea las rotaciones locales de cada joint a su rango fisiológico.
Implementación simple por defecto: clamp de ángulos de Euler.
Para una versión más robusta, descomponer en swing-twist y aplicar
límites por componente (TODO).
"""

from __future__ import annotations
import numpy as np

from src.skeleton.skeleton import Skeleton
from src.utils.math_utils import quat_to_euler, euler_to_quat
from config.settings import BVH_ROTATION_ORDER


# -----------------------------------------------------------------------------
# Tabla de límites por joint (en grados).
# Cada entrada: (axis_min, axis_max) para X, Y, Z en orden.
# None significa "sin límite".
# -----------------------------------------------------------------------------
JOINT_LIMITS_DEG: dict[str, dict[str, tuple[float, float] | None]] = {
    "LeftKnee":      {"X": (0, 150),   "Y": (-5, 5),    "Z": (-5, 5)},
    "RightKnee":     {"X": (0, 150),   "Y": (-5, 5),    "Z": (-5, 5)},
    "LeftElbow":     {"X": (0, 150),   "Y": (-5, 5),    "Z": (-5, 5)},
    "RightElbow":    {"X": (0, 150),   "Y": (-5, 5),    "Z": (-5, 5)},
    "LeftHip":       {"X": (-20, 120), "Y": (-45, 45),  "Z": (-30, 45)},
    "RightHip":      {"X": (-20, 120), "Y": (-45, 45),  "Z": (-45, 30)},
    "LeftAnkle":     {"X": (-50, 20),  "Y": (-20, 20),  "Z": (-20, 20)},
    "RightAnkle":    {"X": (-50, 20),  "Y": (-20, 20),  "Z": (-20, 20)},
    "LeftShoulder":  {"X": (-90, 180), "Y": (-90, 90),  "Z": (-90, 90)},
    "RightShoulder": {"X": (-90, 180), "Y": (-90, 90),  "Z": (-90, 90)},
    "Spine":         {"X": (-30, 30),  "Y": (-30, 30),  "Z": (-30, 30)},
    "Chest":         {"X": (-30, 30),  "Y": (-30, 30),  "Z": (-30, 30)},
    "Neck":          {"X": (-45, 45),  "Y": (-60, 60),  "Z": (-40, 40)},
    "Head":          {"X": (-30, 30),  "Y": (-30, 30),  "Z": (-30, 30)},
}


def apply_constraints(skeleton: Skeleton) -> None:
    """
    Recorre todos los frames y joints, clampea ángulos al rango permitido.

    Modifica skeleton.local_rotations in-place.
    """
    assert skeleton.local_rotations is not None

    num_frames = skeleton.local_rotations.shape[0]
    order = BVH_ROTATION_ORDER  # p.ej. "ZXY"

    for joint in skeleton.traverse():
        if joint.is_end_site or joint.is_root():
            continue
        limits = JOINT_LIMITS_DEG.get(joint.name)
        if limits is None:
            continue

        for frame in range(num_frames):
            quat = skeleton.local_rotations[frame, joint.index]
            euler = quat_to_euler(quat, order=order, degrees=True)
            clamped = _clamp_euler(euler, order, limits)
            skeleton.local_rotations[frame, joint.index] = euler_to_quat(
                clamped, order=order, degrees=True
            )


def _clamp_euler(
    euler: np.ndarray,
    order: str,
    limits: dict[str, tuple[float, float] | None],
) -> np.ndarray:
    """
    Clampea cada componente de Euler al rango definido en `limits`.
    El array euler está en el orden dado por `order` (p.ej. ZXY → [Z, X, Y]).
    """
    out = euler.copy()
    for i, axis_char in enumerate(order):
        lim = limits.get(axis_char)
        if lim is None:
            continue
        out[i] = float(np.clip(out[i], lim[0], lim[1]))
    return out
