"""
Construcción de la jerarquía esquelética estándar.

Topología fija de 21 joints reales + End Sites en cada cadena terminal.
La forma exacta del esqueleto (offsets, longitudes de hueso) se inyecta
después desde la T-pose en `tpose_calibration.py`. Aquí solo definimos
la TOPOLOGÍA — quién es padre de quién y qué canales tiene cada uno.
"""

import numpy as np

from .joint import Joint
from .skeleton import Skeleton
from config.settings import BVH_ROTATION_ORDER

ROOT_CHANNELS = ["Xposition", "Yposition", "Zposition"] + [
    f"{ax}rotation" for ax in BVH_ROTATION_ORDER
]
JOINT_CHANNELS = [f"{ax}rotation" for ax in BVH_ROTATION_ORDER]


def build_default_skeleton() -> Skeleton:
    """
    Construye el esqueleto canónico que usaremos en todo el proyecto.

    Los offsets se inicializan a cero; se llenarán en tpose_calibration
    una vez que tengamos las longitudes reales del sujeto.
    """
    hips = Joint("Hips", channels=ROOT_CHANNELS)

    # Columna
    spine = Joint("Spine", channels=JOINT_CHANNELS)
    chest = Joint("Chest", channels=JOINT_CHANNELS)
    neck = Joint("Neck", channels=JOINT_CHANNELS)
    head = Joint("Head", channels=JOINT_CHANNELS)
    head_end = Joint("HeadEnd", is_end_site=True)

    hips.add_child(spine)
    spine.add_child(chest)
    chest.add_child(neck)
    neck.add_child(head)
    head.add_child(head_end)

    # Brazo izquierdo
    l_sho = Joint("LeftShoulder", channels=JOINT_CHANNELS)
    l_elb = Joint("LeftElbow", channels=JOINT_CHANNELS)
    l_wri = Joint("LeftWrist", channels=JOINT_CHANNELS)
    l_hand_end = Joint("LeftHandEnd", is_end_site=True)
    chest.add_child(l_sho)
    l_sho.add_child(l_elb)
    l_elb.add_child(l_wri)
    l_wri.add_child(l_hand_end)

    # Brazo derecho
    r_sho = Joint("RightShoulder", channels=JOINT_CHANNELS)
    r_elb = Joint("RightElbow", channels=JOINT_CHANNELS)
    r_wri = Joint("RightWrist", channels=JOINT_CHANNELS)
    r_hand_end = Joint("RightHandEnd", is_end_site=True)
    chest.add_child(r_sho)
    r_sho.add_child(r_elb)
    r_elb.add_child(r_wri)
    r_wri.add_child(r_hand_end)

    # Pierna izquierda
    l_hip = Joint("LeftHip", channels=JOINT_CHANNELS)
    l_knee = Joint("LeftKnee", channels=JOINT_CHANNELS)
    l_ank = Joint("LeftAnkle", channels=JOINT_CHANNELS)
    l_toe = Joint("LeftToe", channels=JOINT_CHANNELS)
    l_toe_end = Joint("LeftToeEnd", is_end_site=True)
    hips.add_child(l_hip)
    l_hip.add_child(l_knee)
    l_knee.add_child(l_ank)
    l_ank.add_child(l_toe)
    l_toe.add_child(l_toe_end)

    # Pierna derecha
    r_hip = Joint("RightHip", channels=JOINT_CHANNELS)
    r_knee = Joint("RightKnee", channels=JOINT_CHANNELS)
    r_ank = Joint("RightAnkle", channels=JOINT_CHANNELS)
    r_toe = Joint("RightToe", channels=JOINT_CHANNELS)
    r_toe_end = Joint("RightToeEnd", is_end_site=True)
    hips.add_child(r_hip)
    r_hip.add_child(r_knee)
    r_knee.add_child(r_ank)
    r_ank.add_child(r_toe)
    r_toe.add_child(r_toe_end)

    return Skeleton(hips)


# -----------------------------------------------------------------------------
# Cadenas IK: qué subcadenas resuelve FABRIK
# -----------------------------------------------------------------------------
# Cada cadena va desde raíz de la subcadena hasta su end-effector.
# FABRIK las resuelve independientemente; la raíz pélvica actúa de subbase.
IK_CHAINS = {
    "spine":      ["Hips", "Spine", "Chest", "Neck", "Head"],
    "left_arm":   ["Chest", "LeftShoulder", "LeftElbow", "LeftWrist"],
    "right_arm":  ["Chest", "RightShoulder", "RightElbow", "RightWrist"],
    "left_leg":   ["Hips", "LeftHip", "LeftKnee", "LeftAnkle", "LeftToe"],
    "right_leg":  ["Hips", "RightHip", "RightKnee", "RightAnkle", "RightToe"],
}
