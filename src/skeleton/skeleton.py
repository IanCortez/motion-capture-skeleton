"""
Clase Skeleton: árbol completo de joints + datos de movimiento.

Centraliza:
- La jerarquía (raíz + descendientes accesibles vía traversal).
- Las longitudes de hueso medidas en T-pose (inviolables).
- La data de movimiento por frame:
    * root_positions: (T, 3)
    * local_rotations: (T, N_joints, 4)   en cuaterniones
"""

from __future__ import annotations
from typing import Iterator
import numpy as np

from .joint import Joint


class Skeleton:
    def __init__(self, root: Joint):
        self.root = root
        self.joints: list[Joint] = []
        self.bone_lengths: dict[str, float] = {}
        self.root_positions: np.ndarray | None = None      # (T, 3)
        self.local_rotations: np.ndarray | None = None     # (T, N, 4)
        self._finalize()

    # ------------------------------------------------------------------ #
    # Traversal
    # ------------------------------------------------------------------ #
    def traverse(self) -> Iterator[Joint]:
        """Preorder DFS. Es el orden que usa BVH para canales."""
        yield from self._traverse_recursive(self.root)

    def _traverse_recursive(self, joint: Joint) -> Iterator[Joint]:
        yield joint
        for child in joint.children:
            yield from self._traverse_recursive(child)

    def get_joint(self, name: str) -> Joint:
        for joint in self.traverse():
            if joint.name == name:
                return joint
        raise KeyError(f"Joint '{name}' not found in skeleton")

    def num_joints(self) -> int:
        """Número de joints reales (excluye End Sites)."""
        return sum(1 for j in self.traverse() if not j.is_end_site)

    # ------------------------------------------------------------------ #
    # Inicialización
    # ------------------------------------------------------------------ #
    def _finalize(self) -> None:
        """Indexa los joints en orden de traversal."""
        idx = 0
        self.joints = []
        for j in self.traverse():
            if not j.is_end_site:
                j.index = idx
                idx += 1
            self.joints.append(j)

    def allocate_motion(self, num_frames: int) -> None:
        """Reserva memoria para los datos de animación."""
        n = self.num_joints()
        self.root_positions = np.zeros((num_frames, 3))
        # Inicializa con identidad (x=0, y=0, z=0, w=1)
        self.local_rotations = np.tile(
            np.array([0.0, 0.0, 0.0, 1.0]),
            (num_frames, n, 1),
        )

    # ------------------------------------------------------------------ #
    # Setters convenientes
    # ------------------------------------------------------------------ #
    def set_bone_length(self, joint_name: str, length: float) -> None:
        self.bone_lengths[joint_name] = length

    def set_offset(self, joint_name: str, offset: np.ndarray) -> None:
        self.get_joint(joint_name).offset = np.asarray(offset, dtype=float)

    def __repr__(self) -> str:
        return f"Skeleton({self.num_joints()} joints, root={self.root.name})"
