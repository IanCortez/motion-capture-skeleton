"""
Clase Joint: nodo único de la jerarquía esquelética.

Cada Joint conoce su padre, sus hijos, su offset local (en T-pose) y los
canales que expone en BVH. Las rotaciones locales por frame se guardan
en una matriz separada en Skeleton.motion_data, no aquí.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Joint:
    name: str
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    channels: list[str] = field(default_factory=list)
    parent: Joint | None = None
    children: list[Joint] = field(default_factory=list)
    is_end_site: bool = False
    index: int = -1  # Asignado por Skeleton.finalize()

    def add_child(self, child: Joint) -> None:
        child.parent = self
        self.children.append(child)

    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def __repr__(self) -> str:
        return f"Joint({self.name}, children={len(self.children)})"
