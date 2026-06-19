"""
Paso 6.3 — FABRIK: Forward And Backward Reaching Inverse Kinematics.

Implementación basada en:
  Aristidou & Lasenby, "FABRIK: A fast, iterative solver for the Inverse
  Kinematics problem", Graphical Models, 2011.

Uso típico en este pipeline:
  Por cada frame, ya tienes estimaciones directas de cada centro articular
  desde los marcadores. Esas estimaciones no respetarán exactamente las
  longitudes de hueso medidas en T-pose (ruido, deformación de piel).
  FABRIK proyecta esas posiciones a una configuración válida en longitudes.

API principal:
  solve_chain(positions, bone_lengths, target, max_iter, tol) -> positions
"""

from __future__ import annotations
import numpy as np

from config.settings import FABRIK_TOLERANCE, FABRIK_MAX_ITERATIONS


# =============================================================================
# Solver para cadena simple (no ramificada)
# =============================================================================
def solve_chain(
    positions: np.ndarray,
    bone_lengths: np.ndarray,
    target: np.ndarray,
    root_fixed: np.ndarray | None = None,
    max_iter: int = FABRIK_MAX_ITERATIONS,
    tol: float = FABRIK_TOLERANCE,
) -> np.ndarray:
    """
    Resuelve una cadena de N joints para que el end-effector alcance `target`,
    respetando bone_lengths.

    Parameters
    ----------
    positions : (N, 3) posiciones iniciales de los joints (incluye end-effector).
    bone_lengths : (N-1,) longitudes objetivo de cada hueso.
    target : (3,) posición deseada del end-effector.
    root_fixed : (3,) posición fija del root. Por defecto = positions[0].
    max_iter : máximo de iteraciones.
    tol : tolerancia de convergencia.

    Returns
    -------
    (N, 3) posiciones corregidas.
    """
    positions = positions.copy().astype(float)
    n = len(positions)
    root = root_fixed if root_fixed is not None else positions[0].copy()

    total_length = bone_lengths.sum()
    dist_to_target = np.linalg.norm(target - root)

    # Caso unreachable: estirar la cadena hacia el target
    if dist_to_target > total_length:
        direction = (target - root) / (dist_to_target + 1e-12)
        positions[0] = root
        for i in range(n - 1):
            positions[i + 1] = positions[i] + direction * bone_lengths[i]
        return positions

    # Caso reachable: iterar forward + backward
    for _ in range(max_iter):
        # ----- Forward pass: end-effector → root -----
        positions[-1] = target
        for i in range(n - 2, -1, -1):
            r = np.linalg.norm(positions[i + 1] - positions[i])
            lam = bone_lengths[i] / (r + 1e-12)
            positions[i] = (1 - lam) * positions[i + 1] + lam * positions[i]

        # ----- Backward pass: root → end-effector -----
        positions[0] = root
        for i in range(n - 1):
            r = np.linalg.norm(positions[i + 1] - positions[i])
            lam = bone_lengths[i] / (r + 1e-12)
            positions[i + 1] = (1 - lam) * positions[i] + lam * positions[i + 1]

        if np.linalg.norm(positions[-1] - target) < tol:
            break

    return positions


# =============================================================================
# Solver con sub-base centroide (para esqueletos ramificados)
# =============================================================================
def solve_branched(
    chains_positions: list[np.ndarray],
    chains_bone_lengths: list[np.ndarray],
    targets: list[np.ndarray],
    subbase_position: np.ndarray,
    max_iter: int = FABRIK_MAX_ITERATIONS,
    tol: float = FABRIK_TOLERANCE,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Resuelve N cadenas que comparten una sub-base (p.ej. brazos compartiendo Chest,
    o piernas compartiendo Hips). Sigue el método de la sección 4.2 del paper.

    Parameters
    ----------
    chains_positions : lista de (N_i, 3) por cada cadena. positions[0] es la subbase.
    chains_bone_lengths : lista de (N_i - 1,) por cadena.
    targets : lista de (3,) end-effector target por cadena.
    subbase_position : (3,) posición inicial de la sub-base.

    Returns
    -------
    (lista de (N_i, 3) corregidas, sub-base position final)
    """
    chains_positions = [p.copy().astype(float) for p in chains_positions]
    subbase = subbase_position.copy()

    for _ in range(max_iter):
        # Cada cadena hace forward (target → subbase) y deja libre su raíz
        candidate_subbases = []
        for i, (pos, lens, tgt) in enumerate(zip(chains_positions, chains_bone_lengths, targets)):
            pos[-1] = tgt
            for j in range(len(pos) - 2, -1, -1):
                r = np.linalg.norm(pos[j + 1] - pos[j])
                lam = lens[j] / (r + 1e-12)
                pos[j] = (1 - lam) * pos[j + 1] + lam * pos[j]
            candidate_subbases.append(pos[0].copy())
            chains_positions[i] = pos

        # Promediar las posiciones de subbase propuestas por cada cadena
        new_subbase = np.mean(candidate_subbases, axis=0)

        # Backward: cada cadena desde la nueva subbase hacia su target
        max_error = 0.0
        for i, (pos, lens, tgt) in enumerate(zip(chains_positions, chains_bone_lengths, targets)):
            pos[0] = new_subbase
            for j in range(len(pos) - 1):
                r = np.linalg.norm(pos[j + 1] - pos[j])
                lam = lens[j] / (r + 1e-12)
                pos[j + 1] = (1 - lam) * pos[j] + lam * pos[j + 1]
            chains_positions[i] = pos
            max_error = max(max_error, np.linalg.norm(pos[-1] - tgt))

        subbase = new_subbase
        if max_error < tol:
            break

    return chains_positions, subbase
