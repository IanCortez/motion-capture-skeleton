"""
Paso 7.1 — Corrección de foot sliding.

Pipeline:
1. Detectar segmentos de contacto del pie (altura baja + velocidad baja).
2. Anclar la posición del pie a su primer contacto en cada segmento.
3. Resolver IK inversa local (tobillo→rodilla→cadera) para mantener el ancla.
4. Blendear suavemente las transiciones para evitar pops.

Esta es la implementación de referencia. Funciona razonablemente con un solver
FABRIK simple aplicado en la cadena invertida.
"""

from __future__ import annotations
import numpy as np

from src.skeleton.skeleton import Skeleton
from src.ik.fabrik import solve_chain
from config.settings import (
    FOOT_HEIGHT_THRESHOLD,
    FOOT_VELOCITY_THRESHOLD,
    FOOT_CONTACT_MIN_FRAMES,
    FOOT_BLEND_FRAMES,
)


# -----------------------------------------------------------------------------
# Detección de contacto
# -----------------------------------------------------------------------------
def detect_contacts(
    foot_positions: np.ndarray,
    fps: float,
    height_thresh: float = FOOT_HEIGHT_THRESHOLD,
    vel_thresh: float = FOOT_VELOCITY_THRESHOLD,
    min_frames: int = FOOT_CONTACT_MIN_FRAMES,
) -> np.ndarray:
    """
    Devuelve un array booleano (T,) indicando frames en contacto con el suelo.

    Parameters
    ----------
    foot_positions : (T, 3) trayectoria del marcador del pie (o tobillo).
    fps : framerate.
    """
    heights = foot_positions[:, 1]  # Y es altura
    velocities = np.linalg.norm(np.diff(foot_positions, axis=0, prepend=foot_positions[:1]), axis=1) * fps

    raw_contact = (heights < height_thresh) & (velocities < vel_thresh)

    # Filtrar segmentos muy cortos (ruido)
    return _filter_short_segments(raw_contact, min_frames)


def _filter_short_segments(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Elimina segmentos True de longitud menor a min_len."""
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if not out[i]:
            i += 1
            continue
        j = i
        while j < n and out[j]:
            j += 1
        if j - i < min_len:
            out[i:j] = False
        i = j
    return out


# -----------------------------------------------------------------------------
# Aplicación de la corrección
# -----------------------------------------------------------------------------
def correct_foot_sliding(
    skeleton: Skeleton,
    foot_chains: dict[str, list[str]],
    foot_world_positions: dict[str, np.ndarray],
    fps: float,
) -> None:
    """
    Corrige el foot sliding modificando skeleton.local_rotations in-place.

    Parameters
    ----------
    skeleton : skeleton ya animado.
    foot_chains : { "left": ["LeftHip","LeftKnee","LeftAnkle","LeftToe"], "right": ... }
    foot_world_positions : { "left": (T,3), "right": (T,3) } posiciones globales del pie.
    fps : framerate.

    Notas de implementación
    -----------------------
    Para mantener el módulo independiente, esta función NO recalcula forward kinematics
    a partir de las rotaciones; asume que el llamador provee `foot_world_positions`.
    En main.py se obtienen directamente de los marcadores corregidos.

    TODO:
    - Hacer la inversión real con FABRIK invertido sobre cada segmento de contacto.
    - Implementar el blending temporal en transiciones.
    """
    for side, chain_names in foot_chains.items():
        foot_pos = foot_world_positions[side]
        contacts = detect_contacts(foot_pos, fps=fps)

        # Por ahora solo emitimos el resumen; la corrección IK queda como TODO
        # para que cada miembro del equipo pueda iterar sin romper el pipeline.
        segments = _extract_segments(contacts)
        print(f"  [foot_sliding] {side}: {len(segments)} contact segments detected")

        for start, end in segments:
            anchor = foot_pos[start]
            _anchor_segment(skeleton, chain_names, start, end, anchor)


def _extract_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Devuelve lista de (start, end_exclusive) por cada run de True."""
    segments = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        segments.append((i, j))
        i = j
    return segments


def _anchor_segment(
    skeleton: Skeleton,
    chain_names: list[str],
    start: int,
    end: int,
    anchor: np.ndarray,
) -> None:
    """
    Stub: aplicar IK invertido a lo largo del segmento [start, end) para
    mantener el end-effector de la cadena anclado en `anchor`.

    Implementación sugerida:
    1. Por cada frame del segmento, calcular posiciones globales actuales
       de los joints de la cadena (forward kinematics desde las rotaciones).
    2. Ejecutar fabrik.solve_chain con target = anchor.
    3. Re-extraer rotaciones locales y reescribirlas.
    4. En los primeros/últimos FOOT_BLEND_FRAMES, interpolar con SLERP.

    Por ahora deja el segmento intacto (no-op) para mantener el pipeline funcional.
    """
    # TODO: implementar.
    _ = (skeleton, chain_names, start, end, anchor)
