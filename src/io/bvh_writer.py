"""
Escritor de archivos BVH (Biovision Hierarchy).

Toma un Skeleton ya animado (root_positions + local_rotations en cuaterniones)
y lo serializa al formato BVH estándar. Es el output final de la etapa 3.

Formato BVH:
    HIERARCHY        ← topología + offsets + canales
    MOTION           ← per-frame: root_pos + todas las rotaciones locales
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

from src.skeleton.skeleton import Skeleton
from src.skeleton.joint import Joint
from src.utils.math_utils import quat_to_euler
from config.settings import BVH_ROTATION_ORDER


# =============================================================================
# Serialización de la HIERARCHY
# =============================================================================
def _write_hierarchy(skeleton: Skeleton) -> str:
    lines = ["HIERARCHY"]
    _write_joint(skeleton.root, lines, indent=0, is_root=True)
    return "\n".join(lines)


def _write_joint(joint: Joint, lines: list[str], indent: int, is_root: bool = False) -> None:
    pad = "\t" * indent

    if joint.is_end_site:
        lines.append(f"{pad}End Site")
        lines.append(f"{pad}{{")
        lines.append(f"{pad}\tOFFSET {_fmt_vec(joint.offset)}")
        lines.append(f"{pad}}}")
        return

    keyword = "ROOT" if is_root else "JOINT"
    lines.append(f"{pad}{keyword} {joint.name}")
    lines.append(f"{pad}{{")
    lines.append(f"{pad}\tOFFSET {_fmt_vec(joint.offset)}")
    lines.append(f"{pad}\tCHANNELS {len(joint.channels)} {' '.join(joint.channels)}")

    for child in joint.children:
        _write_joint(child, lines, indent + 1, is_root=False)

    lines.append(f"{pad}}}")


def _fmt_vec(v: np.ndarray) -> str:
    return f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"


# =============================================================================
# Serialización de la MOTION
# =============================================================================
def _write_motion(skeleton: Skeleton, frame_time: float) -> str:
    assert skeleton.root_positions is not None, "No hay motion data. Llamar a allocate_motion() primero."
    assert skeleton.local_rotations is not None

    # Saneamiento final: ningún NaN/inf debe llegar al BVH. Un NaN en la
    # traslación de la raíz hace DESAPARECER todo el esqueleto en ese frame
    # (el visor no puede ubicarlo); un NaN en una rotación borra el subárbol
    # de ese joint. Se trabaja sobre copias para no mutar el skeleton.
    root_positions = _sanitize_positions(skeleton.root_positions)
    local_rotations = _sanitize_rotations(skeleton.local_rotations)

    num_frames = root_positions.shape[0]
    lines = [
        "MOTION",
        f"Frames: {num_frames}",
        f"Frame Time: {frame_time:.6f}",
    ]

    for frame_idx in range(num_frames):
        values: list[float] = []
        for joint in skeleton.traverse():
            if joint.is_end_site:
                continue
            # Root: traslación + rotación
            if joint.is_root():
                pos = root_positions[frame_idx]
                values.extend([pos[0], pos[1], pos[2]])
            # Rotación local en Euler (mismo orden que CHANNELS)
            quat = local_rotations[frame_idx, joint.index]
            euler = quat_to_euler(quat, order=BVH_ROTATION_ORDER, degrees=True)
            values.extend(euler.tolist())
        lines.append(" ".join(f"{v:.6f}" for v in values))

    return "\n".join(lines)


_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0])


def _sanitize_positions(root_positions: np.ndarray) -> np.ndarray:
    """
    Devuelve una copia de las traslaciones de la raíz sin NaN/inf.

    Cada eje se rellena por interpolación temporal sobre los frames válidos
    (los extremos se sostienen). Si un eje no tiene ningún valor finito, se
    pone a 0 para que el esqueleto quede en el origen en lugar de desaparecer.
    """
    pos = np.asarray(root_positions, dtype=float).copy()
    pos[~np.isfinite(pos)] = np.nan
    if not np.isnan(pos).any():
        return pos

    num_frames = pos.shape[0]
    t = np.arange(num_frames)
    n_bad = int(np.isnan(pos).any(axis=1).sum())
    for a in range(pos.shape[1]):
        col = pos[:, a]
        valid = ~np.isnan(col)
        if valid.all():
            continue
        if not valid.any():
            pos[:, a] = 0.0
        else:
            pos[:, a] = np.interp(t, t[valid], col[valid])
    print(f"  [write_bvh] AVISO: {n_bad} frame(s) con traslación de raíz no "
          f"finita; rellenados por interpolación para no perder el esqueleto.")
    return pos


def _sanitize_rotations(local_rotations: np.ndarray) -> np.ndarray:
    """
    Devuelve una copia de las rotaciones locales sin NaN/inf.

    Para cada joint, un cuaternión no finito (o de norma ~0) se reemplaza por
    el último válido del propio joint (mantiene la pose); si aún no hubo uno
    válido, se usa la identidad. Así ningún subárbol desaparece en el visor.
    """
    rot = np.asarray(local_rotations, dtype=float).copy()
    finite = np.isfinite(rot).all(axis=2)
    norms = np.linalg.norm(np.nan_to_num(rot), axis=2)
    valid = finite & (norms > 1e-8)
    if valid.all():
        return rot

    num_frames, num_joints, _ = rot.shape
    n_bad = int((~valid).sum())
    for j in range(num_joints):
        last = _IDENTITY_QUAT
        for f in range(num_frames):
            if valid[f, j]:
                last = rot[f, j]
            else:
                rot[f, j] = last
    print(f"  [write_bvh] AVISO: {n_bad} rotación(es) no finita(s) saneadas "
          f"(se sostuvo la última pose válida) para no perder joints.")
    return rot


# =============================================================================
# API pública
# =============================================================================
def write_bvh(
    skeleton: Skeleton,
    output_path: Path | str,
    frame_time: float
) -> Path:
    """
    Escribe el skeleton animado a un archivo BVH.

    Parameters
    ----------
    skeleton : Skeleton con motion data ya rellenada.
    output_path : ruta del archivo .bvh a crear.
    frame_time : segundos entre frames (1/fps).

    Returns
    -------
    Path absoluto al archivo escrito.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hierarchy = _write_hierarchy(skeleton)
    motion = _write_motion(skeleton, frame_time)

    with open(output_path, "w") as f:
        f.write(hierarchy + "\n" + motion + "\n")

    return output_path.resolve()