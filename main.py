"""
Entry point del pipeline completo de la Etapa 3.

Flujo de datos:
  C3D → MarkerTrajectories
            ↓
    T-pose calibration
            ↓
Skeleton con offsets reales
            ↓
Centros articulares por frame
            ↓
FABRIK (corrección de longitudes)
            ↓
Extracción de rotaciones
            ↓
Restricciones biomecánicas
            ↓
Post-proceso (foot, segmentación)
            ↓
        BVH file

Uso:
    python main.py --input data/input/captura.c3d --output data/output/captura.bvh
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

# Asegurar que el package se puede importar al ejecutar como script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import OUTPUT_DIR
from config.marker_set import MARKER_NAMES, JOINT_CENTER_RULES
from src.io.markers_reader import load_markers, MarkerTrajectories
from src.io.bvh_writer import write_bvh
from src.skeleton.hierarchy_builder import build_default_skeleton, IK_CHAINS
from src.skeleton.skeleton import Skeleton
from src.calibration.tpose_calibration import (
    average_tpose_markers,
    compute_joint_centers,
    apply_to_skeleton,
)
from src.calibration.auto_calibration import calibrate_auto
from src.ik.fabrik import solve_chain
from src.rotations.extractor import extract_rotations
from src.rotations.constraints import apply_constraints
from src.postprocess.foot_sliding import correct_foot_sliding
from src.postprocess.segmentation import segment_motion


# =============================================================================
# Estimación de centros articulares por frame
# =============================================================================
def estimate_joint_centers_per_frame(
    markers: MarkerTrajectories,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Aplica las reglas de marker_set a CADA frame para obtener centros articulares.

    Returns
    -------
    positions : (T, M, 3) posiciones globales de cada joint por frame.
    name_to_idx : mapping joint_name → índice M.
    """
    joint_names = list(JOINT_CENTER_RULES.keys())
    name_to_idx = {n: i for i, n in enumerate(joint_names)}

    T = markers.num_frames
    M = len(joint_names)
    out = np.zeros((T, M, 3))

    for f in range(T):
        frame_markers = {name: markers[name][f] for name in markers.marker_names()}
        centers = compute_joint_centers(frame_markers)
        for name, idx in name_to_idx.items():
            if name in centers:
                out[f, idx] = centers[name]

    return out, name_to_idx


# =============================================================================
# Refinamiento con FABRIK por cadena
# =============================================================================
def refine_with_fabrik(
    skeleton: Skeleton,
    positions: np.ndarray,
    name_to_idx: dict[str, int],
) -> np.ndarray:
    """
    Para cada cadena IK y cada frame, ajusta las posiciones de los joints
    para respetar las longitudes de hueso medidas en T-pose.

    Returns
    -------
    Nueva (T, M, 3) con posiciones refinadas.
    """
    refined = positions.copy()
    num_frames = positions.shape[0]

    for chain_name, joint_names in IK_CHAINS.items():
        # Filtrar a joints presentes en name_to_idx
        chain = [n for n in joint_names if n in name_to_idx]
        if len(chain) < 2:
            continue

        # Longitudes de hueso desde el skeleton (T-pose)
        bone_lengths = []
        for child in chain[1:]:
            if child in skeleton.bone_lengths:
                bone_lengths.append(skeleton.bone_lengths[child])
            else:
                bone_lengths.append(0.0)
        bone_lengths = np.array(bone_lengths)
        if (bone_lengths == 0).any():
            continue  # cadena incompleta, omitir

        indices = [name_to_idx[n] for n in chain]

        for f in range(num_frames):
            chain_positions = refined[f, indices, :]
            target = chain_positions[-1]
            root_fixed = chain_positions[0]
            corrected = solve_chain(
                chain_positions, bone_lengths, target=target, root_fixed=root_fixed
            )
            refined[f, indices, :] = corrected

    return refined


# =============================================================================
# Pipeline completo
# =============================================================================
def run_pipeline(
    markers: MarkerTrajectories,
    output_path: Path,
    calibration_mode: str = "tpose",
    auto_reference: str = "first_frame",
) -> Path:
    """
    Pipeline completo: trayectorias 3D → BVH.

    Parameters
    ----------
    markers : trayectorias ya cargadas.
    output_path : destino del .bvh.
    calibration_mode : 'tpose' usa el período estático inicial (lo más preciso si está disponible).
                       'auto'  no requiere T-pose; usa estadística sobre toda la captura.
    auto_reference : sólo se usa si calibration_mode='auto'. Modos disponibles:
                     'first_frame', 'mean_pose', 'auto_neutral'.
    """
    print(f"[1/8] Loaded markers: {markers}")

    print("[2/8] Building skeleton topology...")
    skeleton = build_default_skeleton()

    print(f"[3/8] Calibration (mode='{calibration_mode}')...")
    if calibration_mode == "tpose":
        tpose_avg = average_tpose_markers(markers)
        tpose_centers = compute_joint_centers(tpose_avg)
        apply_to_skeleton(skeleton, tpose_centers)
        rest_centers = tpose_centers
    elif calibration_mode == "auto":
        rest_centers = calibrate_auto(skeleton, markers, reference_mode=auto_reference)
    else:
        raise ValueError(f"calibration_mode debe ser 'tpose' o 'auto', recibido: '{calibration_mode}'")
    print(f"      → {skeleton}")
    print(f"      → bone lengths: {len(skeleton.bone_lengths)} measured")

    print("[4/8] Estimating joint centers per frame...")
    positions, name_to_idx = estimate_joint_centers_per_frame(markers)
    print(f"      → positions array: {positions.shape}")

    print("[5/8] FABRIK refinement (bone length consistency)...")
    positions = refine_with_fabrik(skeleton, positions, name_to_idx)

    print("[6/8] Extracting joint rotations...")
    skeleton.allocate_motion(markers.num_frames)
    # Root position desde los Hips estimados
    if "Hips" in name_to_idx:
        skeleton.root_positions = positions[:, name_to_idx["Hips"], :].copy()
    extract_rotations(skeleton, positions, name_to_idx, rest_centers)

    print("[7/8] Applying biomechanical constraints...")
    apply_constraints(skeleton)

    print("[7b/8] Post-process: foot sliding + segmentation...")
    foot_chains = {"left": IK_CHAINS["left_leg"], "right": IK_CHAINS["right_leg"]}
    foot_world = {}
    if "LeftToe" in name_to_idx:
        foot_world["left"] = positions[:, name_to_idx["LeftToe"], :]
    if "RightToe" in name_to_idx:
        foot_world["right"] = positions[:, name_to_idx["RightToe"], :]
    correct_foot_sliding(skeleton, foot_chains, foot_world, fps=markers.fps)

    segments = segment_motion(skeleton, fps=markers.fps)
    print(f"      → {len(segments)} motion segment(s) detected: {segments}")

    print(f"[8/8] Writing BVH → {output_path}")
    final_path = write_bvh(skeleton, output_path, frame_time=1.0 / markers.fps)
    print(f"      → done: {final_path}")
    return final_path


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Motion capture pipeline (Etapa 3: Trayectorias 3D → BVH)")
    parser.add_argument("--input", type=Path, help="Archivo de trayectorias 3D (.c3d)")
    parser.add_argument("--output", type=Path, help="Archivo BVH de salida")
    parser.add_argument("--frames", type=int, default=240, help="Frames a generar en modo demo")
    parser.add_argument(
        "--calibration", choices=["tpose", "auto"], default="tpose",
        help="'tpose' usa el período estático inicial (default). 'auto' calibra "
             "estadísticamente sin necesidad de T-pose.",
    )
    parser.add_argument(
        "--auto-reference", choices=["first_frame", "mean_pose", "auto_neutral"],
        default="first_frame",
        help="Sólo con --calibration=auto: cómo elegir la pose de referencia.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # load_markers despacha automáticamente según extensión (.c3d).
    # Para C3D el framerate y unidades se leen del propio archivo.
    markers = load_markers(args.input, expected_markers=MARKER_NAMES)
    output_path = args.output or (OUTPUT_DIR / (args.input.stem + ".bvh"))

    # Reporte de oclusiones (útil para diagnosticar calidad de los datos de Kelvin)
    occ = markers.occlusion_report()
    occluded = {k: v for k, v in occ.items() if v > 0}
    if occluded:
        print(f"  [oclusiones] {len(occluded)} marcadores con gaps: "
              f"{ {k: f'{v:.1f}%' for k, v in sorted(occluded.items())} }")

    run_pipeline(
        markers, output_path,
        calibration_mode=args.calibration,
        auto_reference=args.auto_reference,
    )


if __name__ == "__main__":
    main()
