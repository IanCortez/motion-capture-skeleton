"""
Global configuration constants for the motion capture pipeline.

Centraliza valores que se usan en múltiples módulos: framerate de captura,
umbrales del post-proceso, paths por defecto. Cambiar aquí afecta todo el pipeline.
"""

from pathlib import Path


# -----------------------------------------------------------------------------
# Lectura de C3D
# -----------------------------------------------------------------------------
# Eje "arriba" del sistema de coordenadas del C3D de entrada.
#   'Z' → típico de Vicon y muchos sistemas ópticos (se convierte a Y-up).
#   'Y' → si el C3D ya viene en Y-up.
# Las unidades NO se configuran aquí: se leen de POINT:UNITS del archivo.
C3D_UP_AXIS = "Y"

# -----------------------------------------------------------------------------
# T-pose calibration
# -----------------------------------------------------------------------------
TPOSE_DURATION_SEC = 1.0        # Duración de la ventana estática de T-pose
TPOSE_START_FRAME = 0           # Frame inicial de la T-pose

# -----------------------------------------------------------------------------
# FABRIK
# -----------------------------------------------------------------------------
FABRIK_TOLERANCE = 1e-3         # Tolerancia de convergencia (cm)
FABRIK_MAX_ITERATIONS = 40      # Máximo de iteraciones por frame

# -----------------------------------------------------------------------------
# Foot sliding correction
# -----------------------------------------------------------------------------
FOOT_HEIGHT_THRESHOLD = 5.0     # cm sobre el suelo para considerar contacto
FOOT_VELOCITY_THRESHOLD = 5.0   # cm/s
FOOT_CONTACT_MIN_FRAMES = 5     # Frames mínimos para validar contacto
FOOT_BLEND_FRAMES = 6           # Frames de blending en transiciones

# -----------------------------------------------------------------------------
# BVH export
# -----------------------------------------------------------------------------
BVH_ROTATION_ORDER = "ZXY"      # Orden de rotación en CHANNELS
BVH_ROOT_NAME = "Hips"

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
