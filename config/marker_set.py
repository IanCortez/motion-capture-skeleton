"""
Definición del marker set Helen Hayes / Plug-in Gait simplificado.

Cada marcador tiene un nombre canónico que se usará en todo el pipeline.
Cambiar aquí el marker set permite adaptar el proyecto a otro estándar
sin tocar los algoritmos.
"""

# -----------------------------------------------------------------------------
# Lista canónica de marcadores
# -----------------------------------------------------------------------------
# Si Kelvin entrega marcadores con otros nombres, mapear aquí:
MARKER_NAMES = [
    # Cabeza
    "LFHD", "RFHD", "LBHD", "RBHD",
    # Tronco
    "C7", "T10", "CLAV", "STRN",
    # Brazo izquierdo
    "LSHO", "LELB", "LWRA", "LWRB", "LFIN",
    # Brazo derecho
    "RSHO", "RELB", "RWRA", "RWRB", "RFIN",
    # Pelvis
    "LASI", "RASI", "LPSI", "RPSI",
    # Pierna izquierda
    "LKNE", "LANK", "LHEE", "LTOE",
    # Pierna derecha
    "RKNE", "RANK", "RHEE", "RTOE",
]

# -----------------------------------------------------------------------------
# Mapeo de labels de C3D → nombres canónicos
# -----------------------------------------------------------------------------
# Los archivos .c3d a veces traen labels con otros nombres o sufijos
# (p.ej. "RShoulder", "R_SHO", "Subject1:RSHO"). El reader ya quita el
# prefijo de sujeto antes de ':'. Aquí se mapea cualquier label restante
# que no coincida con MARKER_NAMES.
#
# Si el C3D de Kelvin ya usa los nombres de MARKER_NAMES, dejar este dict vacío.
# Ejemplo de uso:
#   C3D_LABEL_MAP = {"RShoulder": "RSHO", "L_Knee": "LKNE", ...}
#
# El archivo 0007_*.c3d (set Motion Analysis "Helen Hayes" completo) usa otros
# nombres que se traducen al subset Plug-in-Gait del proyecto. Renombres 1:1:
C3D_LABEL_MAP: dict[str, str] = {
    # Pelvis: front/back waist -> ASIS / PSIS
    "LFWT": "LASI", "RFWT": "RASI",
    "LBWT": "LPSI", "RBWT": "RPSI",
    # Talones
    "LHEL": "LHEE", "RHEL": "RHEE",
    # Muñecas: outer/inner -> A/B (el centro se calcula como su midpoint, así
    # que la asignación A<->B no altera el resultado)
    "LOWR": "LWRA", "LIWR": "LWRB",
    "ROWR": "RWRA", "RIWR": "RWRB",
    # Codos y rodillas laterales ya coinciden (LELB, RELB, LKNE, RKNE),
    # igual que cabeza, tronco, tobillos y dedos del pie (LTOE/RTOE).
}

# -----------------------------------------------------------------------------
# Marcadores derivados (virtuales)
# -----------------------------------------------------------------------------
# Algunos nombres canónicos no tienen equivalente 1:1 en el archivo, sino que
# se reconstruyen combinando marcadores presentes. El reader los calcula tras
# aplicar C3D_LABEL_MAP, usando los nombres tal como vienen en el archivo.
# Solo se crean si NO existe ya un marcador con ese nombre canónico y si están
# presentes todas sus fuentes.
#
# Métodos: "midpoint" (promedio, ignora NaN) y "single" (copia la 1ª fuente).
DERIVED_MARKERS: dict[str, dict] = {
    # Hombro Plug-in-Gait (acromion) ≈ punto medio entre hombro frontal y dorsal
    "LSHO": {"method": "midpoint", "from": ["LFSH", "LBSH"]},
    "RSHO": {"method": "midpoint", "from": ["RFSH", "RBSH"]},
    # Marcador de mano/dedo ≈ punto medio entre las marcas interna y externa
    "LFIN": {"method": "midpoint", "from": ["LIHAND", "LOHAND"]},
    "RFIN": {"method": "midpoint", "from": ["RIHAND", "ROHAND"]},
}

# -----------------------------------------------------------------------------
# Clusters: qué marcadores definen cada centro articular
# -----------------------------------------------------------------------------
# Estos clusters se usan en tpose_calibration.compute_joint_centers().
# Cada entrada describe cómo derivar el centro articular a partir de marcadores.
#
# Métodos soportados:
#   "midpoint"  → promedio simple de los marcadores listados
#   "harrington" → regresión Harrington (solo para caderas)
#   "single"    → copia directa del marcador (caso degenerado)
#
JOINT_CENTER_RULES = {
    "Hips":          {"method": "midpoint",   "markers": ["LASI", "RASI", "LPSI", "RPSI"]},
    "LeftHip":       {"method": "harrington", "side": "left"},
    "RightHip":      {"method": "harrington", "side": "right"},
    "LeftKnee":      {"method": "single",     "markers": ["LKNE"]},
    "LeftAnkle":     {"method": "single",     "markers": ["LANK"]},
    "LeftToe":       {"method": "single",     "markers": ["LTOE"]},
    "RightKnee":     {"method": "single",     "markers": ["RKNE"]},
    "RightAnkle":    {"method": "single",     "markers": ["RANK"]},
    "RightToe":      {"method": "single",     "markers": ["RTOE"]},
    "Spine":         {"method": "midpoint",   "markers": ["T10", "STRN"]},
    "Chest":         {"method": "midpoint",   "markers": ["C7", "CLAV"]},
    "Neck":          {"method": "single",     "markers": ["C7"]},
    "Head":          {"method": "midpoint",   "markers": ["LFHD", "RFHD", "LBHD", "RBHD"]},
    "LeftShoulder":  {"method": "single",     "markers": ["LSHO"]},
    "LeftElbow":     {"method": "single",     "markers": ["LELB"]},
    "LeftWrist":     {"method": "midpoint",   "markers": ["LWRA", "LWRB"]},
    "LeftHand":      {"method": "single",     "markers": ["LFIN"]},
    "RightShoulder": {"method": "single",     "markers": ["RSHO"]},
    "RightElbow":    {"method": "single",     "markers": ["RELB"]},
    "RightWrist":    {"method": "midpoint",   "markers": ["RWRA", "RWRB"]},
    "RightHand":     {"method": "single",     "markers": ["RFIN"]},
}