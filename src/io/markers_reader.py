"""
Lector de trayectorias 3D de marcadores.

Soporta tres fuentes:
  - C3D  (formato estándar de motion capture / biomecánica)  ← entrada real del proyecto
  - Sintético (generado en memoria, para tests end-to-end)

El dispatcher `load_markers()` elige el reader según la extensión del archivo.

Sobre el formato C3D
--------------------
Un archivo .c3d es binario y trae su propia metadata, que el reader extrae
en vez de depender de config.settings:
  - POINT:RATE   → framerate de captura (Hz)
  - POINT:LABELS → nombres de los marcadores
  - POINT:UNITS  → unidades (mm / cm / m); se convierten internamente a cm
  - residual de cada punto: un valor < 0 indica marcador OCLUIDO en ese frame;
    esos puntos se marcan como NaN para que el resto del pipeline los maneje.
"""

from __future__ import annotations
import re
import difflib
from pathlib import Path
import numpy as np
import pandas as pd

from config.marker_set import C3D_LABEL_MAP, MARKER_NAMES
from config.settings import C3D_UP_AXIS

try:
    from config.marker_set import DERIVED_MARKERS
except ImportError:        # retrocompatible si aún no está definido
    DERIVED_MARKERS: dict[str, dict] = {}


# =============================================================================
# Contenedor de trayectorias
# =============================================================================
class MarkerTrajectories:
    """Contenedor de trayectorias de marcadores. Posiciones siempre en cm, Y-up."""

    def __init__(self, data: dict[str, np.ndarray], fps: float):
        self.data = data
        self.fps = fps
        self.num_frames = next(iter(data.values())).shape[0] if data else 0

    def __getitem__(self, marker_name: str) -> np.ndarray:
        return self.data[marker_name]

    def __contains__(self, marker_name: str) -> bool:
        return marker_name in self.data

    def marker_names(self) -> list[str]:
        return list(self.data.keys())

    def occlusion_report(self) -> dict[str, float]:
        """Porcentaje de frames ocluidos (NaN) por marcador."""
        return {
            name: float(np.isnan(traj).any(axis=1).mean() * 100.0)
            for name, traj in self.data.items()
        }

    def __repr__(self) -> str:
        return (f"MarkerTrajectories({len(self.data)} markers × "
                f"{self.num_frames} frames @ {self.fps}Hz)")


# =============================================================================
# Dispatcher por extensión
# =============================================================================
def load_markers(
    path: Path | str,
    expected_markers: list[str] | None = None
) -> MarkerTrajectories:
    """
    Carga trayectorias eligiendo el reader según la extensión del archivo.

    Parameters
    ----------
    path : ruta al archivo (.c3d).
    fps  : framerate. Para C3D se IGNORA (se lee del archivo). Para es requerido.
    expected_markers : si se pasa, valida que estén presentes (tras el mapeo de labels).
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".c3d":
        return read_c3d(path, expected_markers=expected_markers)
    raise ValueError(f"Extensión no soportada: '{ext}'. Usar .c3d")


# =============================================================================
# Lectura de C3D
# =============================================================================
_UNIT_TO_CM = {"mm": 0.1, "millimeter": 0.1, "cm": 1.0, "centimeter": 1.0,
               "m": 100.0, "meter": 100.0}


def read_c3d(path: Path | str, expected_markers: list[str] | None = None,
             label_map: dict[str, str] | None = None,
             derived_markers: dict[str, dict] | None = None) -> MarkerTrajectories:
    """
    Lee un archivo .c3d y devuelve trayectorias en cm, sistema Y-up.

    Parameters
    ----------
    path : ruta al .c3d.
    expected_markers : valida presencia de estos marcadores tras el mapeo.
    label_map : mapeo { label_en_c3d : nombre_canonico }. Por defecto usa
                config.marker_set.C3D_LABEL_MAP.

    Notas
    -----
    - Usa ezc3d (primario) o el paquete c3d puro-Python (fallback).
    - Convierte unidades a cm según POINT:UNITS.
    - Marca como NaN los frames con residual negativo (marcador ocluido).
    - Aplica la conversión de ejes definida en settings.C3D_UP_AXIS.
    """
    path = Path(path)
    label_map = label_map if label_map is not None else C3D_LABEL_MAP
    derived_markers = (derived_markers if derived_markers is not None
                       else DERIVED_MARKERS)

    raw_data, fps, units = _read_c3d_backend(path)

    scale = _UNIT_TO_CM.get(units.lower().strip(), 1.0)
    if units.lower().strip() not in _UNIT_TO_CM:
        print(f"  [read_c3d] AVISO: unidad '{units}' desconocida, asumiendo cm.")

    # 1) Limpia los labels: normaliza espacios/tabs y quita el prefijo de
    #    sujeto/namespace ('Subject0007:LFHD' -> 'LFHD').
    cleaned = _clean_labels(raw_data)

    # 2) Reensambla canales separados por eje. Algunos archivos traen un
    #    "punto" por componente ('LFHD X', 'LFHD Y', 'LFHD Z') en vez de un
    #    único punto 3D; aquí se recombinan en un marcador (n_frames, 3).
    cleaned = _reassemble_axis_split_channels(cleaned)

    # 3) Escala a cm, convierte ejes y aplica el nombre canónico.
    data: dict[str, np.ndarray] = {}
    mapping_report: dict[str, str] = {}
    for label, traj in cleaned.items():
        canonical = _canonical_label(label, label_map)
        mapping_report[label] = canonical
        traj = traj * scale                       # → cm
        traj = _apply_axis_convention(traj, C3D_UP_AXIS)
        if canonical in data:
            print(f"  [read_c3d] AVISO: '{label}' y otro label mapean ambos a "
                  f"'{canonical}'. Se conserva el primero.")
            continue
        data[canonical] = traj

    # 4) Marcadores derivados (p.ej. LSHO = midpoint(LFSH, LBSH)) para nombres
    #    canónicos que no tienen equivalente 1:1 en el archivo.
    _apply_derived_markers(data, derived_markers)

    markers = MarkerTrajectories(data, fps=fps)

    if expected_markers:
        missing = [m for m in expected_markers if m not in data]
        if missing:
            raise ValueError(
                _format_missing_markers_error(
                    path, missing, data, raw_data, mapping_report
                )
            )

    return markers


def _read_c3d_backend(path: Path) -> tuple[dict[str, np.ndarray], float, str]:
    """
    Lee el C3D crudo. Devuelve (data_dict, fps, units).
    data_dict mapea label_original → (n_frames, 3) con NaN en oclusiones.
    """
    try:
        return _read_c3d_ezc3d(path)
    except ImportError:
        return _read_c3d_pure(path)


def _read_c3d_ezc3d(path: Path) -> tuple[dict[str, np.ndarray], float, str]:
    """Backend ezc3d (preferido: robusto y rápido)."""
    import ezc3d

    c = ezc3d.c3d(str(path))
    point_params = c["parameters"]["POINT"]

    fps = float(point_params["RATE"]["value"][0])
    labels = list(point_params["LABELS"]["value"])
    units_list = point_params.get("UNITS", {}).get("value", ["mm"])
    units = units_list[0] if units_list else "mm"

    # points shape: (4, n_markers, n_frames). La 4ª fila es coordenada
    # homogénea, NO el residual. El residual real está en meta_points.
    points = np.asarray(c["data"]["points"])
    n_markers = points.shape[1]
    n_frames = points.shape[2]

    # Residuales: meta_points['residuals'] tiene shape (1, n_markers, n_frames).
    # Un residual < 0 indica marcador ocluido/inválido en ese frame.
    residuals = np.asarray(c["data"]["meta_points"]["residuals"])
    has_residuals = residuals.size == n_markers * n_frames

    data: dict[str, np.ndarray] = {}
    for i, label in enumerate(labels):
        traj = points[0:3, i, :].T.copy()         # (n_frames, 3)
        if has_residuals:
            traj[residuals[0, i, :] < 0] = np.nan
        # Si no hay info de residuales, se respetan los NaN ya presentes.
        data[label] = traj

    return data, fps, units


def _read_c3d_pure(path: Path) -> tuple[dict[str, np.ndarray], float, str]:
    """Backend c3d puro-Python (fallback si ezc3d no está instalado)."""
    import c3d as c3dlib

    with open(path, "rb") as handle:
        reader = c3dlib.Reader(handle)
        fps = float(reader.point_rate)
        labels = [lbl.strip() for lbl in reader.point_labels]

        # Unidades: POINT:UNITS
        try:
            units = reader.get("POINT:UNITS").string_value.strip()
        except Exception:
            units = "mm"

        frames = []
        for _, points, _ in reader.read_frames():
            # points: (n_markers, 5) → X, Y, Z, residual, camera
            frame = points[:, :3].copy()
            residual = points[:, 3]
            frame[residual < 0] = np.nan
            frames.append(frame)

    arr = np.stack(frames, axis=0)                # (n_frames, n_markers, 3)
    data = {labels[i]: arr[:, i, :] for i in range(len(labels))}
    return data, fps, units


# =============================================================================
# Normalización de labels
# =============================================================================
# Detecta un sufijo de eje al final del label: "LFHD X", "LFHD_Y", "LFHD-Z".
# Exige un separador (espacio/guion/underscore) antes de la letra para no
# recortar marcadores cuyo nombre acabe en X/Y/Z sin separador.
_AXIS_SUFFIX_RE = re.compile(r"^(.+?)[\s_\-]([XYZ])$", re.IGNORECASE)


def _normalize_ws(label: str) -> str:
    """Colapsa tabs/espacios múltiples a un único espacio y recorta extremos."""
    return " ".join(str(label).split())


def _strip_namespace(label: str) -> str:
    """Quita el prefijo de sujeto/namespace: 'Subject0007:LFHD' -> 'LFHD'."""
    if ":" in label:
        label = label.rsplit(":", 1)[-1].strip()
    return label


def _split_axis_suffix(label: str) -> tuple[str, str | None]:
    """('LFHD X') -> ('LFHD', 'X'); ('LFHD') -> ('LFHD', None)."""
    m = _AXIS_SUFFIX_RE.match(label)
    if m:
        return m.group(1).strip(), m.group(2).upper()
    return label, None


def _clean_labels(raw_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Normaliza espacios y quita el namespace de cada label del C3D."""
    out: dict[str, np.ndarray] = {}
    for label, traj in raw_data.items():
        clean = _strip_namespace(_normalize_ws(label))
        if clean in out:
            print(f"  [read_c3d] AVISO: label duplicado tras limpiar namespace "
                  f"('{clean}'). Se conserva el primero.")
            continue
        out[clean] = traj
    return out


def _channel_values(traj: np.ndarray) -> np.ndarray:
    """
    Extrae la serie escalar de un 'punto' que en realidad es un solo canal.

    Cuando un archivo separa cada marcador en ejes, suele guardar el valor en
    una sola de las 3 columnas (las otras quedan en 0 o NaN). Se elige la
    columna con mayor varianza temporal (ignorando NaN), que es la que lleva
    la señal real; así funciona sea cual sea la columna que use el exportador.
    """
    if traj.ndim == 1:
        return traj
    if traj.shape[1] == 1:
        return traj[:, 0]
    with np.errstate(all="ignore"):
        var = np.nanvar(traj, axis=0)
    var = np.where(np.isnan(var), -np.inf, var)
    return traj[:, int(np.argmax(var))]


def _reassemble_axis_split_channels(
    data: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """
    Recombina labels separados por eje ('LFHD X/Y/Z') en marcadores 3D.

    - Un nombre base con >=2 ejes distintos se trata como canales separados y
      se reensambla en un punto (n_frames, 3), tomando de cada canal su serie
      escalar (ver ``_channel_values``).
    - Un sufijo de eje suelto (un único 'X' sin sus hermanos Y/Z) se interpreta
      como un punto 3D simplemente mal etiquetado: se le quita el sufijo para
      que coincida con su nombre canónico, conservando sus 3 coordenadas.
    - Los labels sin sufijo de eje pasan sin cambios.
    """
    groups: dict[str, dict[str, str]] = {}
    base_order: list[str] = []
    for label in data:
        base, axis = _split_axis_suffix(label)
        if axis is None:
            continue
        if base not in groups:
            groups[base] = {}
            base_order.append(base)
        groups[base][axis] = label

    split_bases = {b: ax for b, ax in groups.items() if len(ax) >= 2}

    out: dict[str, np.ndarray] = {}
    used: set[str] = set()

    if split_bases:
        print(f"  [read_c3d] Detectados canales separados por eje para "
              f"{len(split_bases)} marcadores; reensamblando a puntos 3D.")
        for base in base_order:
            if base not in split_bases:
                continue
            axes = groups[base]
            n_frames = data[next(iter(axes.values()))].shape[0]
            traj = np.full((n_frames, 3), np.nan)
            for k, ax in enumerate("XYZ"):
                if ax in axes:
                    used.add(axes[ax])
                    traj[:, k] = _channel_values(data[axes[ax]])
            out[base] = traj

    # Resto: sufijos sueltos (se les quita el eje) y labels normales.
    for label, traj in data.items():
        if label in used:
            continue
        base, axis = _split_axis_suffix(label)
        key = base if axis is not None else label
        if key in out:
            # Evita pisar un marcador ya reensamblado o duplicado.
            if label not in out:
                out[label] = traj
            continue
        out[key] = traj
    return out


def _canonical_label(raw_label: str, label_map: dict[str, str]) -> str:
    """
    Normaliza un label de C3D al nombre canónico del proyecto.

    Orden de resolución (tolerante a espacios, namespace y mayúsculas):
      1. Limpia espacios y quita el prefijo de sujeto ('Subject0007:RSHO' -> 'RSHO').
      2. Coincidencia exacta en ``label_map``.
      3. Ya es un nombre canónico (``MARKER_NAMES``).
      4. Coincidencia en ``label_map`` o en ``MARKER_NAMES`` ignorando mayúsculas.
      5. Si nada coincide, se devuelve el label limpio tal cual.
    """
    label = _strip_namespace(_normalize_ws(raw_label))

    if label in label_map:
        return label_map[label]
    if label in _CANONICAL_SET:
        return label

    upper = label.upper()
    map_ci = {k.upper(): v for k, v in label_map.items()}
    if upper in map_ci:
        return map_ci[upper]
    if upper in _CANONICAL_CI:
        return _CANONICAL_CI[upper]
    return label


_CANONICAL_SET = set(MARKER_NAMES)
_CANONICAL_CI = {name.upper(): name for name in MARKER_NAMES}


def _apply_derived_markers(
    data: dict[str, np.ndarray], rules: dict[str, dict]
) -> dict[str, np.ndarray]:
    """
    Crea marcadores virtuales combinando otros ya presentes.

    Reglas: { nombre_canónico: {"method": "midpoint"|"single", "from": [labels]} }
    - No sobrescribe un marcador que ya exista con ese nombre.
    - Si falta alguna fuente, omite la regla (el validador lo reportará si era
      un marcador esperado).
    - "midpoint" usa nanmean: si una fuente está ocluida en un frame, usa la otra.
    """
    for name, rule in rules.items():
        if name in data:
            continue
        srcs = rule.get("from", [])
        if not srcs or not all(s in data for s in srcs):
            continue
        method = rule.get("method", "midpoint")
        if method == "single":
            data[name] = data[srcs[0]].copy()
        else:  # midpoint
            with np.errstate(all="ignore"):
                data[name] = np.nanmean(
                    np.stack([data[s] for s in srcs], axis=0), axis=0
                )
    return data


def _format_missing_markers_error(
    path: Path,
    missing: list[str],
    data: dict[str, np.ndarray],
    raw_data: dict[str, np.ndarray],
    mapping_report: dict[str, str],
) -> str:
    """Mensaje de error detallado para diagnosticar mapeos de labels fallidos."""
    found = sorted(data.keys())
    suggestions: dict[str, list[str]] = {}
    for m in missing:
        near = difflib.get_close_matches(m, found, n=3, cutoff=0.4)
        if near:
            suggestions[m] = near

    lines = [
        f"Faltan marcadores en {path.name} tras el mapeo de labels: {sorted(missing)}",
        f"Marcadores canónicos detectados ({len(found)}): {found}",
        f"Labels crudos en el archivo ({len(raw_data)}): {sorted(raw_data.keys())}",
        f"Mapeo aplicado (label_limpio -> canónico): {mapping_report}",
    ]
    if suggestions:
        lines.append(f"Posibles coincidencias por similitud: {suggestions}")
    lines.append(
        "Sugerencia: añade los pares faltantes a config.marker_set.C3D_LABEL_MAP "
        "(o revisa el prefijo de sujeto y los sufijos de eje X/Y/Z)."
    )
    return "\n".join(lines)


def _apply_axis_convention(traj: np.ndarray, up_axis: str) -> np.ndarray:
    """
    Convierte el sistema de coordenadas del C3D al sistema interno (Y-up).

    Muchos sistemas (Vicon, etc.) capturan con Z-up. BVH usa Y-up.

    up_axis : 'Y' → no se toca; 'Z' → Z-up a Y-up.
    """
    if up_axis.upper() == "Y":
        return traj
    if up_axis.upper() == "Z":
        # (x, y, z)_Zup → (x, z, -y)_Yup
        out = np.empty_like(traj)
        out[:, 0] = traj[:, 0]
        out[:, 1] = traj[:, 2]
        out[:, 2] = -traj[:, 1]
        return out
    raise ValueError(f"C3D_UP_AXIS debe ser 'Y' o 'Z', recibido '{up_axis}'")
