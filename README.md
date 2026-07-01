# Motion Capture Pipeline — Etapa 3 (Trayectorias 3D → BVH)

Implementación del pipeline de motion capture correspondiente a la **Tercera Etapa (Ian)** del proyecto: toma trayectorias 3D de marcadores como input y produce un archivo BVH con información cinemática del esqueleto.

## Estructura del proyecto

```
motion_capture/
├── main.py                       # Entry point: ejecuta el pipeline completo
├── requirements.txt
├── config/
│   ├── settings.py               # Constantes globales: fps, umbrales, paths
│   └── marker_set.py             # Definición del marker set (Helen Hayes)
├── src/
│   ├── io/
│   │   ├── markers_reader.py     # Lectura de trayectorias 3D (C3D)
│   │   └── bvh_writer.py         # Escritura de archivos BVH
│   ├── skeleton/
│   │   ├── joint.py              # Clase Joint (nodo de la jerarquía)
│   │   ├── skeleton.py           # Clase Skeleton (árbol completo)
│   │   └── hierarchy_builder.py  # Construcción de la jerarquía estándar
│   ├── calibration/
│   │   ├── tpose_calibration.py  # Paso 6.1: centros articulares (con T-pose)
│   │   └── auto_calibration.py   # Paso 6.1 (alt): calibración sin T-pose
│   ├── ik/
│   │   └── fabrik.py             # Paso 6.3: solver FABRIK
│   ├── rotations/
│   │   ├── extractor.py          # Paso 6.4: posiciones → rotaciones
│   │   └── constraints.py        # Paso 6.5: restricciones biomecánicas
│   ├── postprocess/
│   │   ├── foot_sliding.py       # Paso 7.1: corrección de foot sliding
│   │   └── segmentation.py       # Paso 7.2: segmentación del movimiento
│   └── utils/
│       └── math_utils.py         # Cuaterniones, swing-twist, Euler
├── data/
│   ├── input/                    # Archivos .c3d de entrada (output de Kelvin)
│   └── output/                   # Archivos BVH generados
└── docs/
```

## Formato de entrada: C3D

La entrada real del proyecto son archivos **.c3d** (estándar de motion capture).
El reader (`src/io/markers_reader.py`) extrae del propio archivo:

- **Framerate** (`POINT:RATE`) — no se hardcodea.
- **Unidades** (`POINT:UNITS`) — mm / cm / m, se convierten internamente a cm.
- **Labels** de los marcadores — se normalizan vía `config/marker_set.py`.
- **Oclusiones** — los frames con residual < 0 se marcan como NaN.

Dos parámetros se configuran una sola vez en `config/settings.py` según el
sistema de captura de Kelvin:

- `C3D_UP_AXIS` — eje vertical del C3D. `'Z'` para Vicon/Optitrack (por
  defecto), `'Y'` si el archivo ya viene en Y-up.
- `C3D_LABEL_MAP` (en `marker_set.py`) — mapeo de nombres de marcador si
  el C3D no usa los nombres canónicos del proyecto.

El backend de lectura es `ezc3d` (recomendado); si no está instalado, cae
automáticamente al paquete `c3d` puro-Python.

## Instalación

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## Uso

### Modo demo
```bash
python main.py --demo
```
Genera un archivo `.c3d` sintético (mm, Z-up, como un Vicon real), lo lee y
ejecuta el pipeline completo produciendo `data/output/demo.bvh`. Sirve para
verificar de extremo a extremo el path de C3D sin esperar los datos de Kelvin.

### Modo real (con .c3d de Kelvin)
```bash
# Con T-pose al inicio de la captura (preciso, default)
python main.py --input data/input/captura.c3d --output data/output/captura.bvh

# Sin T-pose (calibración estadística automática)
python main.py --input data/input/captura.c3d --calibration auto
```
El reader despacha automáticamente según la extensión (`.c3d`).

## Modos de calibración

El proyecto soporta dos modos de medición del esqueleto:

### `--calibration tpose` (default)
Usa los primeros segundos de la captura como pose estática T (configurable
en `TPOSE_DURATION_SEC`). Es el modo más preciso *cuando hay T-pose*.

### `--calibration auto`
No requiere período estático. Estima:
- **Longitudes de hueso** por mediana de las distancias entre centros
  articulares conectados, a lo largo de TODA la captura. Como el cuerpo es
  rígido, esa distancia es teóricamente constante y la mediana es robusta
  a ruido y deformación de piel.
- **Direcciones de hueso** desde una pose de referencia configurable con
  `--auto-reference`:
  - `first_frame` (default) — pose del frame 0.
  - `mean_pose` — pose promedio a lo largo de la captura.
  - `auto_neutral` — frame de máxima envergadura + mínima velocidad
    (heurística que aproxima un T-pose espontáneo).

En las pruebas, las longitudes de hueso difieren <2% entre ambos modos.

**Limitación común a ambos modos:** el twist (rotación sobre el eje del
hueso) requiere triadas de marcadores o un marcador adicional por segmento.
El extractor actual asume twist = 0 en cualquier modo de calibración.

## Diseño modular

Cada módulo es independiente y solo expone interfaces vía sus funciones/clases públicas. Para modificar la etapa de IK, por ejemplo, basta con tocar `src/ik/fabrik.py` sin romper el resto. Las dependencias entre módulos son explícitas y unidireccionales: `io → skeleton → calibration → ik → rotations → postprocess`.

## Referencias

- M. Gleicher, "Retargetting Motion to New Characters", SIGGRAPH '98.
- A. Aristidou and J. Lasenby, "FABRIK: A fast, iterative solver for the Inverse Kinematics problem", Graphical Models, 2011.
