"""Presets de calidad, listas de GPU y constantes de imagen."""

import os

# Imagenes Docker. Por defecto las UNIFICADAS (2 imagenes: imagen = Qwen+Z-Image, 3d = TRELLIS.2+Puppeteer).
# Para volver a las viejas (4 imagenes, un pod por etapa): ASSETGEN_IMAGENES=viejas en el entorno, o IMAGENES_VIEJAS = True aqui.
IMAGENES_VIEJAS = os.environ.get("ASSETGEN_IMAGENES", "").strip().lower() in ("viejas", "viejo", "old", "legacy")
UNIFICADAS = not IMAGENES_VIEJAS

IMG_IMAGEN = "ghcr.io/jahziel1/assetgen-imagen:latest"
IMG_3D = "ghcr.io/jahziel1/assetgen-3d:latest"
IMG_QWEN = "ghcr.io/jahziel1/assetgen-qwen:latest"          # viejas
IMG_ZIMAGE = "ghcr.io/jahziel1/assetgen-zimage:latest"
IMG_TRELLIS_VIEJA = "ghcr.io/jahziel1/assetgen-trellis2:latest"
IMG_PUPPETEER_VIEJA = "ghcr.io/jahziel1/assetgen-puppeteer:latest"
IMG_TRELLIS = IMG_3D if UNIFICADAS else IMG_TRELLIS_VIEJA
IMG_PIXAL3D = "ghcr.io/jahziel1/assetgen-pixal3d:latest"    # Pixal3D (sobre TRELLIS.2): imagen propia, no se comparte pod con el rig
IMG_PUPPETEER = IMG_3D if UNIFICADAS else IMG_PUPPETEER_VIEJA

# Parametros de run_trellis2.py por preset (variables de entorno) y estimacion de segundos por asset.
PRESETS_3D = {
    "boceto": {"env": {"PIPELINE_TYPE": "512", "TEX_SIZE": "1024", "DECIM": "20000"}, "seg_asset": 30},
    "medio": {"env": {"PIPELINE_TYPE": "1024_cascade", "TEX_SIZE": "2048", "DECIM": "50000"}, "seg_asset": 60},
    # alto: el 3D sale en calidad (150k caras, textura 4096) y se postprocesa en local (blender/postproceso): remesh a quads
    #       (~20k) + normal map y ORM horneados a 2048. Va ANTES del rig y de las plantillas.
    "alto": {"env": {"PIPELINE_TYPE": "1024_cascade", "TEX_SIZE": "4096", "DECIM": "150000"}, "seg_asset": 150,
             "post": {"caras": 20000, "tex": 2048}},
    # maximo: solo `assetgen 3d` (construir lo rechaza). Resolucion de voxel 1536 (baja sola si hay demasiados tokens), ~1M caras,
    #         textura 4096. Mucha mas VRAM y tiempo que `alto`; sin postproceso (el GLB sale crudo, para evaluar o remesh a mano).
    "maximo": {"env": {"PIPELINE_TYPE": "1536_cascade", "TEX_SIZE": "4096", "DECIM": "1000000", "MAX_GLB_MB": "400"}, "seg_asset": 220,
               "solo_3d": True},
}
# Motores 3D (imagen -> GLB): trellis2 (por defecto) y pixal3d. Cada uno con su imagen, script local, venv y carpeta de salida en el pod.
MODELOS_3D = ("trellis2", "pixal3d")
MODELO_3D_DEF = "trellis2"
MOTORES_3D = {
    "trellis2": {"imagen": IMG_TRELLIS, "script": "run_trellis2.py", "remoto": "/root/run_trellis2_local.py",
                 "venv": "/root/venv-trellis2", "out": "/root/out/trellis2", "pesos": "trellis2"},
    "pixal3d": {"imagen": IMG_PIXAL3D, "script": "run_pixal3d.py", "remoto": "/root/run_pixal3d_local.py",
                "venv": "/root/venv-pixal3d", "out": "/root/out/pixal3d", "pesos": "pixal3d"},
}
# Variables de entorno de run_pixal3d.py por preset; la estimacion por asset es la de TRELLIS + SEG_EXTRA_PIXAL3D.
ENV_PIXAL3D = {
    "boceto": {"RESOLUTION": "1024", "TEX_SIZE": "1024", "DECIM": "20000"},
    "medio": {"RESOLUTION": "1024", "TEX_SIZE": "2048", "DECIM": "50000"},
    "alto": {"RESOLUTION": "1536", "TEX_SIZE": "4096", "DECIM": "150000"},
    "maximo": {"RESOLUTION": "1536", "TEX_SIZE": "4096", "DECIM": "1000000", "MAX_GLB_MB": "400"},
}
SEG_EXTRA_PIXAL3D = 30


def env_3d(preset, modelo=MODELO_3D_DEF):
    """Variables de entorno del script del motor para el preset."""
    return dict(ENV_PIXAL3D[preset] if modelo == "pixal3d" else PRESETS_3D[preset]["env"])


def seg_asset_3d(preset, modelo=MODELO_3D_DEF):
    """Segundos estimados por asset (o por salida) segun preset y motor."""
    return PRESETS_3D[preset]["seg_asset"] + (SEG_EXTRA_PIXAL3D if modelo == "pixal3d" else 0)


# Presets reservados: se aceptan en la linea de comandos y el yaml pero construir se niega a correrlos (no hay GPU ni receta aun).
PRESETS_RESERVADOS = ("aaa",)
PRESETS_TODOS = list(PRESETS_3D) + list(PRESETS_RESERVADOS)

# GPUs (gpuId de RunPod). 48 GB primero; las de 24 GB sirven para TRELLIS.2 (low_vram) pero no se garantizan para Puppeteer.
GPU_48 = [
    "NVIDIA RTX A6000", "NVIDIA A40", "NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation", "NVIDIA L40",
]
GPU_24 = [
    "NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090", "NVIDIA RTX A5000", "NVIDIA L4",
]
MODOS_GPU = {
    "normal": GPU_48 + GPU_24,
    "rapido": ["NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation"] + [g for g in GPU_48 if g not in ("NVIDIA L40S", "NVIDIA RTX 6000 Ada Generation")] + GPU_24,
}
TOPE_PRECIO_H = 1.10      # USD/h por defecto
PREFIJO_POD = "assetgen-"

# Tiempos de planeacion (segundos) para estimar el costo antes de encender
ARRANQUE_EST = 12 * 60    # imagen + SSH + pesos
RIG_SEG_ASSET = 150
