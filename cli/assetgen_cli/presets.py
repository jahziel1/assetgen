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
IMG_PUPPETEER = IMG_3D if UNIFICADAS else IMG_PUPPETEER_VIEJA

# Parametros de run_trellis2.py por preset (variables de entorno) y estimacion de segundos por asset.
PRESETS_3D = {
    "boceto": {"env": {"PIPELINE_TYPE": "512", "TEX_SIZE": "1024", "DECIM": "20000"}, "seg_asset": 30},
    "medio": {"env": {"PIPELINE_TYPE": "1024_cascade", "TEX_SIZE": "2048", "DECIM": "50000"}, "seg_asset": 60},
    # alto: el 3D sale en calidad (150k caras, textura 4096) y se postprocesa en local (blender/postproceso): remesh a quads
    #       (~20k) + normal map y ORM horneados a 2048. Va ANTES del rig y de las plantillas.
    "alto": {"env": {"PIPELINE_TYPE": "1024_cascade", "TEX_SIZE": "4096", "DECIM": "150000"}, "seg_asset": 150,
             "post": {"caras": 20000, "tex": 2048}},
}
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
