"""Presets de calidad, listas de GPU y constantes de imagen."""

IMG_TRELLIS = "ghcr.io/jahziel1/assetgen-trellis2:latest"
IMG_PUPPETEER = "ghcr.io/jahziel1/assetgen-puppeteer:latest"

# Parametros de run_trellis2.py por preset (variables de entorno) y estimacion de segundos por asset.
PRESETS_3D = {
    "boceto": {"env": {"PIPELINE_TYPE": "512", "TEX_SIZE": "1024", "DECIM": "20000"}, "seg_asset": 30},
    "medio": {"env": {"PIPELINE_TYPE": "1024_cascade", "TEX_SIZE": "2048", "DECIM": "50000"}, "seg_asset": 60},
    "alto": {"env": {"PIPELINE_TYPE": "1024_cascade", "TEX_SIZE": "4096", "DECIM": "150000"}, "seg_asset": 150},
}

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
