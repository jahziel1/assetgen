#!/usr/bin/env bash
# Descarga los pesos del modelo (no van dentro de la imagen). Uso: prepare_weights.sh <trellis2|hunyuan|puppeteer|qwen>
# Idempotente. Respeta HF_HOME / HF_TOKEN si estan definidos.
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
MODEL="${1:-${ASSETGEN_MODEL:-}}"
[ -n "$MODEL" ] || { echo "uso: $0 <trellis2|hunyuan|puppeteer|qwen>"; exit 2; }

case "$MODEL" in
  trellis2)
    PY=/root/venv-trellis2/bin/python
    $PY -c 'from huggingface_hub import snapshot_download as s; s("microsoft/TRELLIS.2-4B"); s("ZhengPeng7/BiRefNet")'
    # DINOv3 oficial (facebook/) es gated: se convierten los pesos publicos de timm a transformers.DINOv3ViTModel
    if [ ! -f /root/dinov3-vitl16-hf/config.json ]; then
      CUDA_VISIBLE_DEVICES= $PY /opt/assetgen/convert_dinov3_timm_to_hf.py /root/dinov3-vitl16-hf
    fi
    ;;
  hunyuan)
    PY=/root/venv-hunyuan/bin/python
    $PY -c 'from huggingface_hub import snapshot_download as s; s("tencent/Hunyuan3D-2.1"); s("facebook/dinov2-giant")'
    mkdir -p /root/hunyuan/hy3dpaint/ckpt
    wget -nc -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P /root/hunyuan/hy3dpaint/ckpt
    ;;
  puppeteer)
    source /root/venv-pup/bin/activate
    cd /root/puppeteer
    # los download.py asumen cwd = raiz de cada subproyecto; los ckpt de skeleton/skinning quedan anidados y se mueven
    if [ ! -d skeleton/skeleton_ckpts ]; then
      (cd skeleton && python download.py)
      mv skeleton/skeleton/skeleton_ckpts skeleton/ && rm -rf skeleton/skeleton
    fi
    if [ ! -d skinning/skinning_ckpts ]; then
      (cd skinning && ln -sfn ../../skeleton/third_partys/Michelangelo third_partys/Michelangelo && python download.py)
      mv skinning/skinning/skinning_ckpts skinning/ && rm -rf skinning/skinning
    fi
    (cd animation && python download.py)
    ;;
  qwen)
    PY=/root/venv-qwen/bin/python
    # Qwen-Image base (~57 GB) + LoRA Lightning 8 pasos V2.0 bf16 (Apache 2.0, no gated)
    $PY -c 'from huggingface_hub import snapshot_download as s, hf_hub_download as h; s("Qwen/Qwen-Image"); h("lightx2v/Qwen-Image-Lightning","Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors")'
    ;;
  *) echo "modelo desconocido: $MODEL"; exit 2 ;;
esac
echo "PESOS_LISTOS $MODEL"
