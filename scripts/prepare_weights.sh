#!/usr/bin/env bash
# Descarga los pesos del modelo (no van dentro de la imagen). Uso: prepare_weights.sh <trellis2|hunyuan|puppeteer|qwen|qwen_edit|zimage>
# Idempotente. Respeta HF_HOME / HF_TOKEN si estan definidos.
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
MODEL="${1:-${ASSETGEN_MODEL:-}}"
[ -n "$MODEL" ] || { echo "uso: $0 <trellis2|hunyuan|puppeteer|qwen|qwen_edit|zimage>"; exit 2; }

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
    PY=/root/venv-imagen/bin/python; [ -x "$PY" ] || PY=/root/venv-qwen/bin/python   # imagen unificada o imagen vieja
    # Qwen-Image base (~57 GB) + LoRA Lightning 8 pasos V2.0 bf16 (Apache 2.0, no gated)
    $PY -c 'from huggingface_hub import snapshot_download as s, hf_hub_download as h; s("Qwen/Qwen-Image"); h("lightx2v/Qwen-Image-Lightning","Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors")'
    ;;
  qwen_edit)
    PY=/root/venv-imagen/bin/python
    # Qwen-Image-Edit-2511 (editor multi-imagen, ~57 GB, Apache 2.0, no gated) + LoRA Lightning 8 pasos V1.0 bf16 (Apache 2.0)
    $PY -c 'from huggingface_hub import snapshot_download as s, hf_hub_download as h; s("Qwen/Qwen-Image-Edit-2511"); h("lightx2v/Qwen-Image-Edit-2511-Lightning","Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors")'
    ;;
  zimage)
    PY=/root/venv-imagen/bin/python; [ -x "$PY" ] || PY=/root/venv-zimage/bin/python   # imagen unificada o imagen vieja
    # Z-Image-Turbo (Apache 2.0, no gated) + ControlNet oficial Fun-Controlnet-Union-2.1 (Apache 2.0), archivo 2602-8steps (pose)
    $PY -c 'from huggingface_hub import snapshot_download as s, hf_hub_download as h; s("Tongyi-MAI/Z-Image-Turbo", allow_patterns=["*.json","*.txt","*.safetensors","*.jinja"], ignore_patterns=["assets/*"]); h("alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1","Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors")'
    ;;
  *) echo "modelo desconocido: $MODEL"; exit 2 ;;
esac
echo "PESOS_LISTOS $MODEL"
