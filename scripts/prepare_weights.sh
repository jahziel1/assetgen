#!/usr/bin/env bash
# Descarga los pesos del modelo (no van dentro de la imagen). Uso: prepare_weights.sh <trellis2|pixal3d|hunyuan|puppeteer|qwen|qwen_edit|zimage>
# Idempotente. Respeta HF_HOME / HF_TOKEN si estan definidos.
set -euo pipefail
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
MODEL="${1:-${ASSETGEN_MODEL:-}}"
[ -n "$MODEL" ] || { echo "uso: $0 <trellis2|pixal3d|hunyuan|puppeteer|qwen|qwen_edit|zimage>"; exit 2; }

case "$MODEL" in
  trellis2)
    PY=/root/venv-trellis2/bin/python
    $PY -c 'from huggingface_hub import snapshot_download as s; s("microsoft/TRELLIS.2-4B"); s("ZhengPeng7/BiRefNet")'
    # DINOv3 oficial (facebook/) es gated: se convierten los pesos publicos de timm a transformers.DINOv3ViTModel
    if [ ! -f /root/dinov3-vitl16-hf/config.json ]; then
      CUDA_VISIBLE_DEVICES= $PY /opt/assetgen/convert_dinov3_timm_to_hf.py /root/dinov3-vitl16-hf
    fi
    ;;
  pixal3d)
    PY=/root/venv-pixal3d/bin/python
    # Todo lo que pide el codigo de Pixal3D (inference.py + pixal3d/pipelines), ninguno gated:
    #  - TencentARC/Pixal3D (MIT): pipeline.json + los ckpts de una imagen. Incluye TAMBIEN los decodificadores (ss_dec, shape_dec, tex_dec):
    #    pipeline.json los referencia como "<repo>/ckpts/..." dentro del mismo repo, asi que NO hace falta microsoft/TRELLIS.2-4B.
    #    Se excluyen los ckpts *_mv (multivista, no usado; ~22 GB).
    #  - Ruicheng/moge-2-vitl (MIT): estimacion de camara (MoGe-2).
    #  - camenduru/dinov3-vitl16-pretrain-lvd1689m: espejo PUBLICO (no gated) de facebook/dinov3-vitl16-pretrain-lvd1689m; es el que usa inference.py (DINOv3 License).
    #  - ZhengPeng7/BiRefNet (MIT): quitar fondo. El pipeline.json pide briaai/RMBG-2.0 (gated y no comercial); run_pixal3d.py lo sustituye por este.
    #  - valeoai/NAF (Apache 2.0): upsampler de features; el codigo lo carga con torch.hub (repo de GitHub + checkpoint de un release), se deja en la cache de torch.hub.
    $PY - <<'PYEOF'
import os, shutil, tempfile, urllib.request, zipfile
from huggingface_hub import snapshot_download as s
s("TencentARC/Pixal3D", allow_patterns=["pipeline.json", "LICENSE", "NOTICE", "ckpts/*_bf16.json", "ckpts/*_bf16.safetensors", "ckpts/*_fp16.json", "ckpts/*_fp16.safetensors"])
s("Ruicheng/moge-2-vitl")
s("camenduru/dinov3-vitl16-pretrain-lvd1689m")
s("ZhengPeng7/BiRefNet")
import torch
NAF_SHA = "37f2dfc180f2de53d98bd601109c0da0dd6b0f43"
NAF_CKPT = "https://github.com/valeoai/NAF/releases/download/model/naf_release.pth"
hub = torch.hub.get_dir(); os.makedirs(hub, exist_ok=True)
dst = os.path.join(hub, "valeoai_NAF_main")   # nombre que busca torch.hub.load("valeoai/NAF", ...) sin ref
if not os.path.isdir(os.path.join(dst, "src")):
    shutil.rmtree(dst, ignore_errors=True)
    with tempfile.TemporaryDirectory(dir=hub) as t:
        zp = os.path.join(t, "naf.zip")
        urllib.request.urlretrieve(f"https://github.com/valeoai/NAF/archive/{NAF_SHA}.zip", zp)
        zipfile.ZipFile(zp).extractall(t)
        os.rename(os.path.join(t, f"NAF-{NAF_SHA}"), dst)
torch.hub.load_state_dict_from_url(NAF_CKPT, map_location="cpu", progress=False)   # cachea hub/checkpoints/naf_release.pth
print("NAF listo en", dst)
PYEOF
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
