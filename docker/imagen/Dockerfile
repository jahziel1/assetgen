# assetgen-imagen: Qwen-Image + LoRA Lightning 8 pasos Y Z-Image Turbo + ControlNet de pose, en UN solo venv (sin pesos).
# Une docker/qwen y docker/zimage (siguen existiendo hasta que la prueba real pase). Un solo venv: ambas recetas usaban el mismo
# conjunto de paquetes (torch 2.6.0+cu124, diffusers>=0.43 que ya trae ZImageControlNetPipeline). Los pesos se bajan con
# prepare_weights.sh qwen|zimage (cada uno solo si hace falta).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 UV_NO_CACHE=1 UV_LINK_MODE=copy

RUN set -eux; pip install uv; \
    uv venv /root/venv-imagen --python 3.11; . /root/venv-imagen/bin/activate; \
    uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124; \
    uv pip install diffusers transformers accelerate sentencepiece protobuf pillow numpy huggingface_hub safetensors peft pyyaml; \
    python -c "from diffusers import QwenImagePipeline, ZImagePipeline, ZImageControlNetPipeline, ZImageControlNetModel" \
      || { uv pip install "git+https://github.com/huggingface/diffusers"; \
           python -c "from diffusers import ZImagePipeline, ZImageControlNetPipeline, ZImageControlNetModel"; }; \
    mkdir -p /opt/assetgen/licenses; uv pip freeze > /opt/assetgen/imagen-requirements.lock; cat /opt/assetgen/imagen-requirements.lock; \
    find /root/venv-imagen -name __pycache__ -type d -prune -exec rm -rf {} + ; \
    rm -rf /root/venv-imagen/lib/python3.11/site-packages/nvidia/*/include; \
    pip uninstall -y uv; rm -rf /root/.cache /tmp/* /var/tmp/*

COPY scripts/gen_images.py scripts/gen_images_zimage.py scripts/prompts_example.yaml scripts/prepare_weights.sh /opt/assetgen/
RUN chmod +x /opt/assetgen/prepare_weights.sh \
    && printf 'export ASSETGEN_MODEL=imagen\n' > /etc/profile.d/assetgen.sh \
    && cat /etc/profile.d/assetgen.sh >> /root/.bashrc

ENV ASSETGEN_MODEL=imagen
LABEL org.opencontainers.image.description="assetgen: Qwen-Image Lightning + Z-Image Turbo/ControlNet (sin pesos; ejecutar /opt/assetgen/prepare_weights.sh qwen|zimage)"
