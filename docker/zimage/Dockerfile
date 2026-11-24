# Z-Image Turbo + ControlNet de pose (Fun-Controlnet-Union-2.1) - imagen sin pesos. Receta: recetas/zimage.sh + zimage_controlnet.sh (torch 2.6.0+cu124).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 UV_NO_CACHE=1 UV_LINK_MODE=copy

RUN set -eux; pip install uv; \
    uv venv /root/venv-zimage --python 3.11; . /root/venv-zimage/bin/activate; \
    uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124; \
    uv pip install diffusers transformers accelerate sentencepiece protobuf pillow numpy huggingface_hub safetensors peft pyyaml; \
    python -c "from diffusers import ZImagePipeline, ZImageControlNetPipeline, ZImageControlNetModel" \
      || { uv pip install "git+https://github.com/huggingface/diffusers"; \
           python -c "from diffusers import ZImagePipeline, ZImageControlNetPipeline, ZImageControlNetModel"; }; \
    mkdir -p /opt/assetgen/licenses; uv pip freeze > /opt/assetgen/zimage-requirements.lock; cat /opt/assetgen/zimage-requirements.lock; \
    rm -rf /root/.cache /tmp/* /var/tmp/*

COPY scripts/gen_images_zimage.py scripts/prompts_example.yaml scripts/prepare_weights.sh /opt/assetgen/
RUN chmod +x /opt/assetgen/prepare_weights.sh \
    && printf 'export ASSETGEN_MODEL=zimage\n' > /etc/profile.d/assetgen.sh \
    && cat /etc/profile.d/assetgen.sh >> /root/.bashrc

ENV ASSETGEN_MODEL=zimage
LABEL org.opencontainers.image.description="assetgen: Z-Image Turbo + ControlNet de pose (sin pesos; ejecutar /opt/assetgen/prepare_weights.sh zimage)"
