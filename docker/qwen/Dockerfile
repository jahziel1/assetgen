# Qwen-Image + LoRA Lightning 8 pasos - imagen sin pesos. Receta: recetas/qwen_image.sh + qwen_lightning.sh (torch 2.6.0+cu124).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 UV_NO_CACHE=1 UV_LINK_MODE=copy

RUN set -eux; pip install uv; \
    uv venv /root/venv-qwen --python 3.11; . /root/venv-qwen/bin/activate; \
    uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124; \
    uv pip install diffusers transformers accelerate sentencepiece protobuf pillow huggingface_hub safetensors peft pyyaml; \
    mkdir -p /opt/assetgen/licenses; uv pip freeze > /opt/assetgen/qwen-requirements.lock; cat /opt/assetgen/qwen-requirements.lock; \
    rm -rf /root/.cache /tmp/* /var/tmp/*

COPY scripts/gen_images.py scripts/prompts_example.yaml scripts/prepare_weights.sh /opt/assetgen/
RUN chmod +x /opt/assetgen/prepare_weights.sh \
    && printf 'export ASSETGEN_MODEL=qwen\n' > /etc/profile.d/assetgen.sh \
    && cat /etc/profile.d/assetgen.sh >> /root/.bashrc

ENV ASSETGEN_MODEL=qwen
LABEL org.opencontainers.image.description="assetgen: Qwen-Image + Lightning 8 pasos (sin pesos; ejecutar /opt/assetgen/prepare_weights.sh qwen)"
