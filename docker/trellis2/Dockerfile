# TRELLIS.2 (microsoft/TRELLIS.2) - imagen sin pesos. Receta: recetas/trellis2.sh (torch 2.6.0+cu124, transformers 4.56.2).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ARG MAX_JOBS=4
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0" FORCE_CUDA=1 MAX_JOBS=${MAX_JOBS} \
    PIP_NO_CACHE_DIR=1 UV_NO_CACHE=1 UV_LINK_MODE=copy

# codigo + entorno + dependencias (sin cache) en una sola capa
RUN set -eux; pip install uv; \
    cd /root; mkdir trellis2; cd trellis2; \
    git init -q; git remote add origin https://github.com/microsoft/TRELLIS.2; \
    git fetch -q --depth 1 origin 75fbf0183001ed9876c8dbb35de6b68552ee08bd; git checkout -q FETCH_HEAD; \
    git submodule update --init --recursive; \
    uv venv /root/venv-trellis2 --python 3.11; . /root/venv-trellis2/bin/activate; \
    uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124; \
    uv pip install setuptools wheel packaging ninja numpy; \
    uv pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless transformers tensorboard pandas lpips zstandard kornia timm pillow plyfile huggingface_hub; \
    uv pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8; \
    uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl; \
    mkdir -p /root/ext; cd /root/ext; \
    git clone -q --depth 1 -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git; \
    git clone -q --depth 1 -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git; \
    git clone -q --depth 1 --recursive https://github.com/JeffreyXiang/CuMesh.git; \
    git clone -q --depth 1 --recursive https://github.com/JeffreyXiang/FlexGEMM.git; \
    cp -r /root/trellis2/o-voxel o-voxel; \
    for d in nvdiffrast nvdiffrec CuMesh FlexGEMM o-voxel; do uv pip install ./$d --no-build-isolation; done; \
    mkdir -p /opt/assetgen/licenses; cp /root/trellis2/LICENSE /opt/assetgen/licenses/TRELLIS.2.LICENSE || true; \
    cp /root/ext/nvdiffrast/LICENSE.txt /opt/assetgen/licenses/nvdiffrast.LICENSE || true; \
    rm -rf /root/ext /root/trellis2/.git /root/.cache /tmp/* /var/tmp/*; \
    find /root/trellis2 -name .git -prune -exec rm -rf {} + ; true

# DINOv3 via timm (sin -U timm: subiria torch) + transformers con la API que espera TRELLIS.2
RUN . /root/venv-trellis2/bin/activate && uv pip install timm==1.0.31 einops && uv pip install transformers==4.56.2

COPY scripts/run_trellis2.py scripts/convert_dinov3_timm_to_hf.py scripts/prepare_weights.sh /opt/assetgen/
RUN chmod +x /opt/assetgen/prepare_weights.sh && ln -sf /opt/assetgen/run_trellis2.py /root/run_trellis2.py \
    && printf 'export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda\nexport ASSETGEN_MODEL=trellis2\n' > /etc/profile.d/assetgen.sh \
    && cat /etc/profile.d/assetgen.sh >> /root/.bashrc

ENV ASSETGEN_MODEL=trellis2
LABEL org.opencontainers.image.description="assetgen: TRELLIS.2 (sin pesos; ejecutar /opt/assetgen/prepare_weights.sh trellis2)"
