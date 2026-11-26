# Pixal3D (TencentARC/Pixal3D, SIGGRAPH 2026) - imagen sin pesos. Se construye SOBRE el entorno de TRELLIS.2
# (misma receta que docker/trellis2: torch 2.6.0+cu124, flash-attn, nvdiffrast/nvdiffrec/CuMesh/FlexGEMM/o-voxel)
# y encima: requirements de Pixal3D, MoGe-2 (camara), natten 0.21.0 (compilado) y utils3d 0.0.2.
# Pesos: /opt/assetgen/prepare_weights.sh pixal3d.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# MAX_JOBS=2: con 4 el runner (16 GB) perdio comunicacion durante la compilacion de las extensiones CUDA
ARG MAX_JOBS=2
# natten: NO hay wheel precompilado para torch 2.6 (solo torch 2.7+/2.12+/2.13+; ver docs de NATTEN), se compila.
# Compilar tambien 9.0 (Hopper) anade los kernels CUTLASS hopper_fna: es lo mas lento de toda la imagen.
# Sin 9.0: la lista de GPUs del CLI (presets.py, GPU_48 + GPU_24) no tiene Hopper; anadir 9.0 aqui si algun dia se usa H100.
ARG NATTEN_ARCH="8.6;8.9"
ARG NATTEN_WORKERS=2
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0" FORCE_CUDA=1 MAX_JOBS=${MAX_JOBS} \
    PIP_NO_CACHE_DIR=1 UV_NO_CACHE=1 UV_LINK_MODE=copy

# 1) entorno TRELLIS.2 (venv propio) + extensiones CUDA, sin cache, en una sola capa.
#    El codigo de TRELLIS.2 solo se necesita para compilar o-voxel: Pixal3D lleva su propia copia de los modulos que usa.
RUN set -eux; pip install uv; \
    cd /root; mkdir trellis2; cd trellis2; \
    git init -q; git remote add origin https://github.com/microsoft/TRELLIS.2; \
    git fetch -q --depth 1 origin 75fbf0183001ed9876c8dbb35de6b68552ee08bd; git checkout -q FETCH_HEAD; \
    git submodule update --init --recursive; \
    uv venv /root/venv-pixal3d --python 3.11; . /root/venv-pixal3d/bin/activate; \
    printf 'torch==2.6.0\ntorchvision==0.21.0\n' > /root/constraints.txt; \
    uv pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124; \
    uv pip install -c /root/constraints.txt setuptools wheel packaging ninja cmake numpy; \
    uv pip install -c /root/constraints.txt imageio imageio-ffmpeg tqdm easydict opencv-python-headless transformers tensorboard pandas lpips zstandard kornia timm pillow plyfile huggingface_hub; \
    uv pip install -c /root/constraints.txt git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8; \
    uv pip install -c /root/constraints.txt https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl; \
    mkdir -p /root/ext; cd /root/ext; \
    git clone -q --depth 1 -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git; \
    git clone -q --depth 1 -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git; \
    git clone -q --depth 1 --recursive https://github.com/JeffreyXiang/CuMesh.git; \
    git clone -q --depth 1 --recursive https://github.com/JeffreyXiang/FlexGEMM.git; \
    cp -r /root/trellis2/o-voxel o-voxel; \
    for d in nvdiffrast nvdiffrec CuMesh FlexGEMM o-voxel; do echo "== build $d $(date +%T)"; free -m | sed -n 2p; uv pip install -c /root/constraints.txt ./$d --no-build-isolation; done; \
    mkdir -p /opt/assetgen/licenses; cp /root/trellis2/LICENSE /opt/assetgen/licenses/TRELLIS.2.LICENSE || true; \
    cp /root/ext/nvdiffrast/LICENSE.txt /opt/assetgen/licenses/nvdiffrast.LICENSE || true; \
    rm -rf /root/ext /root/trellis2 /root/.cache /tmp/* /var/tmp/*; true

# 2) Pixal3D: codigo fijado a commit + requirements SIN gradio (no se usa la demo) + MoGe-2 fijado + utils3d 0.0.2.
#    - MoGe: HEAD de microsoft/MoGe ya es MoGe-3 (V3, 2026-08-19: otro paquete utils3d_moge, flex-gemm git, gradio, numpy>=2).
#      Se fija 07444410 (2025-11-02, ultimo commit MoGe-2 antes de V3; trae moge.model.v2 y el repo HF Ruicheng/moge-2-vitl).
#      Se instala con --no-deps para no traer opencv-python (no headless), gradio ni el utils3d de su pin; sus dependencias reales van a mano.
#    - utils3d 0.0.2 va el ULTIMO (como pide el README) y con --reinstall-package para pisar el utils3d de TRELLIS.2.
#    - Todo con -c constraints: ningun paquete puede subir/bajar torch.
RUN set -eux; . /root/venv-pixal3d/bin/activate; \
    cd /root; mkdir pixal3d; cd pixal3d; \
    git init -q; git remote add origin https://github.com/TencentARC/Pixal3D; \
    git fetch -q --depth 1 origin f7cf38429b0bd264f1995f0f8743a88b1c728b94; git checkout -q FETCH_HEAD; \
    grep -vE '^(git\+https://github.com/microsoft/MoGe|gradio)' requirements.txt > /root/req_pixal3d.txt; \
    uv pip install -c /root/constraints.txt -r /root/req_pixal3d.txt einops; \
    uv pip install -c /root/constraints.txt --no-deps git+https://github.com/microsoft/MoGe.git@07444410f1e33f402353b99d6ccd26bd31e469e8; \
    uv pip install -c /root/constraints.txt click scipy matplotlib requests; \
    uv pip install -c /root/constraints.txt --reinstall-package utils3d https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl; \
    mkdir -p /opt/assetgen/licenses; cp LICENSE /opt/assetgen/licenses/Pixal3D.LICENSE; cp NOTICE /opt/assetgen/licenses/Pixal3D.NOTICE; \
    curl -fsSL https://raw.githubusercontent.com/microsoft/MoGe/07444410f1e33f402353b99d6ccd26bd31e469e8/LICENSE -o /opt/assetgen/licenses/MoGe.LICENSE || true; \
    curl -fsSL https://huggingface.co/camenduru/dinov3-vitl16-pretrain-lvd1689m/resolve/main/LICENSE.md -o /opt/assetgen/licenses/DINOv3.LICENSE || true; \
    rm -rf /root/req_pixal3d.txt /root/.cache /tmp/* /var/tmp/*; \
    find /root/pixal3d -name .git -prune -exec rm -rf {} + ; true

# 3) natten 0.21.0 (compilado; capa propia porque es la mas larga y para poder reintentar sin rehacer lo anterior)
#    --no-build-isolation: necesita el torch del venv; cmake esta en el venv. NATTEN_N_WORKERS acota la RAM (cada nvcc con CUTLASS ~3-5 GB).
RUN set -eux; . /root/venv-pixal3d/bin/activate; \
    echo "== build natten $(date +%T) arch=${NATTEN_ARCH}"; free -m | sed -n 2p; \
    NATTEN_CUDA_ARCH="${NATTEN_ARCH}" NATTEN_N_WORKERS=${NATTEN_WORKERS} uv pip install -c /root/constraints.txt natten==0.21.0 --no-build-isolation; \
    echo "== fin natten $(date +%T)"; \
    rm -rf /root/venv-pixal3d/lib/python3.11/site-packages/nvidia/*/include; \
    find /root/venv-pixal3d -name __pycache__ -type d -prune -exec rm -rf {} + ; pip uninstall -y uv; \
    rm -rf /root/constraints.txt /root/.cache /tmp/* /var/tmp/*; true

COPY scripts/run_pixal3d.py scripts/prepare_weights.sh /opt/assetgen/
RUN chmod +x /opt/assetgen/prepare_weights.sh && ln -sf /opt/assetgen/run_pixal3d.py /root/run_pixal3d.py \
    && printf 'export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda\nexport ASSETGEN_MODEL=pixal3d\n' > /etc/profile.d/assetgen.sh \
    && cat /etc/profile.d/assetgen.sh >> /root/.bashrc

ENV ASSETGEN_MODEL=pixal3d
LABEL org.opencontainers.image.description="assetgen: Pixal3D (MIT) + DINOv3 (DINOv3 License) sobre TRELLIS.2, sin pesos; ejecutar /opt/assetgen/prepare_weights.sh pixal3d"
