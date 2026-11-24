# assetgen-3d: TRELLIS.2 + Puppeteer (rigging) en UNA imagen, con sus dos venvs (sin pesos).
# Une docker/trellis2 y docker/puppeteer (siguen existiendo hasta que la prueba real pase). No se pueden compartir venvs:
# TRELLIS.2 exige torch 2.6+cu124 y transformers 4.56.2; Puppeteer exige torch 2.1.1+cu118 y numpy 1.26 (+ un venv de bpy aparte).
# Pesos: prepare_weights.sh trellis2 | puppeteer.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# MAX_JOBS=2: con 4 el runner (16 GB) perdio comunicacion durante la compilacion de las extensiones CUDA
ARG MAX_JOBS=2
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0" FORCE_CUDA=1 MAX_JOBS=${MAX_JOBS} \
    PIP_NO_CACHE_DIR=1 UV_NO_CACHE=1 UV_LINK_MODE=copy

# TRELLIS.2: codigo + entorno + dependencias + extensiones CUDA (sin cache) en una sola capa. timm sin -U (subiria torch) y transformers con la API que espera TRELLIS.2
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
    for d in nvdiffrast nvdiffrec CuMesh FlexGEMM o-voxel; do echo "== build $d $(date +%T)"; free -m | sed -n 2p; uv pip install ./$d --no-build-isolation; done; \
    uv pip install timm==1.0.31 einops; uv pip install transformers==4.56.2; \
    rm -rf /root/venv-trellis2/lib/python3.11/site-packages/nvidia/*/include; \
    mkdir -p /opt/assetgen/licenses; cp /root/trellis2/LICENSE /opt/assetgen/licenses/TRELLIS.2.LICENSE || true; \
    cp /root/ext/nvdiffrast/LICENSE.txt /opt/assetgen/licenses/nvdiffrast.LICENSE || true; \
    find /root/venv-trellis2 -name __pycache__ -type d -prune -exec rm -rf {} + ; pip uninstall -y uv; \
    rm -rf /root/ext /root/trellis2/.git /root/.cache /tmp/* /var/tmp/*; \
    find /root/trellis2 -name .git -prune -exec rm -rf {} + ; true

# --- Puppeteer (venv-pup torch 2.1.1+cu118 + venv-bpy)
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      ffmpeg libgl1 libegl1 libsm6 libxext6 libxrender1 libxi6 libxxf86vm1 libxfixes3 python3.10-dev build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN set -eux; pip install uv; \
    cd /root; mkdir puppeteer; cd puppeteer; \
    git init -q; git remote add origin https://github.com/Seed3D/Puppeteer; \
    git fetch -q --depth 1 origin 1c0f9fc6ad209667a0ec5ceac9b59964938a8b51; git checkout -q FETCH_HEAD; \
    git submodule update --init --recursive --depth 1; \
    uv venv --python 3.10 /root/venv-pup; . /root/venv-pup/bin/activate; \
    uv pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118; \
    uv pip install "setuptools<70" wheel pybind11 "cython==0.29.36" "numpy==1.26.4"; \
    uv pip install --no-build-isolation tetgen==0.5.2 mesh2sdf==1.1.0; \
    grep -v '^xformers\|^tetgen\|^cython\|^mesh2sdf' requirements.txt > /root/req_pup.txt; uv pip install -r /root/req_pup.txt; \
    uv pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118 --no-deps; \
    uv pip install --no-build-isolation flash-attn==2.6.3; \
    uv pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.1+cu118.html; \
    uv pip install --no-index pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt211/download.html; \
    uv pip install "numpy==1.26.4" "huggingface_hub<0.26"; \
    uv venv --python 3.11 /root/venv-bpy; \
    uv pip install --python /root/venv-bpy/bin/python bpy==4.2.0 numpy trimesh pillow; \
    sed -i 's/"bin_size": 0,/"bin_size": None,/' animation/renderer.py; grep -q '"bin_size": None,' animation/renderer.py; \
    mkdir -p /opt/assetgen/licenses; cp LICENSE /opt/assetgen/licenses/Puppeteer.LICENSE; \
    find /root/venv-pup /root/venv-bpy -name __pycache__ -type d -prune -exec rm -rf {} + ; pip uninstall -y uv; \
    rm -rf /root/req_pup.txt /root/.cache /tmp/* /var/tmp/*; \
    find /root/puppeteer -name .git -prune -exec rm -rf {} + ; true

COPY scripts/run_trellis2.py scripts/convert_dinov3_timm_to_hf.py scripts/prepare_weights.sh /opt/assetgen/
COPY scripts/puppeteer/ /opt/assetgen/
RUN chmod +x /opt/assetgen/*.sh && ln -sf /opt/assetgen/run_trellis2.py /root/run_trellis2.py && cd /opt/assetgen \
    && for f in glb2obj.py export_glb.py pt2npz.py render_sheet.py; do ln -sf /opt/assetgen/$f /root/puppeteer/$f; done \
    && for f in rig.sh anim.sh gpurun.sh ff.sh mkstats.py; do ln -sf /opt/assetgen/$f /root/$f; ln -sf /opt/assetgen/$f /root/puppeteer/$f; done \
    && printf 'export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda\nexport ASSETGEN_MODEL=3d\n' > /etc/profile.d/assetgen.sh \
    && cat /etc/profile.d/assetgen.sh >> /root/.bashrc

ENV ASSETGEN_MODEL=3d
LABEL org.opencontainers.image.description="assetgen: TRELLIS.2 + Puppeteer (sin pesos; ejecutar /opt/assetgen/prepare_weights.sh trellis2|puppeteer)"
