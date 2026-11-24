# Puppeteer (Seed3D/Puppeteer, Apache-2.0) - imagen sin pesos. Receta: recetas/puppeteer.sh (torch 2.1.1 cu118, venv bpy aparte).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ARG MAX_JOBS=4
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0" FORCE_CUDA=1 MAX_JOBS=${MAX_JOBS} \
    PIP_NO_CACHE_DIR=1 UV_NO_CACHE=1 UV_LINK_MODE=copy

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
    rm -rf /root/req_pup.txt /root/.cache /tmp/* /var/tmp/*; \
    find /root/puppeteer -name .git -prune -exec rm -rf {} + ; true

COPY scripts/puppeteer/ scripts/prepare_weights.sh /opt/assetgen/
RUN chmod +x /opt/assetgen/*.sh && cd /opt/assetgen \
    && for f in glb2obj.py export_glb.py pt2npz.py render_sheet.py; do ln -sf /opt/assetgen/$f /root/puppeteer/$f; done \
    && for f in rig.sh anim.sh gpurun.sh ff.sh mkstats.py; do ln -sf /opt/assetgen/$f /root/$f; ln -sf /opt/assetgen/$f /root/puppeteer/$f; done \
    && printf 'export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda\nexport ASSETGEN_MODEL=puppeteer\n' > /etc/profile.d/assetgen.sh \
    && cat /etc/profile.d/assetgen.sh >> /root/.bashrc

ENV ASSETGEN_MODEL=puppeteer
LABEL org.opencontainers.image.description="assetgen: Puppeteer rigging y animacion (sin pesos; ejecutar /opt/assetgen/prepare_weights.sh puppeteer)"
