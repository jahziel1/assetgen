# Hunyuan3D-2.1 - imagen sin pesos. Receta: recetas/hunyuan.sh (venv Python 3.10 con uv, bpy 4.0.0, setuptools<81).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ARG MAX_JOBS=4
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0" FORCE_CUDA=1 MAX_JOBS=${MAX_JOBS} \
    PIP_NO_CACHE_DIR=1 UV_NO_CACHE=1 UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python

RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      libxrender1 libxi6 libxkbcommon0 libsm6 libxxf86vm1 libxfixes3 libgl1 libglib2.0-0 libegl1 wget \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; pip install uv; \
    cd /root; mkdir hunyuan; cd hunyuan; \
    git init -q; git remote add origin https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1; \
    git fetch -q --depth 1 origin 82920d643c0dc2f7bfd7255f45f62d386edfe60c; git checkout -q FETCH_HEAD; \
    uv venv /root/venv-hunyuan --python 3.10 --managed-python; . /root/venv-hunyuan/bin/activate; \
    uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124; \
    grep -v "^--extra-index-url" requirements.txt | sed 's/^bpy==4.0$/bpy==4.0.0/' > /root/hunyuan_req.txt; \
    uv pip install setuptools wheel pip; \
    uv pip install -r /root/hunyuan_req.txt --extra-index-url https://download.blender.org/pypi/ --index-strategy unsafe-best-match; \
    uv pip install "setuptools<81"; \
    (cd hy3dpaint/custom_rasterizer && uv pip install -e . --no-build-isolation && rm -rf build); \
    (cd hy3dpaint/DifferentiableRenderer && rm -f mesh_inpaint_processor && \
       c++ -O3 -Wall -shared -std=c++11 -fPIC $(python -m pybind11 --includes) mesh_inpaint_processor.cpp \
       -o mesh_inpaint_processor$(python -c 'import sysconfig;print(sysconfig.get_config_var("EXT_SUFFIX"))')); \
    ls hy3dpaint/DifferentiableRenderer/mesh_inpaint_processor*.so; \
    mkdir -p /opt/assetgen/licenses; cp LICENSE /opt/assetgen/licenses/Hunyuan3D-2.1.LICENSE; \
    cp NOTICE /opt/assetgen/licenses/Hunyuan3D-2.1.NOTICE 2>/dev/null || true; \
    rm -rf /root/hunyuan/.git /root/hunyuan_req.txt /root/.cache /tmp/* /var/tmp/*

COPY scripts/run_hunyuan.py scripts/prepare_weights.sh /opt/assetgen/
RUN chmod +x /opt/assetgen/prepare_weights.sh && ln -sf /opt/assetgen/run_hunyuan.py /root/run_hunyuan.py \
    && printf 'export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda\nexport ASSETGEN_MODEL=hunyuan\n' > /etc/profile.d/assetgen.sh \
    && cat /etc/profile.d/assetgen.sh >> /root/.bashrc

ENV ASSETGEN_MODEL=hunyuan
LABEL org.opencontainers.image.description="assetgen: Hunyuan3D 2.1 (sin pesos; ejecutar /opt/assetgen/prepare_weights.sh hunyuan). Licencia Tencent: no aplica en UE, Reino Unido ni Corea del Sur."
