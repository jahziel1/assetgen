# assetgen

Pipeline de generación de assets para juegos con modelos abiertos en GPUs rentadas (RunPod):
texto → imagen → modelo 3D con textura → rigging → animación.

## Imágenes Docker (GHCR)

Dos imagenes de uso diario (unificadas), construidas con GitHub Actions (`.github/workflows/build.yml`). Base:
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (conserva SSH y arranque de RunPod, y casi siempre ya esta en cache en los hosts de RunPod).
Las extensiones CUDA se compilan para sm 8.6, 8.9 y 9.0 (A40/A6000/3090, L40S/4090, H100).

| Imagen | Contenido | Venvs |
|---|---|---|
| `ghcr.io/jahziel1/assetgen-imagen` | Qwen-Image + LoRA Lightning 8 pasos (`gen_images.py`) y Z-Image Turbo + ControlNet de pose (`gen_images_zimage.py`) | `/root/venv-imagen` (uno solo) |
| `ghcr.io/jahziel1/assetgen-3d` | TRELLIS.2 + DINOv3 (convertido) + BiRefNet y Puppeteer (rigging + animacion) con export GLB por bpy | `/root/venv-trellis2`, `/root/venv-pup`, `/root/venv-bpy` |
| `ghcr.io/jahziel1/assetgen-hunyuan` | Hunyuan3D 2.1 (forma + textura PBR). Privada por licencia | `/root/venv-hunyuan` |

Las imagenes viejas (`assetgen-qwen`, `assetgen-zimage`, `assetgen-trellis2`, `assetgen-puppeteer`) y sus Dockerfiles se conservan; el CLI vuelve a usarlas con
`ASSETGEN_IMAGENES=viejas`. Los dos modelos de `assetgen-imagen` comparten venv porque `diffusers>=0.43` y `transformers>=5` se niegan a usar el torch 2.4 del sistema
(piden torch>=2.6), asi que no se puede reutilizar el torch de la base; los venvs de `assetgen-3d` no se pueden unir (torch 2.6 + transformers 4.56.2 frente a torch 2.1.1 + numpy 1.26).

Sin pesos: al arrancar el pod ejecuta `/opt/assetgen/prepare_weights.sh <qwen|zimage|trellis2|puppeteer|hunyuan>`.
Los scripts de ejecucion estan en `/opt/assetgen/` (`run_trellis2.py`, `rig.sh`, `anim.sh`, `gen_images*.py`...).

Cada modelo conserva su propia licencia (ver `LICENSES.md`).
