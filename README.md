# assetgen

Pipeline de generación de assets para juegos con modelos abiertos en GPUs rentadas (RunPod):
texto → imagen → modelo 3D con textura → rigging → animación.

## Imágenes Docker (GHCR)

Una imagen por modelo, construida con GitHub Actions (`.github/workflows/build.yml`). Base:
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (conserva SSH y arranque de RunPod). Las extensiones CUDA se
compilan para sm 8.6, 8.9 y 9.0 (A40/A6000/3090, L40S/4090, H100).

| Imagen | Contenido |
|---|---|
| `ghcr.io/jahziel1/assetgen-trellis2` | TRELLIS.2 + DINOv3 (convertido) + BiRefNet |
| `ghcr.io/jahziel1/assetgen-hunyuan` | Hunyuan3D 2.1 (forma + textura PBR) |
| `ghcr.io/jahziel1/assetgen-puppeteer` | Puppeteer (rigging + animación) y export GLB con bpy |

Sin pesos: al arrancar el pod ejecuta `/opt/assetgen/prepare_weights.sh <trellis2|hunyuan|puppeteer>`.
Los scripts de ejecución están en `/opt/assetgen/` (`run_trellis2.py`, `run_hunyuan.py`, `rig.sh`, `anim.sh`...);
usan los venvs `/root/venv-trellis2`, `/root/venv-hunyuan`, `/root/venv-pup` y `/root/venv-bpy`.

Cada modelo conserva su propia licencia (ver `LICENSES.md`).
