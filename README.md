# assetgen

Pipeline de generación de assets para juegos con modelos abiertos en GPUs rentadas (RunPod):
texto → imagen → modelo 3D con textura → rigging → animación.

- Imagen Docker con TRELLIS.2, Hunyuan3D 2.1 y Puppeteer (se construye con GitHub Actions y se publica en GHCR).
- Los pesos de los modelos se descargan de Hugging Face al arrancar; no se incluyen en la imagen.
- Cada modelo conserva su propia licencia (ver `LICENSES.md`).

> En construcción.
