# Licencias

Este repositorio (Dockerfiles, workflows y scripts de `scripts/`) es solo pegamento de orquestacion.
Cada modelo y dependencia conserva su propia licencia. Las imagenes no contienen pesos: se descargan de Hugging Face
al arrancar con `prepare_weights.sh`, y cada imagen incluye copia de las licencias del codigo que empaqueta en
`/opt/assetgen/licenses/`. Verificado leyendo el LICENSE de cada repositorio y la ficha del modelo en Hugging Face (noviembre 2026).

| Componente | Licencia | Notas |
|---|---|---|
| TRELLIS.2 (microsoft/TRELLIS.2, pesos TRELLIS.2-4B) | MIT | Uso comercial permitido. |
| DINOv3 (Meta; pesos timm `vit_large_patch16_dinov3_qkvb.lvd1689m`) | DINOv3 License (propia de Meta, no es open source estandar) | Ver https://ai.meta.com/resources/models-and-libraries/dinov3-license y el LICENSE.md de facebookresearch/dinov3. Los pesos de timm son una redistribucion y siguen bajo esta licencia; convertirlos al formato `transformers` no la cambia. Incluye clausulas de cumplimiento de controles comerciales/sanciones. |
| BiRefNet (ZhengPeng7/BiRefNet) | MIT | Sustituye a RMBG-2.0 (gated). |
| Pixal3D (TencentARC/Pixal3D, codigo y pesos) | MIT | Ficha de Hugging Face `license: mit`, no gated (verificado 2026-11-25). Antes tenia una licencia propia con restricciones comerciales; la cambiaron a MIT hacia junio de 2026 (discusion #2 del repo en HF). Usa los decodificadores de TRELLIS.2-4B (MIT). |
| MoGe-2 (microsoft/MoGe, pesos Ruicheng/moge-2-vitl) | MIT | Lo usa Pixal3D para estimar la camara. Ficha de HF `license: mit`, no gated. |
| DINOv3 redistribuido (camenduru/dinov3-vitl16-pretrain-lvd1689m) | DINOv3 License | Copia publica (no gated) de los pesos de Meta que usa Pixal3D; misma licencia y obligaciones que la fila de DINOv3. |
| NATTEN, utils3d | MIT | Dependencias de Pixal3D. |
| NAF (valeoai/NAF, codigo y `naf_release.pth`) | Apache 2.0 | Upsampler de features que Pixal3D carga con torch.hub; lo cachea `prepare_weights.sh pixal3d`. |
| Hunyuan3D 2.1 (Tencent-Hunyuan/Hunyuan3D-2.1, pesos tencent/Hunyuan3D-2.1) | Tencent Hunyuan 3D 2.1 Community License | **No aplica en la Union Europea, Reino Unido ni Corea del Sur**: el uso, y tambien el uso de los Outputs, fuera del "Territory" no esta licenciado. Si el producto/servicio supera 1 millon de usuarios activos mensuales hay que pedir licencia a Tencent. Sujeta a su Acceptable Use Policy; no se pueden usar Outputs para mejorar otros modelos de IA. Redistribuir (incluida esta imagen publica) exige conservar la licencia y el aviso, y solo se permite dentro del Territory. |
| Puppeteer (Seed3D/Puppeteer) | Apache 2.0 | Los checkpoints descargados (Michelangelo, PartField, CoTracker3, Video-Depth-Anything, etc.) tienen cada uno su licencia; revisar antes de uso comercial. |
| Qwen-Image (Qwen/Qwen-Image) | Apache 2.0 | Verificado en la ficha de Hugging Face (`license: apache-2.0`, no gated). Uso comercial permitido. |
| Qwen-Image-Lightning (lightx2v/Qwen-Image-Lightning, LoRA 8 pasos V2.0) | Apache 2.0 | Verificado en la ficha de Hugging Face (`license: apache-2.0`, no gated). Se aplica sobre Qwen-Image. |
| Real-ESRGAN (xinntao/Real-ESRGAN, `RealESRGAN_x4plus.pth`) | BSD 3-Clause | Lo descarga `prepare_weights.sh hunyuan`. |
| MakeHuman (makehumancommunity/makehuman: malla base hm08, ojos, texturas de ojo) | CC0 1.0 (assets); el codigo de MakeHuman es AGPL pero aqui no se usa | `LICENSE.md` del repositorio, seccion C: "These assets have been released under CC0 1.0 Universal." (base, targets, texturas, ropa/MHCLO); cada archivo lo repite: "This asset was explicitly released as CC0 in september 2020". Texto legal en `LICENSE.ASSETS.md` (https://github.com/makehumancommunity/makehuman). `blender/remaster/assets_makehuman.py` los descarga a `~/.cache/assetgen/makehuman/`; no se suben al repositorio. |
| MediaPipe Face Landmarker (google-ai-edge/mediapipe, modelo `face_landmarker.task`) | Apache 2.0 | Detecta landmarks 2D en `cara_landmarks2d.py` (venv aparte); el modelo se descarga a `~/.cache/assetgen/mediapipe/`. Licencia del modelo: ver su tarjeta (Model Card) de Google AI Edge antes de uso comercial. |
| nvdiffrast (NVlabs) | NVIDIA Source Code License (1-Way Commercial) | Dependencia compilada en la imagen TRELLIS.2; uso comercial permitido en el sentido de la licencia, con restricciones propias (leer `/opt/assetgen/licenses/nvdiffrast.LICENSE`). |
| CuMesh, FlexGEMM (JeffreyXiang) | MIT | Dependencias de TRELLIS.2. |
| nvdiffrec (rama renderutils), flash-attention, PyTorch, bpy/Blender | Cada una con su licencia (ver sus repositorios) | No auditadas una a una en este repositorio. |

**DINOv3 y un juego comercial** (leido en el LICENSE.md de facebookresearch/dinov3, version del 19 de agosto de 2025): concede uso
mundial, gratuito y comercial. Obligaciones: copia del Agreement solo si se redistribuyen los DINO Materials (el juego no los lleva, solo
mallas generadas), mencion en publicaciones de investigacion, cumplir controles de exportacion y sanciones, y no hacer ingenieria inversa.
**No exige** un texto "Built with DINOv3" (eso circula en resumenes, pero no esta en la licencia); ponerlo en los creditos es cortesia.

Aviso: esto es un inventario tecnico, no asesoria legal. Para publicar un juego comercial revisa las licencias
(en especial DINOv3, Hunyuan3D y los checkpoints de Puppeteer) con la version vigente al momento del uso.
