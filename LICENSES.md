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
| Hunyuan3D 2.1 (Tencent-Hunyuan/Hunyuan3D-2.1, pesos tencent/Hunyuan3D-2.1) | Tencent Hunyuan 3D 2.1 Community License | **No aplica en la Union Europea, Reino Unido ni Corea del Sur**: el uso, y tambien el uso de los Outputs, fuera del "Territory" no esta licenciado. Si el producto/servicio supera 1 millon de usuarios activos mensuales hay que pedir licencia a Tencent. Sujeta a su Acceptable Use Policy; no se pueden usar Outputs para mejorar otros modelos de IA. Redistribuir (incluida esta imagen publica) exige conservar la licencia y el aviso, y solo se permite dentro del Territory. |
| Puppeteer (Seed3D/Puppeteer) | Apache 2.0 | Los checkpoints descargados (Michelangelo, PartField, CoTracker3, Video-Depth-Anything, etc.) tienen cada uno su licencia; revisar antes de uso comercial. |
| Real-ESRGAN (xinntao/Real-ESRGAN, `RealESRGAN_x4plus.pth`) | BSD 3-Clause | Lo descarga `prepare_weights.sh hunyuan`. |
| nvdiffrast (NVlabs) | NVIDIA Source Code License (1-Way Commercial) | Dependencia compilada en la imagen TRELLIS.2; uso comercial permitido en el sentido de la licencia, con restricciones propias (leer `/opt/assetgen/licenses/nvdiffrast.LICENSE`). |
| CuMesh, FlexGEMM (JeffreyXiang) | MIT | Dependencias de TRELLIS.2. |
| nvdiffrec (rama renderutils), flash-attention, PyTorch, bpy/Blender | Cada una con su licencia (ver sus repositorios) | No auditadas una a una en este repositorio. |

Aviso: esto es un inventario tecnico, no asesoria legal. Para publicar un juego comercial revisa las licencias
(en especial DINOv3, Hunyuan3D y los checkpoints de Puppeteer) con la version vigente al momento del uso.
