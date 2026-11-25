---
name: assetgen
description: Genera assets 3D para juegos (personajes, animales, criaturas, vehículos y máquinas, edificios, objetos) con modelos abiertos en GPUs rentadas de RunPod usando el CLI `assetgen` (texto → imagen → 3D con textura → rig → animación → GLB listo para Godot), como reemplazo de Meshy. Úsalo cuando el usuario quiera "generar los modelos" de un juego, crear un personaje/criatura/prop 3D, preparar el assets.yaml de un proyecto, elegir estilo o nivel de calidad de los assets, revisar el gasto en GPU, o importar los GLB generados a Godot.
---

# assetgen: assets 3D de juego con GPUs rentadas

Responde siempre en el idioma del usuario. El CLI hace el trabajo pesado en pods de RunPod que se crean y se borran solos; tú orquestas,
preguntas lo justo, controlas el gasto y revisas los resultados.

## 0. Antes de empezar (una vez por máquina)
- Localiza el CLI: `command -v assetgen`; si no está en el PATH, busca el repo clonado (`<repo>/cli/bin/assetgen`) o usa `$ASSETGEN_HOME/cli/bin/assetgen`.
  Comprueba con `assetgen --help` (subcomandos: `imagenes`, `construir`, `3d`, `rig`, `remaster`, `gasto`, `estado`, `limpiar`).
- RunPod: `runpodctl` configurado con la API key del usuario. **Nunca** imprimas la clave ni el correo de la cuenta (`runpodctl user` lo muestra).
- Configuración del usuario en `~/.config/assetgen/config.yaml` (ejemplo en `<repo>/cli/ejemplos/config_global.yaml`): `tope_mensual_usd`,
  ruta de Blender y rutas de los packs de Quaternius (UAL1 para humanoides, Ultimate Animated Animals para cuadrúpedos; CC0, no se redistribuyen).

## 1. Arranque de un proyecto: preguntar estilo y calidad
Si el proyecto no tiene `assetgen.yaml`, haz **una sola ronda** de preguntas (con la herramienta de preguntas si existe):
1. **Estilo**: estilizado / realista / low-poly.
2. **Cámara típica**: lejana (RTS, estrategia) / tercera persona / primer plano (héroes, cinemáticas).
3. **Calidad** (sugiere según la cámara y explica el costo):

| Preset | Para qué | 3D (TRELLIS.2) | Sugerido para |
|---|---|---|---|
| `boceto` | explorar ideas rápido y barato | voxel 512, textura 1024, ~20k caras | prototipos |
| `medio` | vista lejana | 1024 cascade, textura 2048, ~50k caras | cámara lejana |
| `alto` | personajes principales | 1024 cascade, textura 4096, ~150k caras + postproceso a quads | tercera persona |
| `maximo` | héroes y primeros planos | 1536 cascade, textura 4096, ~1M caras (necesita retopología para tiempo real) | primer plano |

4. **Presupuesto de polígonos** del motor por modelo, si lo sabe.

Modelo de imagen según el estilo: **estilizado → `qwen`** (Qwen-Image + Lightning, el más fiel al prompt); **realista → `zimage`** (Z-Image Turbo,
mejor piel y textura; en humanoides que se riggean usa `pose: a` con ControlNet de pose); objetos, máquinas, criaturas y edificios → `qwen`.

## 2. Proponer el `assetgen.yaml`
A partir de la idea o del documento de diseño, propón el archivo (plantilla comentada en `<repo>/cli/ejemplos/assetgen.yaml`) y deja que el
usuario lo revise antes de gastar. Campos clave: `proyecto`, `preset`, `estilo`, `camara`, `presupuesto_poligonos`, `variantes` (4 por
defecto), `modelo_imagen`, y por asset `nombre`, `tipo`, `prompt` (en inglés rinde mejor) u `imagen` (si ya la tiene).
- `humanoide`: imagen en pose A, rig con Puppeteer y clips de Quaternius UAL1 (Idle/Walk/Run/Sword_Attack/Death).
- `criatura`: rig con Puppeteer; cuadrúpedos con `animales: true` + `fuente: Wolf|Horse|Fox|Deer|Cow|...` reciben los clips de Quaternius.
- `mecanico`: sin rig; animación procedural con `plantilla: vehiculo | catapulta | rotor`.
- `edificio` / `otro`: solo imagen + 3D.

## 3. Flujo (cada paso con estimación y confirmación)
1. **Gasto**: `assetgen gasto` (mes en curso y tope). Antes de cada paso de GPU corre el mismo comando con `--dry-run` (`--seco` en `construir`)
   y dile al usuario la estimación en USD. Si se pasaría del tope mensual, pregunta.
2. **Imágenes**: `assetgen imagenes --config assetgen.yaml --out imagenes` → N variantes por asset en un solo pod + hoja de contacto.
   Abre la hoja al usuario (`open <hoja>` en macOS) y pregunta qué variante quiere por asset; escribe la elección en `imagenes/seleccion.yaml`.
   Si ninguna sirve, ajusta el prompt y repite solo esos assets (`--solo a,b`).
3. **Construir**: `assetgen construir --config assetgen.yaml --imagenes imagenes --out final` → 3D, rig y animación → `final/<asset>.glb` +
   `resumen.json`. No regenera lo que ya existe (`--forzar` para rehacer).
4. **Revisar**: renders o hoja de contacto de los GLB (~1000 px) y ábrelos al usuario en Blender o en Godot. Para personajes y animales,
   **enséñalos moviéndose** (reproduce sus clips) antes de pulir detalles o generar muchos más: así juzga el usuario la calidad.
5. **Godot**: copia los GLB a la carpeta de assets del proyecto (y, si hay MCP de Godot conectado, crea la escena y prueba las animaciones).

## 4. Reglas
- **Gasto**: un pod por paso; los procesos largos en segundo plano y se espera una sola vez (sin sondeo en bucle). Al terminar, `assetgen estado`
  debe salir vacío; si queda algo, `assetgen limpiar`. Reporta el costo real de cada corrida (lo registra `costos.jsonl`).
- **GPU**: el CLI elige por precio y disponibilidad. "no instances available" cuesta $0: reintenta con otra GPU (`--gpu "NVIDIA A40"`,
  `--nube any`). Si dos pods fallan seguidos, para y avisa. Prefiere 48 GB (A40/A6000/L40S) para los modelos de imagen.
- **Entradas**: no uses modelos ni imágenes de Meshy como entrada. Las referencias del usuario o de dominio público sí.
- **Licencias**: TRELLIS.2 (MIT), Qwen-Image (Apache 2.0) y Puppeteer (Apache 2.0) permiten uso comercial; los packs de Quaternius y los
  assets de MakeHuman son CC0. Ojo: DINOv3, los checkpoints que descarga Puppeteer y Hunyuan3D (uso privado) tienen licencias propias:
  revísalas antes de publicar un juego. Detalle en `<repo>/LICENSES.md`.
- Revisa cada hoja una sola vez y a ~1000 px.

## 5. Problemas conocidos
- Caras en cuerpos enteros: la cara ocupa pocos píxeles y sale blanda. Para un personaje principal genera aparte un primer plano de la
  cabeza (`assetgen remaster concepto --base <concepto> --cabeza`) y su 3D, e injértala (`<repo>/blender/remaster/injertar_cabeza.py`).
- Mallas de ~1M caras (`maximo`) no son para tiempo real: retopología por partes + horneado (`<repo>/blender/remaster/retopologia.py`,
  `hornear.py`), con normales hacia fuera y culling de caras traseras en el motor (la tela generada tiene forro interior).
- Texturas JPEG con perfil ICC no cargan en Godot 4.7: el exportador del visor ya los quita.

## 6. Modo remaster (avanzado)
`assetgen remaster analizar|concepto` sirve para analizar un modelo de un juego existente (glTF, FBX, volcados 3DMigoto) y generar un concepto
del remake. Advertencia aprendida: en juegos viejos con animaciones y esqueleto fijos, meter un personaje nuevo no llega a calidad de remake
(la deformación y los pasos de artista no quedan bien automatizados); lo realista es un "mod HD". Dilo antes de empezar uno.
