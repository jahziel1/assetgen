# assetgen CLI

Comando unico para generar assets de juego en GPUs rentadas de RunPod. Cada corrida **enciende un pod, trabaja, descarga
y borra el pod** (siempre, tambien ante error o Ctrl-C) y anota el costo en `<salida>/costos.jsonl`.

## Requisitos e instalacion

- Python 3.9+, `runpodctl` configurado (`~/.local/bin/runpodctl`, o variable `RUNPODCTL`) y una llave SSH registrada en RunPod
  (por defecto `~/.ssh/runpod_ed25519`, o variable `ASSETGEN_SSH_KEY`).
- `ssh` y `tar` en la maquina local.

```bash
cd cli && pip install -e .        # instala el comando `assetgen` (incluye PyYAML)
# o sin instalar:
./cli/bin/assetgen --help
```

PyYAML es opcional: si no esta instalado (p. ej. un Python gestionado por Homebrew), el CLI usa un lector YAML minimo propio (`assetgen_cli/yamlmini.py`:
mapas, listas, escalares, comentarios; sin anclas ni bloques `|`/`>`). Si `~/.config/assetgen/config.yaml` existe pero no se puede leer, el CLI **se detiene**
con el error en vez de seguir sin tope mensual.

El CLI sube `scripts/run_trellis2.py` del repo al pod, asi que no hace falta reconstruir la imagen para probar cambios.

## Pipeline completo (yaml -> final/<asset>.glb)

```bash
assetgen imagenes  --config assetgen.yaml --out proy/imagenes     # Qwen-Image Lightning o Z-Image Turbo (+ControlNet de pose): N variantes por asset (solo GPUs de 48 GB)
# mira proy/imagenes/hoja_<asset>.png y edita proy/imagenes/seleccion.yaml (asset: variante, por defecto 1)
assetgen construir --config assetgen.yaml --imagenes proy/imagenes --out proy/build
# -> proy/build/final/<asset>.glb con sus clips, proy/build/resumen.json (asset, tipo, clips, caras, MB, costo)
```

`imagenes`: prompt del yaml + sufijo de estilo del proyecto (`estilo`: realista|estilizado|low-poly, o `sufijo_estilo`) + sufijo segun tipo
(`humanoide`: "standing in A-pose with arms slightly away from the body, empty hands, full body"; `mecanico`/`criatura`: "full body, centered").
Escribe `<asset>_v1..vN.png`, `hoja_<asset>.png` (<=1000 px), `seleccion.yaml` (no pisa elecciones existentes) y `prompts.yaml`.
Usa `--variantes N` (def. 4) y `--solo a,b`. El pod tiene disco de >=100 GB (pesos de Qwen ~58 GB).

Modelo de imagen (`modelo_imagen: qwen|zimage`, a nivel proyecto y por asset; por defecto `qwen`):
- `qwen` -> `ghcr.io/jahziel1/assetgen-qwen`, resolucion por defecto 1328 (`tamano_imagen` en el proyecto o el asset la cambia).
- `zimage` -> `ghcr.io/jahziel1/assetgen-zimage` (Z-Image Turbo, Apache 2.0), resolucion por defecto 1536; en A40 ~18 s/imagen (~29 s con ControlNet), VRAM 30-39 GB.
  `pose: a|libre` por asset (por defecto `a` para `humanoide`, `libre` para el resto): con `a` el generador dibuja un esqueleto OpenPose en pose A
  y lo usa con el ControlNet oficial `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` (2602-8steps, Apache 2.0). `pose` solo tiene efecto con `zimage`.
- Si un proyecto mezcla modelos, `imagenes` agrupa por modelo: **un pod por modelo**.
- `literal: true` en un asset usa su `prompt` tal cual, sin sufijos de estilo ni de tipo.

`construir` valida todo antes de encender nada (imagenes, plantillas, rutas de Quaternius y Blender) y despues:
1. `3d` con el preset del proyecto; el GLB se llama como el asset del yaml (`arquero.glb`), no como la imagen.
   1b. Solo con el preset `alto`: **postproceso local** en Blender (`blender/postproceso/postprocesar.py`) -> `<out>/post/<asset>.glb`
   (remesh a ~20k quads con QuadriFlow, UV nuevas y normal map + ORM horneados a 2048 desde la malla de 150k; los quads reales quedan en
   `post/<asset>_quads.blend`). Va **antes** del rig y de las plantillas: Puppeteer riggea la malla final y las animaciones de `mecanico`/`humanoide`
   se aplican sobre ella.
2. `rig` (Puppeteer) **solo** para `humanoide` y `criatura`. El exito se decide por la existencia del GLB riggeado (bpy da segfault al cerrar).
3. En local con Blender (`blender/`): `humanoide` -> `blender/retarget/retarget_ual.py` (clips Idle_Loop, Walk_Loop, Run_Loop, Sword_Attack, Death de
   Quaternius UAL1); `mecanico` -> `blender/mecanica/animar_mecanico.py` con su `plantilla` (`vehiculo|catapulta|rotor`); `criatura` queda riggeada sin
   clips, salvo que el asset traiga `animales:` (ruta del pack Quaternius Ultimate Animated Animals o un nombre de `quaternius.animales` en la config
   global) y `fuente:` (`Wolf`, `Horse`, `Fox`...): entonces `blender/animales/retarget_animal.py` reemplaza el esqueleto de Puppeteer por el de Quaternius
   (pesos transferidos a nuestra malla, ver `blender/animales/`) y conserva todos sus clips (`Walk_Loop`, `Gallop_Loop`, `Idle_Loop`, `Attack`, `Death`...);
   otros tipos quedan como el GLB de 3D.
Es reanudable: no repite lo que ya existe (`--forzar` para rehacer). Intermedios en `<out>/3d`, `post` (solo `alto`), `rig`, `retarget`, `mecanica`, `animales`, `logs`.

`assetgen construir ... --seco` solo valida (yaml, imagenes, config global, tope mensual, plantillas) y muestra el plan y la estimacion maxima; no enciende
nada ni escribe salida.

Verificacion visual de un final (reimporta, lista clips/texturas y renderiza una hoja de cuadros del clip principal):

```bash
blender -b -P blender/verificar_glb.py -- final/arquero.glb hoja.png [--clip Walk_Loop] [--n 6]
```

### Configuracion global (`~/.config/assetgen/config.yaml`, ver `ejemplos/config_global.yaml`)

- `quaternius.ual1`: ruta a `UAL1_Standard.glb` (el pack no se redistribuye; tambien variable `ASSETGEN_UAL1`).
- `blender`: ejecutable de Blender (o variable `BLENDER`, o `blender` en el PATH).
- `tope_mensual_usd`: cada corrida suma el gasto del mes de **todos los proyectos** (`~/.config/assetgen/costos_global.jsonl`, donde se registra cada
  costo ademas del `costos.jsonl` local) y, si lo superaria con la estimacion, pide confirmacion (`--si` para continuar sin preguntar; sin terminal aborta).
  `assetgen gasto` muestra el total del mes. `ASSETGEN_CONFIG` cambia la ruta del archivo (y del registro global).

## Uso de los subcomandos sueltos

```bash
# Imagenes -> GLB con TRELLIS.2 (imagen ghcr.io/jahziel1/assetgen-3d)
assetgen 3d --input imagenes/ --preset medio --out salida/3d --presupuesto 1.0

# GLB -> GLB riggeado con Puppeteer (imagen ghcr.io/jahziel1/assetgen-3d)
assetgen rig --input salida/3d --out salida/rig

# usando un assetgen.yaml (ver ejemplos/assetgen.yaml)
assetgen 3d --config assetgen.yaml --out assets/3d

assetgen estado      # pods de assetgen encendidos y gasto por hora
assetgen limpiar     # borra cualquier pod assetgen-* que haya quedado
```

Opciones comunes de `3d` y `rig`:

| Opcion | Efecto |
|---|---|
| `--preset boceto\|medio\|alto` | calidad (solo `3d`; ver tabla) |
| `--modo normal\|rapido` | lista de GPUs: normal = A6000, A40, L40S, RTX 6000 Ada, L40 y luego 24 GB (4090, 3090, A5000, L4); rapido = L40S primero |
| `--tope 1.10` | precio maximo USD/h por GPU |
| `--gpu id1,id2` | lista explicita de `gpuId` de RunPod (sustituye al modo) |
| `--nube secure\|community\|any` | por defecto `secure` (SSH garantizado) |
| `--presupuesto USD` | aborta antes de encender si la estimacion lo excede; apaga y borra si en ejecucion se supera |
| `--max-min 60` | vida maxima del pod: apagado de seguridad dentro del pod (GraphQL) y vigilante local |
| `--timeout-arranque 15` | minutos maximos hasta tener SSH (imagen de ~11 GB); mientras el pod siga descargando no se borra |
| `--disco 80` | disco del contenedor (GB) |
| `--forzar` | regenerar aunque ya exista el `.glb` (por defecto no se regenera lo existente) |
| `--dry-run` | muestra GPUs candidatas y estimacion, sin encender nada |

### Presets de 3D

| Preset | Pipeline TRELLIS.2 | Textura | Caras |
|---|---|---|---|
| `boceto` | `512` | 1024 | ~20k |
| `medio` | `1024_cascade` | 2048 | ~50k |
| `alto` | `1024_cascade` | 4096 | 150k en bruto; con `construir` se postprocesa a ~20k quads + normal map/ORM 2048 |
| `aaa` | reservado: `construir` se niega a correrlo todavia | | |

El preset `alto` en `construir`: TRELLIS.2 genera en calidad (150k caras, textura 4096) y despues se hace en local (Blender, sin GPU) remesh a quads
(~20k, QuadriFlow) + UV + bake de normal map y ORM a 2048 con la malla de 150k como alta; recien despues se riggea o se aplica la plantilla de animacion.
`medio` y `boceto` no postprocesan. `assetgen 3d` suelto no postprocesa (solo `construir`).

Motor 3D (`modelo_3d: trellis2|pixal3d` en el yaml o `--modelo-3d` en `3d` y `construir`; la bandera gana; por defecto `trellis2`). `pixal3d` usa la imagen
`ghcr.io/jahziel1/assetgen-pixal3d` (sobre TRELLIS.2, con camara estimada por MoGe-2), por lo que 3D y rig nunca comparten pod, y no admite `--multivista`. Junto a cada
`<n>.glb` baja `<n>.camara.json` y `<n>_entrada.png`. Presets (resolucion / textura / caras): `boceto` 1024/1024/20k, `medio` 1024/2048/50k, `alto` 1536/4096/150k, `maximo` 1536/4096/1M; ~30 s mas por asset que TRELLIS.2.
## Imagenes y pods (unificadas)

`imagenes` usa `ghcr.io/jahziel1/assetgen-imagen` (Qwen-Image + Z-Image en un solo venv): **un solo pod** aunque haya assets de qwen y de zimage
(baja los pesos de cada modelo y corre sus scripts uno tras otro; disco 180 GB si son los dos). `construir` usa `ghcr.io/jahziel1/assetgen-3d`
(TRELLIS.2 + Puppeteer): **3D y rig en el mismo pod** (el 3D se baja al terminar y despues se riggea ahi mismo; reanudable: lo que ya existe en `3d/` o `rig/` no se rehace).
Con el preset `alto` (postproceso local en Blender entre el 3D y el rig) siguen siendo dos pods, ambos con la imagen `assetgen-3d`.
Para volver a las 4 imagenes viejas (un pod por etapa): `ASSETGEN_IMAGENES=viejas assetgen ...` (o `IMAGENES_VIEJAS = True` en `assetgen_cli/presets.py`).

## Que hace por dentro

1. Elige GPU de la lista con precio <= tope (consulta `runpodctl gpu list`). Si un pod no obtiene SSH en 15 minutos (o entra en error), lo borra
   y prueba con la siguiente GPU. **Si 2 pods seguidos no arrancan, el CLI se detiene y avisa** (no sigue probando GPUs).
   Deteccion de "listo": la fuente principal es `runpodctl pod get` (estado, `uptimeSeconds`, `ssh.ip`/`ssh.port`) mas un intento real de conexion
   (`ssh -o ConnectTimeout=5 ... true`) cada ~6 s; `runpodctl ssh info` solo se consulta como respaldo (se queda en "not ready" aunque el pod ya corra).
   Cada intento (tambien los fallidos y los que no consiguieron pod) queda en `costos.jsonl` con `resultado`, `estado_final`, hitos y costo.
2. Crea el pod `assetgen-<tipo>-<fecha>` y arma el apagado de seguridad (en el pod y un vigilante local aparte).
3. Sube las entradas por `tar` sobre SSH, corre `prepare_weights.sh`, y despues la generacion.
4. Descarga resultados y `stats.json` (tiempos de arranque, generacion, costo y estadisticas por asset).
5. Borra el pod (con verificacion) y agrega una linea a `costos.jsonl`.

Los pods se llaman `assetgen-*`; `estado` y `limpiar` solo tocan esos.

## assetgen.yaml

Ver `ejemplos/assetgen.yaml`. Campos: `preset`, `estilo`, `sufijo_estilo`, `variantes`, `modelo_imagen`, `modelo_3d`, `tamano_imagen`, `camara`, `presupuesto_poligonos`, `gpu_modo` y la lista `assets`
(`nombre`, `tipo` = humanoide|mecanico|criatura|otro, `plantilla` (obligatoria en mecanico), `prompt` y/o `imagen`, `modelo_imagen`, `pose`, `tamano_imagen`, `literal`). `3d --config` usa los assets con `imagen`.
Animacion guiada por video no esta en el CLI.

## Notas

- Las GPUs de 24 GB sirven para TRELLIS.2 (usa `low_vram`), pero `rig` solo usa GPUs de 40 GB o mas.
- `rig` nombra la salida `<nombre>_rigged.glb` y deja `etapas.json` con tiempos y VRAM por etapa de Puppeteer.
- Limites: el apagado local depende de que la Mac siga encendida; el del pod funciona aunque no.

## Prueba de punta a punta (2026-11-23)

3 imagenes con `3d --preset medio` (A6000, ~$0.69/h) y 1 GLB con `rig` (A40, ~$0.59/h):

| Etapa | 3d (3 assets) | rig (1 asset) |
|---|---|---|
| Crear pod -> SSH listo (descarga de la imagen de ~11 GB) | 209 s | 232 s |
| SSH -> pesos listos | 130 s | 339 s |
| **Arranque total** | **339 s (5.7 min)** | **571 s (9.5 min)** |
| Generacion | 423 s (incluye 111 s de carga del modelo) | 90 s (esqueleto 33 s + skinning 42 s + export) |
| Total pod | 764 s | 665 s |
| Costo | $0.147 | $0.109 |

Cuando el host no tiene la imagen en cache el arranque puede tardar mas; por eso el limite por defecto es de 15 minutos.

## Prueba de punta a punta del pipeline completo (2026-11-23, `pruebas4/e2e2`)

yaml de 3 assets, preset `medio`: `arquero` (humanoide), `carreta` (mecanico, `vehiculo`), `molino` (mecanico, `rotor`).

| Etapa | GPU | Tiempo | Costo |
|---|---|---|---|
| `imagenes` (3 x 4 variantes, Qwen Lightning) | A40 | 429 s (SSH 192 s, pesos 234 s, ~8.5 s/imagen en A40) | $0.070 |
| `construir` -> 3d (3 assets) | A40 | 795 s (generacion 479 s) | $0.152 |
| `construir` -> rig (solo el arquero; `rig.sh` salio con rc=139 por el segfault de bpy y se acepto por existir el GLB) | A40 | 549 s (generacion 87 s) | $0.089 |
| Blender local (retarget / plantillas) | - | 7 s / 2 s / 2 s | 0 |
| **Total** | | | **$0.311** |

Resultados (reimportados en Blender con `blender/verificar_glb.py`): `arquero` 5 clips Quaternius, 52 huesos, textura 2048, 48k caras, 7.6 MB, camina bien;
`carreta` clip `rodar` (4 ruedas giran en bucle) con la rueda lejana oscura y unas motas sueltas; `molino` clip `girar` con la plantilla `rotor` arrastrando
trozos de la torre (ventana, techo) junto con las aspas: **falla parcialmente**.

## Prueba de punta a punta con Z-Image + ControlNet y la nueva deteccion de arranque (2026-11-24, `pruebas4/e2e3`)

`velador` (humanoide, `zimage`, `pose: a`, prompt literal), `lobo` y `caballo` (criatura, `qwen`); preset `medio`; todo en A6000 ($0.69/h).
`imagenes` (un pod por modelo) -> eleccion manual de variante -> `construir`.

| Etapa | Contenedor `running` | SSH listo (pod get + ssh) | Pesos listos | Generacion | Costo |
|---|---|---|---|---|---|
| imagenes zimage (1 x 4, 1536, ControlNet) | 154 s | 154.9 s | ~5 min | ~25-30 s/imagen | $0.078 |
| imagenes qwen (2 x 4, 1328) | 146 s | 147.3 s | 376 s | ~4 min (8 imagenes) | $0.100 |
| 3d (3 assets, medio) | 277 s | 279.4 s | 364 s | 407 s | $0.149 |
| rig (3 assets) | 311 s | 314.0 s | ~1.5 min | 3 x ~80 s | $0.127 |
| **Total** | | | | | **$0.454** |

La deteccion con `pod get` + conexion SSH marco "listo" 1-3 s despues de que el contenedor paso a `running`, siempre con `fuente_listo: pod_get`
(`ssh info` no hizo falta). Casi todo el arranque es la descarga de la imagen (~11 GB). Resultados: `velador` con 5 clips Quaternius (camina bien, 48k caras, 22 huesos
en el rig), `lobo` y `caballo` riggeados con Puppeteer (41 y 38 huesos) sin clips; `rig.sh` salio con rc=139 (segfault de bpy al cerrar) y se acepto por existir el GLB.

## Prueba con las imagenes unificadas (2026-11-24, `pruebas4/e2e4`)

`barril` (objeto, `qwen`) y `velador` (humanoide, `zimage`, `pose: a`), preset `medio`, 2 variantes, A40 ($0.59/h). `imagenes` -> `construir`.

| Pod | Imagen | Contenedor `running` / SSH listo | Pesos listos | Generacion | Costo |
|---|---|---|---|---|---|
| imagenes (qwen + zimage, **1 pod**) | `assetgen-imagen` (10.5 GB) | 262 s / 264.5 s | qwen 312.6 s, zimage 411.3 s | 71 s + 89 s (2 + 2 imagenes) | $0.083 |
| 3d + rig (**1 pod**) | `assetgen-3d` (15.3 GB) | ~385 s / 387.4 s | trellis2 470.5 s, puppeteer 1059.8 s | 3D 481 s (2 assets), rig 84 s (1 humanoide) | $0.189 |
| **Total** | | 652 s de arranque en 2 pods | | | **$0.272** |

Frente a e2e3 (A6000, 4 pods, arranque 154 + 147 + 279 + 314 = 894 s, $0.454 con 3 assets y 8 + 4 imagenes): 2 pods en vez de 4 y 242 s menos de arranque total (-27 %).
El arranque por pod no baja (imagen: 264 s vs ~150 s, mismo tamano de capas propias, otro host; 3d: 387 s para 7.9 GB propios frente a 279-314 s para 3.5 y 4.3 GB), lo que
ahorra tiempo es no repetir pods. La A40 es mas lenta que la A6000 en 3D (481 s para 2 assets). Resultado: barril 47k caras sin clips; velador con 5 clips Quaternius, 22 huesos, 48k caras
(hojas en `pruebas4/e2e4/hoja_imagenes.jpg` y `hoja_resultado.jpg`).

## Modo remaster (agnostico): `assetgen remaster analizar` (paso 1 de la fase 5)

Local, sin GPU y sin costo. Lee un modelo **original** (el que el remake debe reemplazar) y escribe en `--salida`:
`restricciones.yaml` (todo lo medible del archivo, cada valor con su origen `medido`/`inferido`/`manual`, y una seccion `motor:` para los limites
que no se ven en el archivo), `renders/{frente,lado,espalda}.png` (1024 px, ortograficos, misma escala, fondo gris), `vistas.jpg` (3 x 333 px,
entrada del editor de imagenes del paso 2) y `medidas.json` (lo medido sin interpretar).

```bash
assetgen remaster analizar --original modelo.glb --salida analisis/                 # glb/gltf (lector propio) o fbx (via Blender)
assetgen remaster analizar --original piezas_vb_ib/ --layout layout.yaml \          # 3dmigoto: .vb/.ib + layout del vertice
    [--paletas p.json] [--pose datos.json#poses.A] [--esqueleto esq.json|esq.glb] [--texturas img] [--textura-render img] \
    [--arriba +y] [--frente +z] --salida analisis/
```

- Extrae: esqueleto (nombres, jerarquia, reposo), maximo de pesos por vertice, vertices/triangulos y huesos usados por pieza, numero de piezas, tipo de indice
  y maximo de vertices por pieza implicito (uint16 -> 65535), altura/ancho/profundidad, eje arriba y frente, materiales/texturas (resolucion), canales de UV.
- Adaptadores: `remaster_gltf.py` (glb/gltf nativo: conteos exactos de GPU, skins, tipo de indice), `blender/remaster/medir_blender.py` (fbx), `remaster_3dmigoto.py`
  (`.vb/.ib` con `layout.yaml`, ver `ejemplos/layout_3dmigoto.yaml`). Un 3DMigoto no trae esqueleto: los huesos del buffer son indices locales a la paleta de cada
  pieza; con `--paletas` (pieza -> huesos globales) se pasan a globales, con `--pose` (matriz por hueso global) se ensambla el cuerpo (skinning lineal) y con
  `--esqueleto` se aporta la jerarquia (el yaml lo marca `manual`).
- Los ejes salen del layout/linea de comandos (`manual`) o se infieren (`inferido`): arriba = +y en glTF/fbx; frente = hacia donde sobresalen los pies. Revisa `vistas.jpg`.
- Los renders muestran la pose que tenga el archivo (no necesariamente un reposo/A-pose).
- Pruebas: `cli/tests/test_remaster.py` (datos sinteticos, sin Blender).

### `assetgen remaster concepto` (paso 5b): imagen de concepto del remake

Con GPU (un pod de la imagen `assetgen-imagen`, sin rebuild: `gen_concepto.py` y `prepare_weights.sh` se suben al pod). Editor de imagenes
**Qwen-Image-Edit-2511** (Apache 2.0, multi-imagen, `QwenImageEditPlusPipeline`) con la LoRA Lightning de 8 pasos de lightx2v (Apache 2.0).

```bash
assetgen remaster concepto --vistas analisis/ [--referencias ref1.png ref2.png] [--prompt "descripcion del personaje"] \
    [--variantes 4] --salida concepto/ [--presupuesto 0.4] [--nube any]
```

- `--referencias` es opcional. Sin ellas el estilo sale solo del texto (estilo base "realistic AAA remake, PBR" + `--prompt` con el diseno);
  recomendado cuando las referencias podrian ser copiadas tal cual por el editor (el frente tiende a clonar una referencia).
- Por variante (semilla propia): **frente** = editar [frente del original, referencias] (pose A, fondo blanco, 768x1328); **lado** y **espalda** =
  editar [frente generado, lado/espalda del original]. Las vistas del original se recortan, se centran y su fondo liso se pasa a blanco.
- Salidas: `v<n>_{frente,lado,espalda}.png`, `hoja_concepto.jpg` (<= 1000 px, una fila por variante), `concepto.yaml` (modelo, semillas, prompts,
  tiempos, costo) y `seleccion.yaml` (`concepto: <n>`). Tiempos medidos en A40: arranque ~230 s, pesos ~140 s, ~60 s por frente (4 imagenes de entrada), ~30 s por lado/espalda.
- **Modo refinado** (`--base`): parte de un concepto ya generado en vez del original y solo eleva el realismo.
  `assetgen remaster concepto --base concepto_r2/ --base-variante 1 --referencias cara.png brazalete.png tela.png --prompt "..." --variantes 4 --salida concepto_r3/`.
  El frente se edita con [frente base + referencias de MATERIAL]; lado y espalda con [frente refinado + vista base correspondiente]. Las referencias van en
  ese orden (cara/piel, cuero+metal, tela), se reducen a 512 px y el prompt dice que son solo de material (no copiar diseno, pose ni camara). La hoja
  lleva una fila inicial "B" con el concepto base. Brazos pedidos a ~45 grados. **Limitacion medida**: con 3 recortes de detalle el frente copio la composicion de
  las referencias en las 4 variantes (ver reporte de la ronda 3); si pasa, probar 1 sola referencia, o solo texto, o una por pasada.
- Pruebas sin GPU: `cli/tests/test_concepto.py`.
