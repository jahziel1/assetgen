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

El CLI sube `scripts/run_trellis2.py` del repo al pod, asi que no hace falta reconstruir la imagen para probar cambios.

## Pipeline completo (yaml -> final/<asset>.glb)

```bash
assetgen imagenes  --config assetgen.yaml --out proy/imagenes     # Qwen-Image Lightning: N variantes por asset (solo GPUs de 48 GB)
# mira proy/imagenes/hoja_<asset>.png y edita proy/imagenes/seleccion.yaml (asset: variante, por defecto 1)
assetgen construir --config assetgen.yaml --imagenes proy/imagenes --out proy/build
# -> proy/build/final/<asset>.glb con sus clips, proy/build/resumen.json (asset, tipo, clips, caras, MB, costo)
```

`imagenes`: prompt del yaml + sufijo de estilo del proyecto (`estilo`: realista|estilizado|low-poly, o `sufijo_estilo`) + sufijo segun tipo
(`humanoide`: "standing in A-pose with arms slightly away from the body, empty hands, full body"; `mecanico`/`criatura`: "full body, centered").
Escribe `<asset>_v1..vN.png`, `hoja_<asset>.png` (<=1000 px), `seleccion.yaml` (no pisa elecciones existentes) y `prompts.yaml`.
Usa `--variantes N` (def. 4) y `--solo a,b`. El pod tiene disco de >=100 GB (pesos de Qwen ~58 GB).

`construir` valida todo antes de encender nada (imagenes, plantillas, rutas de Quaternius y Blender) y despues:
1. `3d` con el preset del proyecto; el GLB se llama como el asset del yaml (`arquero.glb`), no como la imagen.
2. `rig` (Puppeteer) **solo** para `humanoide` y `criatura`. El exito se decide por la existencia del GLB riggeado (bpy da segfault al cerrar).
3. En local con Blender (`blender/`): `humanoide` -> `blender/retarget/retarget_ual.py` (clips Idle_Loop, Walk_Loop, Run_Loop, Sword_Attack, Death de
   Quaternius UAL1); `mecanico` -> `blender/mecanica/animar_mecanico.py` con su `plantilla` (`vehiculo|catapulta|rotor`); `criatura` queda riggeada sin
   clips; otros tipos quedan como el GLB de 3D.
Es reanudable: no repite lo que ya existe (`--forzar` para rehacer). Intermedios en `<out>/3d`, `rig`, `retarget`, `mecanica`, `logs`.

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
# Imagenes -> GLB con TRELLIS.2 (imagen ghcr.io/jahziel1/assetgen-trellis2)
assetgen 3d --input imagenes/ --preset medio --out salida/3d --presupuesto 1.0

# GLB -> GLB riggeado con Puppeteer (imagen ghcr.io/jahziel1/assetgen-puppeteer)
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
| `alto` | `1024_cascade` | 4096 | ~150k |

## Que hace por dentro

1. Elige GPU de la lista con precio <= tope (consulta `runpodctl gpu list`). Si un pod no obtiene SSH en 15 minutos (o entra en error), lo borra
   y prueba con la siguiente GPU.
2. Crea el pod `assetgen-<tipo>-<fecha>` y arma el apagado de seguridad (en el pod y un vigilante local aparte).
3. Sube las entradas por `tar` sobre SSH, corre `prepare_weights.sh`, y despues la generacion.
4. Descarga resultados y `stats.json` (tiempos de arranque, generacion, costo y estadisticas por asset).
5. Borra el pod (con verificacion) y agrega una linea a `costos.jsonl`.

Los pods se llaman `assetgen-*`; `estado` y `limpiar` solo tocan esos.

## assetgen.yaml

Ver `ejemplos/assetgen.yaml`. Campos: `preset`, `estilo`, `sufijo_estilo`, `variantes`, `camara`, `presupuesto_poligonos`, `gpu_modo` y la lista `assets`
(`nombre`, `tipo` = humanoide|mecanico|criatura|otro, `plantilla` (obligatoria en mecanico), `prompt` y/o `imagen`). `3d --config` usa los assets con `imagen`.
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
