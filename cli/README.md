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

## Uso

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

Ver `ejemplos/assetgen.yaml`. Campos: `preset`, `estilo`, `camara`, `presupuesto_poligonos`, `gpu_modo` y la lista `assets`
(`nombre`, `tipo`, `imagen` o `prompt`, `animaciones`). Por ahora `3d --config` usa `preset` y los assets con `imagen`;
los que solo tienen `prompt` se omiten (la generacion de imagenes llegara como `assetgen imagenes`). Animacion guiada por
video aun no esta en el CLI.

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
