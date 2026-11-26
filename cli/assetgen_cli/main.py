"""assetgen: comando unico. Subcomandos: imagenes, construir, remaster, 3d, rig, estado, gasto, limpiar."""
import argparse, glob, json, os, shutil, sys, tempfile, time

from . import runpod as rp
from .presets import *
from .sesion import Sesion, Abortado, log
from . import globalcfg
from .pipeline import cmd_imagenes, cmd_construir
from .remaster import cmd_remaster
from .comun import _candidatas, _estimar, _revisar_presupuesto

RAIZ_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _guardar_stats(out, datos):
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(datos, f, indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- 3d
SUFIJOS_VISTA = ("_lado", "_espalda")


def planear_3d(imagenes, semillas=None, multivista=False):
    """imagenes {nombre: ruta} -> (base, extras, salidas). Con multivista, `<n>_lado` y `<n>_espalda` son vistas extra de `<n>` (no assets).
    salidas {nombre_base: [nombres de GLB esperados]}: `<n>.glb` sin semillas; `<n>_s<semilla>.glb` con ellas; `<n>_multi.glb` si hay las 2 vistas."""
    extras = {}
    if multivista:
        for n in list(imagenes):
            for suf in SUFIJOS_VISTA:
                if n.endswith(suf) and n[: -len(suf)] in imagenes:
                    extras[n] = imagenes.pop(n)
    salidas = {}
    for n in imagenes:
        sal = [f"{n}_s{x}" for x in semillas] if semillas else [n]
        if multivista and all(n + suf in extras for suf in SUFIJOS_VISTA):
            sal.append(n + "_multi")
        salidas[n] = sal
    return imagenes, extras, salidas


def comando_3d(motor, env):
    """Comando ssh que corre el script del motor en el pod (env: cadena 'K=V K=V')."""
    m = MOTORES_3D[motor]
    return f"mkdir -p {m['out']} && cd /root && {env} flock /root/gpu.lock {m['venv']}/bin/python {m['remoto']} /root/in {m['out']}"


def cmd_3d(args):
    from .config import cargar, modelo_3d as resolver_modelo_3d
    preset, imagenes, cfg = args.preset, {}, {}
    if args.config:
        cfg = cargar(args.config)
        preset = args.preset or cfg["preset"]
        for a in cfg["assets"]:
            if a.get("tipo") == "rig-solo":
                continue
            if a.get("imagen"):
                imagenes[a["nombre"]] = a["imagen"]
            else:
                log(f"AVISO: '{a['nombre']}' solo tiene prompt; falta generar su imagen (corre `assetgen imagenes`). Se omite.")
    if args.input:
        for f in sorted(glob.glob(os.path.join(args.input, "*"))):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                imagenes.setdefault(os.path.splitext(os.path.basename(f))[0], f)
    motor = resolver_modelo_3d(cfg, args)
    if motor == "pixal3d" and args.multivista:
        raise SystemExit("--multivista no aplica a pixal3d (solo trellis2); quita --multivista o usa --modelo-3d trellis2")
    preset = preset or "medio"
    if preset not in PRESETS_3D:
        raise SystemExit(f"preset desconocido '{preset}' (boceto|medio|alto|maximo)")
    out = os.path.abspath(args.out); os.makedirs(out, exist_ok=True)
    semillas = [int(x) for x in args.semillas.split(",") if x.strip()] if args.semillas else []
    imagenes, extras, salidas = planear_3d(imagenes, semillas, args.multivista)
    pendientes = {n: f for n, f in imagenes.items()
                  if args.forzar or not all(os.path.exists(os.path.join(out, g + ".glb")) for g in salidas[n])}
    if len(pendientes) < len(imagenes):
        log(f"ya existen los glb de {len(imagenes) - len(pendientes)} asset(s) en la salida; no se regeneran (usa --forzar)")
    if not pendientes:
        log("nada que generar"); return 0
    extra_env = {}
    if semillas:
        extra_env["SEEDS"] = ",".join(str(x) for x in semillas)
    if args.multivista:
        extra_env["MULTIVISTA"] = "1"
    n_salidas = sum(len(salidas[n]) for n in pendientes)

    c = _candidatas(args)
    p = PRESETS_3D[preset]
    seg = ARRANQUE_EST + 90 + n_salidas * seg_asset_3d(preset, motor)
    log(f"motor 3D: {motor} (imagen {MOTORES_3D[motor]['imagen']})")
    _revisar_presupuesto(args, c, seg)
    if args.dry_run:
        log("dry-run: candidatas " + ", ".join(f"{n} ${pr:.2f}" for _, n, pr, _ in c)); return 0

    stage = tempfile.mkdtemp(prefix="assetgen-in-")
    for n, f in pendientes.items():
        shutil.copy(f, os.path.join(stage, n + os.path.splitext(f)[1].lower()))
        for suf in SUFIJOS_VISTA:
            if n + suf in extras:
                shutil.copy(extras[n + suf], os.path.join(stage, n + suf + os.path.splitext(extras[n + suf])[1].lower()))
    t = {}
    try:
        with Sesion("3d", MOTORES_3D[motor]["imagen"], c, args.disco, out, max_min=args.max_min, presupuesto=args.presupuesto, espera_ssh=args.timeout_arranque * 60,
                    comando=f"3d modelo={motor} preset={preset} n={len(pendientes)}") as s:
            t0 = s.t_crear
            _fase_3d(s, preset, stage, out, t, t0, extra_env, motor)
            gpu, precio = s.gpu, s.precio_h
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    rutas = sorted(glob.glob(os.path.join(out, "*.glb")))
    assets = []
    try:
        assets = json.load(open(os.path.join(out, "stats.json")))
    except Exception:
        pass
    _guardar_stats(out, {"preset": preset, "modelo_3d": motor, "parametros": {**env_3d(preset, motor), **extra_env}, "gpu": gpu, "precio_h": precio, "tiempos_s": t,
                         "costo_usd": s.costo_total, "assets": assets})
    log(f"listo: {len(rutas)} glb en {out}; tiempos {t}")
    return 0


def _fase_3d(s, preset, stage, out, t, t0, extra_env=None, motor="trellis2"):
    """Dentro de una Sesion: sube las imagenes de `stage`, baja los pesos del motor (trellis2|pixal3d), genera y baja los GLB a `out`
    (con pixal3d tambien `<n>.camara.json` y `<n>_entrada.png`)."""
    m = MOTORES_3D[motor]
    env = " ".join(f"{k}={v}" for k, v in {**env_3d(preset, motor), **(extra_env or {})}.items())
    s.subir(stage, sorted(os.listdir(stage)), "/root/in")
    s.subir_archivo(os.path.join(RAIZ_REPO, "scripts", m["script"]), m["remoto"])
    log(f"descargando pesos (prepare_weights.sh {m['pesos']})")
    s.ssh(f"/opt/assetgen/prepare_weights.sh {m['pesos']}")
    t["pesos_listos"] = round(time.time() - t0, 1)
    log(f"pesos listos a los {t['pesos_listos']} s del pod")
    tg = time.time()
    log(f"generando ({preset}: {env})")
    s.ssh(comando_3d(motor, env))
    t["generacion"] = round(time.time() - tg, 1)
    s.bajar(m["out"], out)
    t["total_pod"] = round(time.time() - t0, 1)
    t["ssh_listo"] = s.hitos.get("ssh_listo")
    t["arranque"] = t["pesos_listos"]


# ---------------------------------------------------------------- rig
RIG_RECOGER = r"""mkdir -p /root/out_rig; cp /root/work/*/*_rigged.glb /root/out_rig/ 2>/dev/null; python3 - <<'PY'
import json, glob
d = {}
for f in sorted(glob.glob('/root/stats/*.json')):
    j = json.load(open(f)); d[j['stage']] = {k: v for k, v in j.items() if k != 'stage'}
json.dump(d, open('/root/out_rig/etapas.json', 'w'), indent=1)
PY
ls /root/out_rig"""


def cmd_rig(args):
    glbs = sorted(glob.glob(os.path.join(args.input, "*.glb")))
    if not glbs:
        raise SystemExit(f"No hay .glb en {args.input}")
    out = os.path.abspath(args.out); os.makedirs(out, exist_ok=True)
    pend = [g for g in glbs if args.forzar or not os.path.exists(os.path.join(out, os.path.basename(g)[:-4] + "_rigged.glb"))]
    if not pend:
        log("nada que riggear (ya existen los *_rigged.glb)"); return 0
    c = _candidatas(args, min_gb=40)
    seg = ARRANQUE_EST + 120 + len(pend) * RIG_SEG_ASSET
    _revisar_presupuesto(args, c, seg)
    if args.dry_run:
        log("dry-run: candidatas " + ", ".join(f"{n} ${pr:.2f}" for _, n, pr, _ in c)); return 0
    t = {}
    with Sesion("rig", IMG_PUPPETEER, c, args.disco, out, max_min=args.max_min, presupuesto=args.presupuesto, espera_ssh=args.timeout_arranque * 60,
                comando=f"rig n={len(pend)}") as s:
        t0 = s.t_crear
        s.subir(args.input, [os.path.basename(g) for g in pend], "/root/in")
        fallos = _fase_rig(s, [os.path.basename(g)[:-4] for g in pend], "/root/in", out, t, t0)
        gpu, precio = s.gpu, s.precio_h
    etapas = {}
    try:
        etapas = json.load(open(os.path.join(out, "etapas.json")))
    except Exception:
        pass
    _guardar_stats(out, {"gpu": gpu, "precio_h": precio, "tiempos_s": t, "costo_usd": s.costo_total,
                         "fallos": fallos, "etapas": etapas})
    log(f"listo: {len(glob.glob(os.path.join(out, '*_rigged.glb')))} glb riggeados en {out}; tiempos {t}")
    return 1 if fallos else 0


def _fase_rig(s, nombres, remoto_in, out, t, t0):
    """Dentro de una Sesion: baja pesos de Puppeteer, riggea /root/in/<n>.glb de `remoto_in` y baja los *_rigged.glb a `out`. Devuelve los fallos."""
    log("descargando pesos (prepare_weights.sh puppeteer)")
    s.ssh("/opt/assetgen/prepare_weights.sh puppeteer")
    t["pesos_listos_rig"] = round(time.time() - t0, 1)
    t.setdefault("pesos_listos", t["pesos_listos_rig"])
    tg = time.time()
    fallos = []
    for n in nombres:
        log(f"rig {n}")
        rc = s.ssh(f"cd /root && /root/rig.sh {n} {remoto_in}/{n}.glb", check=False)
        # bpy suele terminar con segfault DESPUES de exportar: manda la existencia del GLB, no el codigo de salida
        if s.ssh(f"test -s /root/work/{n}/{n}_rigged.glb", check=False) != 0:
            fallos.append(n)
        elif rc != 0:
            log(f"aviso: rig.sh de {n} salio con rc={rc} pero el GLB riggeado existe (segfault de bpy al cerrar)")
    t["generacion_rig"] = round(time.time() - tg, 1)
    t.setdefault("generacion", t["generacion_rig"])
    s.ssh(RIG_RECOGER)
    s.bajar("/root/out_rig", out)
    t["total_pod"] = round(time.time() - t0, 1)
    t["ssh_listo"] = s.hitos.get("ssh_listo"); t["arranque"] = t["pesos_listos"]
    return fallos


def cmd_3d_rig(args, preset, pend3d, a_rig, d3, drig):
    """UN pod (imagen assetgen-3d) para 3D + rig: genera los GLB con TRELLIS.2, los riggea ahi mismo con Puppeteer y baja ambos resultados.
    pend3d: {nombre: png} a generar; a_rig: nombres a riggear (su GLB sale del pod o ya esta en d3). Reanudable: las salidas se bajan por etapa."""
    os.makedirs(d3, exist_ok=True); os.makedirs(drig, exist_ok=True)
    p = PRESETS_3D[preset]
    a_rig = [n for n in a_rig if args.forzar or not os.path.exists(os.path.join(drig, n + "_rigged.glb"))]
    if not pend3d and not a_rig:
        log("nada que generar ni riggear"); return 0
    c = _candidatas(args, min_gb=40 if a_rig else 0)
    seg = ARRANQUE_EST + 150 + len(pend3d) * p["seg_asset"] + (len(a_rig) * RIG_SEG_ASSET + 120 if a_rig else 0)
    _revisar_presupuesto(args, c, seg)
    if args.dry_run:
        log("dry-run: candidatas " + ", ".join(f"{n} ${pr:.2f}" for _, n, pr, _ in c)); return 0
    stage = tempfile.mkdtemp(prefix="assetgen-in-")
    for n, f in pend3d.items():
        shutil.copy(f, os.path.join(stage, n + os.path.splitext(f)[1].lower()))
    t3, tr = {}, {}
    fallos = []
    try:
        with Sesion("3d-rig", IMG_TRELLIS, c, max(args.disco, 120), d3, max_min=args.max_min, presupuesto=args.presupuesto,
                    espera_ssh=args.timeout_arranque * 60, comando=f"3d+rig preset={preset} n3d={len(pend3d)} nrig={len(a_rig)}") as s:
            t0 = s.t_crear
            if pend3d:
                _fase_3d(s, preset, stage, d3, t3, t0)
            t_fin3d = time.time()
            if a_rig:
                # GLB a riggear: los recien generados ya estan en el pod; los que existian de antes se suben
                s.ssh("mkdir -p /root/rigin")
                locales = [n + ".glb" for n in a_rig if n not in pend3d and os.path.exists(os.path.join(d3, n + ".glb"))]
                if locales:
                    s.subir(d3, locales, "/root/rigin")
                for n in a_rig:
                    if n in pend3d:
                        s.ssh(f"cp /root/out/trellis2/{n}.glb /root/rigin/ 2>/dev/null; true", check=False)
                faltan = [n for n in a_rig if s.ssh(f"test -s /root/rigin/{n}.glb", check=False) != 0]
                for n in faltan:
                    log(f"AVISO: '{n}' no tiene GLB para riggear (TRELLIS.2 fallo); se omite")
                fallos = _fase_rig(s, [n for n in a_rig if n not in faltan], "/root/rigin", drig, tr, t0)
                fallos += faltan
            gpu, precio = s.gpu, s.precio_h
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    # el costo del pod se reparte entre las dos etapas por tiempo (el arranque cuenta a la primera)
    total = s.costo_total
    dur3 = max((t_fin3d - t0) if pend3d else 0, 0); durr = (time.time() - t_fin3d) if a_rig else 0
    f3 = dur3 / (dur3 + durr) if (dur3 + durr) > 0 and pend3d and a_rig else (1.0 if pend3d else 0.0)
    if pend3d:
        assets = []
        try:
            assets = json.load(open(os.path.join(d3, "stats.json")))
        except Exception:
            pass
        _guardar_stats(d3, {"preset": preset, "parametros": p["env"], "gpu": gpu, "precio_h": precio, "tiempos_s": t3, "pod_compartido": True,
                            "costo_usd": round(total * f3, 4), "assets": assets})
    if a_rig:
        etapas = {}
        try:
            etapas = json.load(open(os.path.join(drig, "etapas.json")))
        except Exception:
            pass
        _guardar_stats(drig, {"gpu": gpu, "precio_h": precio, "tiempos_s": tr, "pod_compartido": True, "costo_usd": round(total * (1 - f3), 4),
                              "fallos": fallos, "etapas": etapas})
    log(f"listo (pod compartido 3d+rig): {len(glob.glob(os.path.join(d3, '*.glb')))} glb 3D, {len(glob.glob(os.path.join(drig, '*_rigged.glb')))} riggeados; "
        f"tiempos 3d={t3} rig={tr}")
    return 1 if fallos else 0


# ---------------------------------------------------------------- estado / limpiar
def cmd_estado(args):
    pods = rp.list_pods()
    if not pods:
        print("Sin pods de assetgen."); return 0
    tot = 0.0
    for p in pods:
        h = rp.costo_hora(p); tot += h
        print(f"{p.get('id')}  {p.get('name')}  {p.get('gpuId') or ''}  "
              f"{p.get('runtimeStatus') or p.get('desiredStatus')}  ${h:.2f}/h  creado {p.get('createdAt', '?')}")
    print(f"Total: {len(pods)} pod(s), ${tot:.2f}/h")
    return 0


def cmd_gasto(args):
    g = globalcfg.gasto_mes(); tope = globalcfg.cargar().get("tope_mensual_usd")
    print(f"Gasto del mes: ${g:.2f}" + (f" de ${float(tope):.2f}" if tope is not None else " (sin tope mensual)") + f"  [{globalcfg.ruta_costos()}]")
    return 0


def cmd_limpiar(args):
    pods = rp.list_pods()
    if not pods:
        print("Nada que limpiar."); return 0
    malos = 0
    for p in pods:
        ok = rp.delete_pod(p["id"])
        print(f"{p['id']} {p.get('name')}: {'borrado' if ok else 'NO SE PUDO BORRAR'}")
        malos += 0 if ok else 1
    return 1 if malos else 0


# ---------------------------------------------------------------- prueba de infraestructura (oculto)
def cmd_smoke(args):
    c = rp.elegir_candidatas(MODOS_GPU["normal"][::-1], args.tope, "SECURE")
    out = os.path.abspath(args.out)
    with Sesion("smoke", args.imagen, c, 30, out, max_min=10, presupuesto=args.presupuesto, espera_ssh=args.espera, comando="smoke") as s:
        s.ssh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; python3 --version; which flock curl tar; free -g | sed -n 2p; ls /opt/assetgen 2>&1 | head -20")
        d = tempfile.mkdtemp(); open(os.path.join(d, "a.txt"), "w").write("hola\n")
        s.subir(d, ["a.txt"], "/root/in"); s.ssh("cat /root/in/a.txt; pgrep -fa safety.sh | head -2")
        s.bajar("/root/in", os.path.join(out, "bajado")); log("viaje de ida y vuelta ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="assetgen", description="Genera assets 3D de juegos en GPUs rentadas (RunPod).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def comunes(p, con_out=True):
        if con_out:
            p.add_argument("--out", required=True, help="carpeta de salida")
        p.add_argument("--modo", choices=["normal", "rapido"], default="normal", help="lista de GPUs (normal: A6000 -> A40 -> L40S ...; rapido: L40S primero)")
        p.add_argument("--tope", type=float, default=TOPE_PRECIO_H, help=f"precio maximo USD/h (def. {TOPE_PRECIO_H})")
        p.add_argument("--gpu", help="lista explicita de gpuId separada por comas (sustituye al modo)")
        p.add_argument("--nube", choices=["secure", "community", "any"], default="secure")
        p.add_argument("--presupuesto", type=float, help="USD: aborta antes de encender si la estimacion lo excede y apaga si se supera")
        p.add_argument("--max-min", type=int, default=60, help="minutos de vida maxima del pod (apagado de seguridad)")
        p.add_argument("--timeout-arranque", type=int, default=15, help="minutos maximos hasta tener SSH (la imagen pesa 10-15 GB); solo se descarta antes si el pod da error")
        p.add_argument("--disco", type=int, default=80, help="disco del contenedor en GB")
        p.add_argument("--forzar", action="store_true", help="regenerar aunque ya exista la salida")
        p.add_argument("--dry-run", action="store_true", help="solo muestra GPU candidatas y estimacion")
        p.add_argument("--si", action="store_true", help="no pedir confirmacion si se superaria el tope mensual global")

    p3 = sub.add_parser("3d", help="imagen -> GLB con TRELLIS.2 o Pixal3D (--modelo-3d)")
    p3.add_argument("--modelo-3d", choices=list(MODELOS_3D), help="motor 3D: trellis2|pixal3d (def.: modelo_3d del yaml o trellis2; pixal3d no admite --multivista y escribe tambien <n>.camara.json y <n>_entrada.png)")
    p3.add_argument("--input", help="carpeta con PNG/JPG/WEBP")
    p3.add_argument("--config", help="assetgen.yaml (usa los assets con 'imagen')")
    p3.add_argument("--preset", choices=list(PRESETS_3D), help="boceto|medio|alto|maximo (def.: el del yaml o medio; 3d no postprocesa: usa `construir` para el postproceso del preset alto; maximo = voxel 1536, ~1M caras, textura 4096)")
    p3.add_argument("--semillas", help="semillas separadas por coma (ej. 42,7): una salida <nombre>_s<semilla>.glb por semilla")
    p3.add_argument("--multivista", action="store_true", help="EXPERIMENTAL: <n>_lado.png y <n>_espalda.png son vistas de <n>.png; ademas escribe <n>_multi.glb (TRELLIS.2 no trae multi-imagen oficial: se concatenan los tokens DINOv3 de las vistas)")
    comunes(p3); p3.set_defaults(fn=cmd_3d)

    pr = sub.add_parser("rig", help="GLB -> GLB riggeado con Puppeteer")
    pr.add_argument("--input", required=True, help="carpeta con .glb")
    comunes(pr); pr.set_defaults(fn=cmd_rig)

    pi = sub.add_parser("imagenes", help="prompts del yaml -> N variantes por asset con Qwen-Image / Z-Image en UN pod (hoja de contacto + seleccion.yaml)")
    pi.add_argument("--config", required=True, help="assetgen.yaml")
    pi.add_argument("--variantes", type=int, help="variantes por asset (def.: 'variantes' del yaml o 4)")
    pi.add_argument("--solo", help="solo estos assets (separados por coma)")
    comunes(pi); pi.set_defaults(fn=cmd_imagenes)

    pc = sub.add_parser("construir", help="imagenes elegidas -> 3D -> rig (humanoide/criatura) -> Blender local -> final/<asset>.glb + resumen.json")
    pc.add_argument("--config", required=True, help="assetgen.yaml")
    pc.add_argument("--imagenes", required=True, help="carpeta de `assetgen imagenes` (con seleccion.yaml)")
    pc.add_argument("--seleccion", help="seleccion.yaml alterno (def.: <imagenes>/seleccion.yaml)")
    pc.add_argument("--solo", help="solo estos assets (separados por coma)")
    pc.add_argument("--preset", choices=PRESETS_TODOS, help="boceto|medio|alto|aaa (def.: el del yaml). alto = 3D en calidad + quads 20k + normal map; aaa reservado")
    pc.add_argument("--modelo-3d", choices=list(MODELOS_3D), help="motor 3D: trellis2|pixal3d (def.: modelo_3d del yaml o trellis2). Con pixal3d el 3D y el rig van en pods separados")
    pc.add_argument("--seco", action="store_true", help="solo valida config, entradas y tope mensual y muestra el plan; no enciende nada ni escribe salida")
    comunes(pc); pc.set_defaults(fn=cmd_construir)

    prm = sub.add_parser("remaster", help="modo remaster agnostico (local): analizar un modelo original -> restricciones.yaml + renders de referencia")
    rsub = prm.add_subparsers(dest="rcmd", required=True)
    pa = rsub.add_parser("analizar", help="original (glb/gltf/fbx/3dmigoto) -> restricciones.yaml, renders (frente/lado/espalda) y vistas.jpg; sin GPU ni costo")
    pa.add_argument("--original", required=True, help="archivo (.glb/.gltf/.fbx/.vb) o carpeta (piezas .vb/.ib)")
    pa.add_argument("--formato", choices=["auto", "gltf", "fbx", "3dmigoto"], default="auto")
    pa.add_argument("--layout", help="3dmigoto: yaml con el formato del vertice (ver ejemplos/layout_3dmigoto.yaml)")
    pa.add_argument("--salida", required=True, help="carpeta de salida (restricciones.yaml, renders/, vistas.jpg, medidas.json)")
    pa.add_argument("--esqueleto", help="esqueleto aparte (glb/gltf/fbx o json/yaml con huesos) si el original no lo trae")
    pa.add_argument("--paletas", help="3dmigoto: json/yaml {pieza: [hueso global por slot local]}")
    pa.add_argument("--pose", help="3dmigoto: matrices por hueso global de una pose de reposo, 'archivo.json#clave' (12 floats: 3 columnas + origen)")
    pa.add_argument("--texturas", help="3dmigoto: imagen o carpeta con las texturas del original (se registran sus resoluciones en el yaml)")
    pa.add_argument("--textura-render", help="3dmigoto: imagen para pintar el render (debe corresponder a las UV de las piezas; si no, el render sale en gris)")
    pa.add_argument("--arriba", help="eje arriba del archivo (+x|-x|+y|-y|+z|-z); por defecto el del layout o inferido")
    pa.add_argument("--frente", help="eje hacia donde mira el personaje; por defecto el del layout o inferido por los pies")
    pa.add_argument("--tam", type=int, default=1024, help="lado de los renders en px (def. 1024)")
    pa.add_argument("--sin-renders", action="store_true", help="solo restricciones.yaml")
    pa.set_defaults(fn=cmd_remaster)
    pk = rsub.add_parser("concepto", help="paso 5b: vistas del original + referencias de estilo -> N variantes de concepto (frente/lado/espalda) con Qwen-Image-Edit en un pod")
    pk.add_argument("--vistas", help="carpeta con frente/lado/espalda del original (salida de `remaster analizar`; sirve su carpeta o su renders/). Obligatorio salvo con --base")
    pk.add_argument("--base", help="MODO REFINADO: carpeta con un concepto ya generado (v<n>_frente|lado|espalda.png) que se conserva y se eleva en realismo; las --referencias pasan a ser solo de material (orden: cara/piel, cuero+metal, tela)")
    pk.add_argument("--base-variante", type=int, default=1, help="con --base: numero de variante a refinar (v<n>_*.png), def. 1")
    pk.add_argument("--cabeza", action="store_true", help="con --base: en vez del cuerpo, genera primeros planos de la CABEZA (frente/lado/espalda, 1024x1024) recortados de las 3 vistas base, para dar mas detalle de rostro a un paso 3D posterior; ignora --referencias")
    pk.add_argument("--detalle", action="store_true", help="con --base: rehace las imagenes ya generadas (--vistas-detalle de la variante --base-variante) con la MISMA composicion pero mas nitidas (ojos, poros, pelos, tela); salida del mismo aspecto (cuadradas: 1328x1328) y mismos nombres v<n>_<vista>.png; ignora --referencias y --variantes")
    pk.add_argument("--vistas-detalle", default="frente,lado", help="con --detalle: vistas a rehacer, separadas por comas (def. frente,lado)")
    pk.add_argument("--pasos", type=int, default=8, help="pasos del editor (def. 8 = LoRA Lightning, rapido); > 8 = modelo completo SIN LoRA con true_cfg 4.0 (receta de la ficha: 40 pasos), mas nitido y ~10x mas lento; vale para cualquier modo")
    pk.add_argument("--referencias", nargs="+", default=[], help="OPCIONAL: imagenes (o carpetas) de referencia de estilo; ideal <= 3. Sin ellas el estilo sale solo del texto (--prompt y el estilo base)")
    pk.add_argument("--prompt", help="descripcion del personaje/estilo que se anade al prompt base (en ingles rinde mejor)")
    pk.add_argument("--variantes", type=int, default=4, help="variantes (semillas distintas), def. 4")
    pk.add_argument("--salida", required=True, help="carpeta de salida (v<n>_frente|lado|espalda.png, hoja_concepto.jpg, concepto.yaml, seleccion.yaml)")
    comunes(pk, con_out=False); pk.set_defaults(fn=cmd_remaster)

    sub.add_parser("gasto", help="gasto del mes (todos los proyectos) y tope mensual").set_defaults(fn=cmd_gasto)
    sub.add_parser("estado", help="pods de assetgen encendidos y gasto por hora").set_defaults(fn=cmd_estado)
    sub.add_parser("limpiar", help="borra todo pod assetgen-* que haya quedado").set_defaults(fn=cmd_limpiar)

    ps = sub.add_parser("smoke"); ps.add_argument("--out", required=True); ps.add_argument("--tope", type=float, default=0.9)
    ps.add_argument("--imagen", default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
    ps.add_argument("--presupuesto", type=float, default=0.3); ps.add_argument("--espera", type=int, default=420)
    ps.set_defaults(fn=cmd_smoke)

    args = ap.parse_args(argv)
    if args.cmd in ("3d", "rig") and args.cmd == "3d" and not (args.input or args.config):
        ap.error("3d necesita --input o --config")
    try:
        return args.fn(args) or 0
    except Abortado as e:
        log(f"ABORTADO: {e}"); return 2
    except KeyboardInterrupt:
        log("interrumpido por el usuario (pod borrado)"); return 130
    except rp.RunpodError as e:
        log(f"ERROR: {e}"); return 1
