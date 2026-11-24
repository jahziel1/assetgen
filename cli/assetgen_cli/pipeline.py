"""Subcomandos encadenados: `imagenes` (Qwen en pod) y `construir` (3D -> rig -> Blender local -> final/)."""
import argparse, glob, json, os, shutil, subprocess, tempfile, time

from . import globalcfg, runpod as rp
from .comun import _candidatas, _revisar_presupuesto
from .config import cargar, prompt_completo, PLANTILLAS
from .glbinfo import glb_info
from .presets import *
from .sesion import Sesion, log

RAIZ_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BLENDER_DIR = os.path.join(RAIZ_REPO, "blender")
IMG_QWEN = "ghcr.io/jahziel1/assetgen-qwen:latest"
TIPOS_RIG = ("humanoide", "criatura")
QWEN_SEG_IMG = 12          # estimacion por imagen en una 48 GB lenta (A6000/A40); L40S ~4 s
QWEN_ARRANQUE = 12 * 60 + 420   # imagen + SSH + descarga de pesos (~58 GB) + carga


def _yaml():
    import yaml
    return yaml


def _a_json(ruta, datos):
    with open(ruta, "w") as f:
        json.dump(datos, f, indent=1, ensure_ascii=False)


# ---------------------------------------------------------------- imagenes
def hoja_contacto(out, nombre, n, destino):
    """Une <nombre>_v1..vN.png en una sola imagen de <=1000 px de ancho, con el numero de variante."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        log("AVISO: sin Pillow no se arman hojas de contacto (pip install pillow)"); return False
    celda = (1000 - 4 * (n + 1)) // n
    hoja = Image.new("RGB", (4 * (n + 1) + celda * n, celda + 8), (40, 40, 40))
    try:
        fuente = ImageFont.load_default(size=26)
    except TypeError:
        fuente = ImageFont.load_default()
    for i in range(n):
        f = os.path.join(out, f"{nombre}_v{i + 1}.png")
        if not os.path.exists(f):
            continue
        im = Image.open(f).convert("RGB"); im.thumbnail((celda, celda))
        x = 4 + i * (celda + 4)
        hoja.paste(im, (x, 4))
        d = ImageDraw.Draw(hoja)
        d.rectangle([x, 4, x + 34, 38], fill=(0, 0, 0))
        d.text((x + 8, 6), str(i + 1), fill=(255, 255, 80), font=fuente)
    hoja.save(destino)
    return True


def escribir_seleccion(out, nombres):
    """seleccion.yaml: asset -> variante elegida (por defecto 1). No pisa elecciones existentes."""
    yaml = _yaml()
    ruta = os.path.join(out, "seleccion.yaml")
    actual = {}
    if os.path.exists(ruta):
        actual = yaml.safe_load(open(ruta)) or {}
    for n in nombres:
        actual.setdefault(n, 1)
    with open(ruta, "w") as f:
        f.write("# asset: variante elegida (mira hoja_<asset>.png). Edita y corre `assetgen construir`.\n")
        yaml.safe_dump(actual, f, allow_unicode=True, sort_keys=False)
    return ruta


def cmd_imagenes(args):
    yaml = _yaml()
    cfg = cargar(args.config)
    nvar = args.variantes or int(cfg.get("variantes", 4))
    assets = [a for a in cfg["assets"] if a.get("prompt")]
    for a in cfg["assets"]:
        if not a.get("prompt"):
            log(f"AVISO: '{a['nombre']}' no tiene 'prompt'; se omite")
    if args.solo:
        sel = set(args.solo.split(",")); assets = [a for a in assets if a["nombre"] in sel]
    if not assets:
        raise SystemExit("No hay assets con 'prompt' en el yaml")
    out = os.path.abspath(args.out); os.makedirs(out, exist_ok=True)

    def completo(a):
        return all(os.path.exists(os.path.join(out, f"{a['nombre']}_v{i + 1}.png")) for i in range(nvar))
    pend = [a for a in assets if args.forzar or not completo(a)]
    if len(pend) < len(assets):
        log(f"ya existen las variantes de {len(assets) - len(pend)} asset(s); no se regeneran (usa --forzar)")
    prompts = {a["nombre"]: {"prompt": prompt_completo(cfg, a), "variants": nvar} for a in assets}
    for n, p in prompts.items():
        log(f"prompt {n}: {p['prompt']}")
    if pend:
        c = _candidatas(args, min_gb=40, solo48=True)
        seg = QWEN_ARRANQUE + 120 + len(pend) * nvar * QWEN_SEG_IMG
        _revisar_presupuesto(args, c, seg)
        if args.dry_run:
            log("dry-run: candidatas " + ", ".join(f"{n} ${pr:.2f}" for _, n, pr, _ in c)); return 0
        stage = tempfile.mkdtemp(prefix="assetgen-img-")
        try:
            cfg_gen = {"size": int(cfg.get("tamano_imagen", 1024)), "variants": nvar,
                       "seeds": cfg.get("seeds", [11, 222, 3333, 44444]),
                       "prompts": {a["nombre"]: prompts[a["nombre"]] for a in pend}}
            with open(os.path.join(stage, "prompts.yaml"), "w") as f:
                yaml.safe_dump(cfg_gen, f, allow_unicode=True, sort_keys=False)
            shutil.copy(os.path.join(stage, "prompts.yaml"), os.path.join(out, "prompts.yaml"))
            t = {}
            with Sesion("imagenes", IMG_QWEN, c, max(args.disco, 100), out, max_min=args.max_min, presupuesto=args.presupuesto,
                        espera_ssh=args.timeout_arranque * 60, comando=f"imagenes n={len(pend)}x{nvar}") as s:
                t0 = s.t_crear
                s.subir(stage, ["prompts.yaml"], "/root/in")
                s.subir_archivo(os.path.join(RAIZ_REPO, "scripts", "gen_images.py"), "/root/gen_images_local.py")
                log("descargando pesos (prepare_weights.sh qwen, ~58 GB)")
                s.ssh("/opt/assetgen/prepare_weights.sh qwen")
                t["pesos_listos"] = round(time.time() - t0, 1)
                log(f"pesos listos a los {t['pesos_listos']} s del pod")
                tg = time.time()
                s.ssh("mkdir -p /root/out/img && cd /root && flock /root/gpu.lock /root/venv-qwen/bin/python /root/gen_images_local.py "
                      "/root/in/prompts.yaml --out /root/out/img" + (" --force" if args.forzar else ""))
                t["generacion"] = round(time.time() - tg, 1)
                s.bajar("/root/out/img", out)
                t["total_pod"] = round(time.time() - t0, 1)
                t["ssh_listo"] = s.hitos.get("ssh_listo")
                gpu, precio = s.gpu, s.precio_h
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        gen = {}
        try:
            gen = json.load(open(os.path.join(out, "stats.json"))).get("images", [])
        except Exception:
            pass
        _a_json(os.path.join(out, "stats.json"), {"modelo": "Qwen-Image Lightning 8 pasos", "gpu": gpu, "precio_h": precio, "tiempos_s": t,
                                                   "costo_usd": s.costo_total, "variantes": nvar, "imagenes": gen})
    elif args.dry_run:
        log("dry-run: nada que generar"); return 0
    faltan = [a["nombre"] for a in assets if not completo(a)]
    if faltan:
        raise SystemExit(f"Faltan variantes de: {', '.join(faltan)}")
    for a in assets:
        hoja_contacto(out, a["nombre"], nvar, os.path.join(out, f"hoja_{a['nombre']}.png"))
    ruta = escribir_seleccion(out, [a["nombre"] for a in assets])
    log(f"listo: {len(assets)} asset(s) x {nvar} variantes en {out}\n  hojas: hoja_<asset>.png   seleccion: {ruta}")
    return 0


# ---------------------------------------------------------------- construir
def _blender(script, args_script, log_path, esperado, timeout=900):
    exe = globalcfg.blender_exe()
    cmd = [exe, "-b", "-P", os.path.join(BLENDER_DIR, script), "--", *args_script]
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as lf:
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
    ok = os.path.exists(esperado) and os.path.getsize(esperado) > 0
    return ok, p.returncode


def _costo_etapa(carpeta, desde):
    """costo_usd de stats.json si se escribio en esta corrida (mtime >= desde)."""
    r = os.path.join(carpeta, "stats.json")
    try:
        if os.path.getmtime(r) >= desde:
            return float(json.load(open(r)).get("costo_usd", 0))
    except Exception:
        pass
    return 0.0


def _args_etapa(args, **kw):
    d = dict(vars(args)); d.update(kw)
    return argparse.Namespace(**d)


def cmd_construir(args):
    from .main import cmd_3d, cmd_rig
    t_ini = time.time()
    yaml = _yaml()
    cfg = cargar(args.config)
    imgs = os.path.abspath(args.imagenes)
    sel_ruta = args.seleccion or os.path.join(imgs, "seleccion.yaml")
    sel = (yaml.safe_load(open(sel_ruta)) or {}) if os.path.exists(sel_ruta) else {}
    assets = cfg["assets"]
    if args.solo:
        s = set(args.solo.split(",")); assets = [a for a in assets if a["nombre"] in s]
    out = os.path.abspath(args.out); os.makedirs(out, exist_ok=True)
    preset = args.preset or cfg["preset"]

    # ---- validar todo ANTES de gastar
    entradas = {}
    for a in assets:
        n = a["nombre"]
        if n in sel:
            f = os.path.join(imgs, f"{n}_v{int(sel[n])}.png")
        elif a.get("imagen"):
            f = a["imagen"]
        else:
            raise SystemExit(f"'{n}': no esta en {sel_ruta} y no tiene 'imagen' en el yaml (corre `assetgen imagenes`)")
        if not os.path.exists(f):
            raise SystemExit(f"'{n}': no existe la imagen {f}")
        entradas[n] = f
    if any(a.get("tipo") == "humanoide" for a in assets):
        globalcfg.ual1()
    if any(a.get("tipo") in ("humanoide", "mecanico") for a in assets):
        globalcfg.blender_exe()

    final = os.path.join(out, "final"); os.makedirs(final, exist_ok=True)
    pendientes = [a for a in assets if args.forzar or not os.path.exists(os.path.join(final, a["nombre"] + ".glb"))]
    if not pendientes:
        log("todos los assets ya tienen final/<asset>.glb (usa --forzar para rehacer)")
    # ---- 1) 3D
    stage = os.path.join(out, "_entradas"); shutil.rmtree(stage, ignore_errors=True); os.makedirs(stage)
    for a in pendientes:
        shutil.copy(entradas[a["nombre"]], os.path.join(stage, a["nombre"] + ".png"))   # el .glb se llamara como el asset
    d3 = os.path.join(out, "3d")
    if pendientes:
        rc = cmd_3d(_args_etapa(args, input=stage, config=None, preset=preset, out=d3))
        if rc:
            return rc
    c3 = _costo_etapa(d3, t_ini)
    # ---- 2) rig (solo humanoide y criatura)
    rig_in = os.path.join(out, "_rig_in"); shutil.rmtree(rig_in, ignore_errors=True); os.makedirs(rig_in)
    a_rig = [a for a in pendientes if a.get("tipo") in TIPOS_RIG]
    fallos = {}
    for a in a_rig:
        g = os.path.join(d3, a["nombre"] + ".glb")
        if os.path.exists(g):
            shutil.copy(g, rig_in)
        else:
            fallos[a["nombre"]] = "TRELLIS.2 no produjo el .glb"
    drig = os.path.join(out, "rig")
    if os.listdir(rig_in):
        pres = None if args.presupuesto is None else max(args.presupuesto - c3, 0.05)
        rc = cmd_rig(_args_etapa(args, input=rig_in, out=drig, presupuesto=pres))
    crig = _costo_etapa(drig, t_ini) if os.path.isdir(drig) else 0.0
    # ---- 3) Blender local
    logs = os.path.join(out, "logs")
    tmp_h, tmp_m = os.path.join(out, "retarget"), os.path.join(out, "mecanica")
    for a in pendientes:
        n, tipo = a["nombre"], a.get("tipo")
        dst = os.path.join(final, n + ".glb")
        if n in fallos:
            continue
        t = time.time()
        if tipo == "humanoide":
            rg = os.path.join(drig, n + "_rigged.glb")
            if not os.path.exists(rg):
                fallos[n] = "Puppeteer no produjo el GLB riggeado"; continue
            os.makedirs(tmp_h, exist_ok=True)
            salida = os.path.join(tmp_h, n + ".glb")
            ok, rc = _blender("retarget/retarget_ual.py", ["--target", rg, "--out", salida, "--source", globalcfg.ual1()],
                              os.path.join(logs, n + "_retarget.log"), salida)
        elif tipo == "mecanico":
            os.makedirs(tmp_m, exist_ok=True)
            salida = os.path.join(tmp_m, n + ".glb")
            ok, rc = _blender("mecanica/animar_mecanico.py", [os.path.join(d3, n + ".glb"), a["plantilla"], salida],
                              os.path.join(logs, n + f"_{a['plantilla']}.log"), salida)
        elif tipo == "criatura":
            salida = os.path.join(drig, n + "_rigged.glb")   # sin pack de animacion para criaturas todavia
            ok, rc = os.path.exists(salida), 0
            log(f"AVISO: '{n}' (criatura) queda riggeada sin clips (la animacion de criaturas no esta automatizada)")
        else:
            salida = os.path.join(d3, n + ".glb"); ok, rc = os.path.exists(salida), 0
        if ok:
            shutil.copy(salida, dst)
            log(f"final/{n}.glb listo ({tipo}, {time.time() - t:.0f} s en Blender)" + (f" [blender rc={rc}]" if rc else ""))
        else:
            fallos[n] = f"fallo en Blender ({tipo}); ver {os.path.join(logs, n + '_*.log')}"
    shutil.rmtree(stage, ignore_errors=True); shutil.rmtree(rig_in, ignore_errors=True)
    # ---- 4) resumen
    c_img = 0.0
    try:
        c_img = float(json.load(open(os.path.join(imgs, "stats.json"))).get("costo_usd", 0))
    except Exception:
        pass
    resumen = {"proyecto": cfg.get("proyecto"), "preset": preset, "costo_3d_usd": round(c3, 4), "costo_rig_usd": round(crig, 4),
               "costo_imagenes_usd": round(c_img, 4), "costo_total_usd": round(c3 + crig + c_img, 4),
               "nota_costo": "costo por asset = parte proporcional de cada etapa (imagenes, 3D, rig)", "assets": []}
    n_tot = len(cfg["assets"])
    for a in assets:
        n, tipo = a["nombre"], a.get("tipo")
        r = {"asset": n, "tipo": tipo, "plantilla": a.get("plantilla")}
        f = os.path.join(final, n + ".glb")
        if os.path.exists(f) and n not in fallos:
            gi = glb_info(f)
            r.update(clips=gi["clips"], caras=gi["caras"], mb=gi["mb"], texturas=gi["texturas"])
            r["costo_usd"] = round(c_img / n_tot + c3 / max(len(pendientes), 1) + (crig / max(len(a_rig), 1) if tipo in TIPOS_RIG else 0), 4)
            r["estado"] = "ok"
        else:
            r["estado"] = "fallo"; r["error"] = fallos.get(n, "sin final")
        resumen["assets"].append(r)
    _a_json(os.path.join(out, "resumen.json"), resumen)
    log(f"resumen: {os.path.join(out, 'resumen.json')} (costo total ${resumen['costo_total_usd']:.3f})")
    for r in resumen["assets"]:
        log(f"  {r['asset']}: {r['estado']} {r.get('clips', r.get('error', ''))} caras={r.get('caras')} MB={r.get('mb')}")
    return 1 if fallos else 0
