"""Subcomandos encadenados: `imagenes` (Qwen/Z-Image en un pod) y `construir` (3D -> rig -> Blender local -> final/)."""
import argparse, glob, json, os, shutil, subprocess, tempfile, time

from . import globalcfg, runpod as rp, yamlmini
from .comun import _candidatas, _revisar_presupuesto
from .config import cargar, prompt_completo, modelo_imagen, pose_de, tamano_imagen, PLANTILLAS
from .glbinfo import glb_info
from .presets import *
from .sesion import Sesion, log

RAIZ_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BLENDER_DIR = os.path.join(RAIZ_REPO, "blender")
# Tiempos medidos (e2e3/e2e4): SSH listo 150-270 s, pesos ~150 s por modelo, 15-20 s por imagen. Se toma ~210 s de arranque y la
# estimacion final se multiplica por MARGEN_IMG (1.5) para no obligar a subir --presupuesto sin necesidad.
ARRANQUE_IMG_S = 210
PESOS_IMG_S = 150
MARGEN_IMG = 1.5
# por modelo de imagen: imagen vieja (una por modelo), venv viejo, script del pod, segundos por imagen (A40/A6000 a la resolucion por defecto),
# arranque estimado hasta pesos listos (imagen + SSH + pesos) y segundos extra de pesos cuando se suma a otro modelo en el mismo pod
MODELOS_IMG = {
    "qwen": dict(imagen_vieja=IMG_QWEN, venv_viejo="/root/venv-qwen", script="gen_images.py", seg_img=18, arranque=ARRANQUE_IMG_S + PESOS_IMG_S, pesos_s=PESOS_IMG_S,
                 nombre="Qwen-Image Lightning 8 pasos"),
    "zimage": dict(imagen_vieja=IMG_ZIMAGE, venv_viejo="/root/venv-zimage", script="gen_images_zimage.py", seg_img=20, arranque=ARRANQUE_IMG_S + PESOS_IMG_S, pesos_s=PESOS_IMG_S,
                   nombre="Z-Image Turbo 9 pasos + ControlNet de pose"),
}
VENV_IMAGEN = "/root/venv-imagen"


def venv_imagen(modelo, unificadas=None):
    return VENV_IMAGEN if (UNIFICADAS if unificadas is None else unificadas) else MODELOS_IMG[modelo]["venv_viejo"]


def agrupar_imagenes(por_modelo, unificadas=None):
    """{modelo: [assets]} -> lista de pods [{'imagen': ..., 'modelos': {modelo: [assets]}}].
    Imagen unificada: UN pod con todos los modelos. Imagenes viejas: un pod por modelo (como antes)."""
    unificadas = UNIFICADAS if unificadas is None else unificadas
    por_modelo = {m: l for m, l in por_modelo.items() if l}
    if not por_modelo:
        return []
    if unificadas:
        return [{"imagen": IMG_IMAGEN, "modelos": por_modelo}]
    return [{"imagen": MODELOS_IMG[m]["imagen_vieja"], "modelos": {m: l}} for m, l in por_modelo.items()]


def segundos_imagenes(grupo, nvar):
    """Estimacion de segundos de un pod de imagenes: (arranque + pesos de cada modelo extra + 30 + generacion) x MARGEN_IMG."""
    ms = list(grupo["modelos"])
    seg = max(MODELOS_IMG[m]["arranque"] for m in ms) + sum(MODELOS_IMG[m]["pesos_s"] for m in ms) - max(MODELOS_IMG[m]["pesos_s"] for m in ms)
    seg += 30 + sum(len(grupo["modelos"][m]) * nvar * MODELOS_IMG[m]["seg_img"] for m in ms)
    return seg * MARGEN_IMG


TIPOS_RIG = ("humanoide", "criatura")


def _yaml():
    return yamlmini.yaml()


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
        actual = yamlmini.leer(ruta) or {}
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
    prompts = {}
    for a in assets:
        m = modelo_imagen(cfg, a)
        pose = pose_de(a) if m == "zimage" else "libre"     # la pose A usa ControlNet: solo existe en zimage
        if m == "qwen" and a.get("pose") == "a":
            log(f"AVISO: '{a['nombre']}' pide pose: a pero qwen no tiene ControlNet; se usa solo el texto (A-pose en el prompt)")
        prompts[a["nombre"]] = {"prompt": prompt_completo(cfg, a), "variants": nvar, "modelo": m, "size": tamano_imagen(cfg, a, m), "pose": pose}
    for n, p in prompts.items():
        log(f"prompt {n} [{p['modelo']} {p['size']} pose={p['pose']}]: {p['prompt']}")
    if pend:
        # imagen unificada: UN pod con todos los modelos (viejas: un pod por modelo)
        por_modelo = {}
        for a in pend:
            por_modelo.setdefault(prompts[a["nombre"]]["modelo"], []).append(a)
        grupos = agrupar_imagenes(por_modelo)
        if not args.dry_run:
            with open(os.path.join(out, "prompts.yaml"), "w") as f:
                yaml.safe_dump({"seeds": cfg.get("seeds", [11, 222, 3333, 44444]), "variants": nvar, "prompts": prompts}, f,
                               allow_unicode=True, sort_keys=False)
        resultados = []
        for G in grupos:
            modelos = G["modelos"]
            etiqueta = "+".join(modelos)
            c = _candidatas(args, min_gb=40, solo48=True)
            seg = segundos_imagenes(G, nvar)
            n_tot = sum(len(l) for l in modelos.values())
            for m, lista in modelos.items():
                log(f"modelo {m}: {len(lista)} asset(s) ({', '.join(a['nombre'] for a in lista)}) en el pod {etiqueta} [{G['imagen']}]")
            _revisar_presupuesto(args, c, seg)
            if args.dry_run:
                log("dry-run: candidatas " + ", ".join(f"{n} ${pr:.2f}" for _, n, pr, _ in c)); continue
            stage = tempfile.mkdtemp(prefix="assetgen-img-"); bajado = tempfile.mkdtemp(prefix="assetgen-imgout-")
            try:
                for m, lista in modelos.items():
                    cfg_gen = {"seeds": cfg.get("seeds", [11, 222, 3333, 44444]), "variants": nvar,
                               "prompts": {a["nombre"]: {k: v for k, v in prompts[a["nombre"]].items() if k != "modelo"} for a in lista}}
                    with open(os.path.join(stage, f"prompts_{m}.yaml"), "w") as f:
                        yaml.safe_dump(cfg_gen, f, allow_unicode=True, sort_keys=False)
                t = {}
                disco = max(args.disco, 100 if len(modelos) == 1 else 180)    # pesos de Qwen (~57 GB) + Z-Image (~35 GB) + imagen
                with Sesion("imagenes-" + etiqueta, G["imagen"], c, disco, out, max_min=args.max_min, presupuesto=args.presupuesto,
                            espera_ssh=args.timeout_arranque * 60, comando=f"imagenes {etiqueta} n={n_tot}x{nvar}") as s:
                    t0 = s.t_crear
                    s.subir(stage, [f"prompts_{m}.yaml" for m in modelos], "/root/in")
                    for m in modelos:
                        s.subir_archivo(os.path.join(RAIZ_REPO, "scripts", MODELOS_IMG[m]["script"]), f"/root/gen_images_{m}.py")
                    t["ssh_listo"] = s.hitos.get("ssh_listo")
                    t["fuente_listo"] = s.hitos.get("fuente_listo")
                    gen = []
                    for m, lista in modelos.items():
                        log(f"descargando pesos (prepare_weights.sh {m})")
                        s.ssh(f"/opt/assetgen/prepare_weights.sh {m}")
                        t[f"pesos_listos_{m}"] = round(time.time() - t0, 1)
                        log(f"pesos de {m} listos a los {t[f'pesos_listos_{m}']} s del pod")
                        tg = time.time()
                        s.ssh(f"mkdir -p /root/out/img_{m} && cd /root && flock /root/gpu.lock {venv_imagen(m)}/bin/python /root/gen_images_{m}.py "
                              f"/root/in/prompts_{m}.yaml --out /root/out/img_{m}" + (" --force" if args.forzar else ""))
                        t[f"generacion_{m}"] = round(time.time() - tg, 1)
                        s.bajar(f"/root/out/img_{m}", os.path.join(bajado, m))
                    t["pesos_listos"] = max(v for k, v in t.items() if k.startswith("pesos_listos_"))
                    t["generacion"] = round(sum(v for k, v in t.items() if k.startswith("generacion_")), 1)
                    t["total_pod"] = round(time.time() - t0, 1)
                    gpu, precio = s.gpu, s.precio_h
                for m in modelos:
                    for f in glob.glob(os.path.join(bajado, m, "*.png")):
                        shutil.move(f, os.path.join(out, os.path.basename(f)))
                    try:
                        gen += json.load(open(os.path.join(bajado, m, "stats.json"))).get("images", [])
                    except Exception:
                        pass
                resultados.append({"modelo": " + ".join(MODELOS_IMG[m]["nombre"] for m in modelos), "imagen": G["imagen"], "gpu": gpu, "precio_h": precio,
                                   "tiempos_s": t, "costo_usd": s.costo_total, "imagenes": gen})
            finally:
                shutil.rmtree(stage, ignore_errors=True); shutil.rmtree(bajado, ignore_errors=True)
        if args.dry_run:
            return 0
        if resultados:
            _a_json(os.path.join(out, "stats.json"), {"costo_usd": round(sum(r["costo_usd"] for r in resultados), 4),
                                                       "variantes": nvar, "grupos": resultados})
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


def modo_pod_compartido(unificadas, post, hay_rig):
    """True si 3D y rig pueden ir en un solo pod: imagenes unificadas, hay algo que riggear y no hay postproceso local entre las etapas."""
    return bool(unificadas and hay_rig and not post)


def _args_etapa(args, **kw):
    d = dict(vars(args)); d.update(kw)
    return argparse.Namespace(**d)


def _seco(args, cfg, preset, assets, pendientes, entradas):
    """--seco: valida la config y las entradas y muestra el plan y la estimacion, sin encender nada ni escribir salida."""
    p = PRESETS_3D[preset]
    n_rig = sum(1 for a in pendientes if a.get("tipo") in TIPOS_RIG)
    seg = 0
    compartido = modo_pod_compartido(UNIFICADAS, p.get("post"), n_rig > 0)
    if pendientes:
        seg += ARRANQUE_EST + 90 + len(pendientes) * p["seg_asset"]
    if n_rig:
        seg += (0 if compartido else ARRANQUE_EST) + 120 + n_rig * RIG_SEG_ASSET
    log(f"pods: {'1 (3D + rig en el mismo pod)' if compartido else ('2 (3D y rig por separado)' if n_rig else '1 (solo 3D)')}; imagenes {'unificadas' if UNIFICADAS else 'viejas'}")
    est = args.tope * seg / 3600     # cota superior: precio maximo permitido por hora (no consulta RunPod)
    log(f"SECO (no se enciende nada). proyecto={cfg.get('proyecto')} preset={preset}"
        + (f" (postproceso local: quads ~{p['post']['caras']} + bake {p['post']['tex']}, antes del rig/plantilla)" if p.get("post") else ""))
    for a in assets:
        log(f"  {a['nombre']}: {a.get('tipo')}{' / ' + a['plantilla'] if a.get('plantilla') else ''}  imagen={entradas[a['nombre']]}"
            f"  {'pendiente' if a in pendientes else 'ya existe en final/'}")
    tope = globalcfg.cargar().get("tope_mensual_usd")
    gasto = globalcfg.gasto_mes()
    log(f"estimacion maxima ~{seg / 60:.0f} min x ${args.tope:.2f}/h = ${est:.2f}; gasto del mes ${gasto:.2f}"
        + (f"; tope mensual ${float(tope):.2f} -> {'SE SUPERARIA (pediria confirmacion)' if gasto + est > float(tope) else 'dentro del tope'}"
           if tope is not None else "; sin tope mensual"))
    log(f"config global: {globalcfg.ruta_global()} ({'leida' if os.path.exists(globalcfg.ruta_global()) else 'no existe'})")
    log("validacion ok")
    return 0


def cmd_construir(args):
    from .main import cmd_3d, cmd_rig, cmd_3d_rig
    t_ini = time.time()
    yaml = _yaml()
    cfg = cargar(args.config)
    imgs = os.path.abspath(args.imagenes)
    sel_ruta = args.seleccion or os.path.join(imgs, "seleccion.yaml")
    sel = (yamlmini.leer(sel_ruta) or {}) if os.path.exists(sel_ruta) else {}
    assets = cfg["assets"]
    if args.solo:
        s = set(args.solo.split(",")); assets = [a for a in assets if a["nombre"] in s]
    out = os.path.abspath(args.out); os.makedirs(out, exist_ok=True)
    preset = args.preset or cfg["preset"]
    if preset in PRESETS_RESERVADOS:
        raise SystemExit(f"El preset '{preset}' esta reservado y aun no se puede correr (usa boceto|medio|alto)")
    if preset not in PRESETS_3D:
        raise SystemExit(f"preset desconocido '{preset}' (boceto|medio|alto)")
    if PRESETS_3D[preset].get("solo_3d"):
        raise SystemExit(f"El preset '{preset}' solo sirve para `assetgen 3d` (sin postproceso ni rig); usa boceto|medio|alto")
    post = PRESETS_3D[preset].get("post")

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
    for a in assets:
        if a.get("tipo") == "criatura" and a.get("animales"):
            globalcfg.fuente_animal(a["animales"], a["fuente"])   # falla pronto si falta el pack
    if post or any(a.get("tipo") in ("humanoide", "mecanico") or (a.get("tipo") == "criatura" and a.get("animales")) for a in assets):
        globalcfg.blender_exe()

    final = os.path.join(out, "final")
    if not args.seco:
        os.makedirs(final, exist_ok=True)
    pendientes = [a for a in assets if args.forzar or not os.path.exists(os.path.join(final, a["nombre"] + ".glb"))]
    if args.seco:
        return _seco(args, cfg, preset, assets, pendientes, entradas)
    if not pendientes:
        log("todos los assets ya tienen final/<asset>.glb (usa --forzar para rehacer)")
    # ---- 1) 3D
    stage = os.path.join(out, "_entradas"); shutil.rmtree(stage, ignore_errors=True); os.makedirs(stage)
    for a in pendientes:
        shutil.copy(entradas[a["nombre"]], os.path.join(stage, a["nombre"] + ".png"))   # el .glb se llamara como el asset
    d3 = os.path.join(out, "3d")
    drig = os.path.join(out, "rig")
    # 3D + rig en UN pod (imagen assetgen-3d) cuando hay assets que riggear y no hay postproceso local entre ambas etapas
    # (el preset alto postprocesa la malla en Blender local ANTES del rig: ahi se usan dos pods, ambos con la misma imagen)
    rig_hecho = False
    if pendientes:
        a_rig_pre = [a["nombre"] for a in pendientes if a.get("tipo") in TIPOS_RIG]
        if modo_pod_compartido(UNIFICADAS, post, bool(a_rig_pre)):
            pend3d = {a["nombre"]: os.path.join(stage, a["nombre"] + ".png") for a in pendientes
                      if args.forzar or not os.path.exists(os.path.join(d3, a["nombre"] + ".glb"))}
            rc = cmd_3d_rig(_args_etapa(args, config=None, preset=preset, out=d3), preset, pend3d, a_rig_pre, d3, drig)
            rig_hecho = True
        else:
            if post and a_rig_pre:
                log("preset con postproceso local antes del rig: 3D y rig van en pods separados")
            rc = cmd_3d(_args_etapa(args, input=stage, config=None, preset=preset, out=d3))
        if rc and not rig_hecho:
            return rc
    c3 = _costo_etapa(d3, t_ini)
    # ---- 1b) postproceso local (preset alto): quads + normal map ANTES del rig y de las plantillas, para que las
    #          animaciones se apliquen sobre la malla final. Desde aqui `d3` apunta a la carpeta con el GLB que usan las etapas siguientes.
    fallos = {}
    if post:
        dpost = os.path.join(out, "post"); os.makedirs(dpost, exist_ok=True)
        for a in pendientes:
            n = a["nombre"]; g = os.path.join(d3, n + ".glb"); salida = os.path.join(dpost, n + ".glb")
            if not os.path.exists(g):
                fallos[n] = "TRELLIS.2 no produjo el .glb"; continue
            if os.path.exists(salida) and os.path.getsize(salida) > 0 and not args.forzar and os.path.getmtime(salida) >= os.path.getmtime(g):
                log(f"post/{n}.glb ya existe; no se rehace (usa --forzar)"); continue
            t = time.time()
            ok, rc = _blender("postproceso/postprocesar.py", [g, salida, "--caras", str(post["caras"]), "--tex", str(post["tex"])],
                              os.path.join(out, "logs", n + "_post.log"), salida, timeout=3600)
            if ok:
                log(f"post/{n}.glb listo (remesh a quads ~{post['caras']} + bake {post['tex']}, {time.time() - t:.0f} s)" + (f" [blender rc={rc}]" if rc else ""))
            else:
                fallos[n] = f"fallo en el postproceso; ver {os.path.join(out, 'logs', n + '_post.log')}"
        d3 = dpost
    # ---- 2) rig (solo humanoide y criatura)
    rig_in = os.path.join(out, "_rig_in"); shutil.rmtree(rig_in, ignore_errors=True); os.makedirs(rig_in)
    a_rig = [a for a in pendientes if a.get("tipo") in TIPOS_RIG and a["nombre"] not in fallos]
    for a in a_rig:
        g = os.path.join(d3, a["nombre"] + ".glb")
        if os.path.exists(g):
            shutil.copy(g, rig_in)
        else:
            fallos[a["nombre"]] = "TRELLIS.2 no produjo el .glb"
    if os.listdir(rig_in) and not rig_hecho:
        pres = None if args.presupuesto is None else max(args.presupuesto - c3, 0.05)
        rc = cmd_rig(_args_etapa(args, input=rig_in, out=drig, presupuesto=pres))
    crig = _costo_etapa(drig, t_ini) if os.path.isdir(drig) else 0.0
    # ---- 3) Blender local
    logs = os.path.join(out, "logs")
    tmp_h, tmp_m, tmp_a = os.path.join(out, "retarget"), os.path.join(out, "mecanica"), os.path.join(out, "animales")
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
            rg = os.path.join(drig, n + "_rigged.glb")
            if a.get("animales"):   # clips de Quaternius Ultimate Animated Animals sobre la malla (metodo b: reemplazo de esqueleto)
                if not os.path.exists(rg):
                    fallos[n] = "Puppeteer no produjo el GLB riggeado"; continue
                os.makedirs(tmp_a, exist_ok=True)
                salida = os.path.join(tmp_a, n + ".glb")
                ok, rc = _blender("animales/retarget_animal.py", [rg, globalcfg.fuente_animal(a["animales"], a["fuente"]), salida],
                                  os.path.join(logs, n + "_animales.log"), salida)
            else:
                salida = rg   # sin pack de animacion: queda riggeada sin clips
                ok, rc = os.path.exists(salida), 0
                log(f"AVISO: '{n}' (criatura) queda riggeada sin clips (agrega 'animales:' + 'fuente:' al asset para animarla con Quaternius)")
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
    resumen = {"proyecto": cfg.get("proyecto"), "preset": preset, "postproceso": post, "costo_3d_usd": round(c3, 4), "costo_rig_usd": round(crig, 4),
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
