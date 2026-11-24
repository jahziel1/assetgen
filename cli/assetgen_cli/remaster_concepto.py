"""`assetgen remaster concepto`: paso 5b del modo remaster. A partir de las vistas del original (frente/lado/espalda) y de
imagenes de referencia de estilo genera N variantes de imagen de concepto del remake (frente, lado, espalda) con un editor
de imagenes con IA (Qwen-Image-Edit-2511 + LoRA Lightning 8 pasos, Apache 2.0) en un pod de RunPod (imagen assetgen-imagen).
Agnostico: nada de ningun juego va en el codigo; el diseno y el estilo llegan por las imagenes y por --prompt.

Flujo por variante (misma semilla en sus 3 vistas):
  frente  = editar [vista frente del original, referencias de estilo] -> personaje de pie en pose A, fondo blanco
  lado    = editar [frente generado, vista lado del original]         -> misma persona de perfil
  espalda = editar [frente generado, vista espalda del original]      -> misma persona de espaldas"""
import json, os, shutil, tempfile, time

from . import yamlmini
from .comun import _candidatas, _revisar_presupuesto
from .presets import IMG_IMAGEN
from .sesion import Sesion, log

RAIZ_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VISTAS = ("frente", "lado", "espalda")
EXTS = (".png", ".jpg", ".jpeg", ".webp")
SEMILLAS = [11, 222, 3333, 44444]
ANCHO, ALTO = 768, 1328               # ~1328 px de alto, multiplos de 16
PASOS = 8
MODELO = "Qwen/Qwen-Image-Edit-2511"
LORA = "lightx2v/Qwen-Image-Edit-2511-Lightning (8 pasos V1.0 bf16)"
LICENCIA = "Apache 2.0"
# tiempos medidos de la imagen unificada (arranque+pesos de Qwen ~360 s) y estimados para el editor (peso ~57 GB)
ARRANQUE_S = 210 + 170
RECARGA_S = 90            # recarga del transformer entre la ronda de frentes y la de lado/espalda
SEG_IMG = 30              # por imagen (A40/A6000, 8 pasos, 2-4 imagenes de entrada)
MARGEN = 1.5
VENV = "/root/venv-imagen"

POSE_A = ("standing upright in a neutral A-pose: both arms straight and clearly spread away from the torso at about 40 degrees, "
          "with a visible gap between each arm and the body, open relaxed hands with fingers slightly spread (not fists), "
          "legs straight and only slightly apart, feet flat on the ground, no weapon or object in the hands")
FORMATO = ("full body from head to feet fully in frame, orthographic look with no perspective, on a plain pure white (#FFFFFF) "
           "background: replace any gray or black background of the pictures with pure white, no gradient, no floor, "
           "soft even lighting, no cast shadow on the background")


def _lista_a_texto(nums):
    nums = [str(n) for n in nums]
    return nums[0] if len(nums) == 1 else ", ".join(nums[:-1]) + " and " + nums[-1]


def prompt_frente(n_refs, extra=""):
    """Imagen 1 = vista frente del original; imagenes 2..n_refs+1 = referencias de estilo."""
    s = ("Redraw the character in Picture 1 as a high quality realistic AAA game character. Keep exactly the design, outfit "
         "pieces, silhouette and body proportions of the character in Picture 1 (head size, limb lengths, clothing layers, "
         "accessories and their placement). ")
    if n_refs:
        refs = _lista_a_texto(range(2, n_refs + 2))
        s += (f"Take the visual style, face, materials, color palette, realism and rendering quality from the style reference "
              f"picture{'s' if n_refs > 1 else ''} {refs}, but do not copy their pose or their camera. ")
    if extra:
        s += extra.strip().rstrip(".") + ". "
    s += f"The character is {POSE_A}, facing the camera, front view, {FORMATO}."
    return s


def prompt_vista(vista, extra=""):
    """Imagen 1 = frente generado; imagen 2 = vista (lado/espalda) del original, solo para la silueta y las piezas."""
    if vista == "lado":
        mirada = "the exact side (profile) view"
        guia = "Picture 2 is a low quality render of the original model seen from the side: use it only to see how the outfit pieces and silhouette look from the side"
    else:
        mirada = "the exact back view, seen from behind"
        guia = "Picture 2 is a low quality render of the original model seen from behind: use it only to see how the outfit pieces and silhouette look from behind"
    s = (f"Show the same person and the same outfit as in Picture 1, seen from {mirada}. Same character, same face and hair, same clothes, "
         f"same colors, materials, proportions and rendering style as Picture 1. {guia}; do not copy its low quality, its flat gray "
         "look or its crouched pose. ")
    if extra:
        s += extra.strip().rstrip(".") + ". "
    s += f"The character is {POSE_A}, {FORMATO}."
    return s


def encontrar_vistas(carpeta):
    """{'frente': ruta, 'lado': ruta, 'espalda': ruta} buscando <vista>.(png|jpg|...) en la carpeta o en su subcarpeta renders/."""
    if not os.path.isdir(carpeta):
        raise SystemExit(f"--vistas no es una carpeta: {carpeta}")
    out = {}
    for base in (carpeta, os.path.join(carpeta, "renders")):
        if not os.path.isdir(base):
            continue
        for f in sorted(os.listdir(base)):
            nombre, ext = os.path.splitext(f)
            if ext.lower() in EXTS and nombre.lower() in VISTAS and nombre.lower() not in out:
                out[nombre.lower()] = os.path.join(base, f)
    faltan = [v for v in VISTAS if v not in out]
    if faltan:
        raise SystemExit(f"en {carpeta} faltan las vistas: {', '.join(faltan)} (espero frente/lado/espalda .png|.jpg; salen de `assetgen remaster analizar`)")
    return out


def listar_referencias(rutas):
    """Acepta archivos y/o carpetas; devuelve las imagenes (orden dado)."""
    res = []
    for r in rutas or []:
        r = os.path.abspath(os.path.expanduser(r))
        if os.path.isdir(r):
            res += [os.path.join(r, f) for f in sorted(os.listdir(r)) if f.lower().endswith(EXTS)]
        elif os.path.isfile(r) and r.lower().endswith(EXTS):
            res.append(r)
        else:
            raise SystemExit(f"referencia no valida (archivo de imagen o carpeta): {r}")
    return res


def preparar_imagen(origen, destino, ancho=ANCHO, alto=ALTO, margen=0.06):
    """Recorta el personaje (todo lo que difiere del fondo de las esquinas) y lo centra en un lienzo del mismo color de fondo con la
    proporcion de la salida, para que el editor vea la figura grande y con la misma proporcion que se pedira. Devuelve el tamano final."""
    from PIL import Image, ImageChops
    im = Image.open(origen).convert("RGB")
    w, h = im.size
    esq = [im.getpixel(p) for p in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))]
    fondo = tuple(sorted(c[i] for c in esq)[len(esq) // 2] for i in range(3))
    dif = ImageChops.difference(im, Image.new("RGB", im.size, fondo)).convert("L").point(lambda v: 255 if v > 18 else 0)
    caja = dif.getbbox()
    if caja and (caja[2] - caja[0]) * (caja[3] - caja[1]) > 0.02 * w * h:
        im = im.crop(caja)
    cw, ch = im.size
    # el fondo uniforme del original pasa a blanco (si no, el editor lo copia al fondo del concepto)
    bg = Image.new("RGB", im.size, fondo)
    mascara = ImageChops.difference(im, bg).convert("L").point(lambda v: 0 if v > 18 else 255)
    im = Image.composite(Image.new("RGB", im.size, (255, 255, 255)), im, mascara)
    fondo = (255, 255, 255)
    k = min(ancho * (1 - 2 * margen) / cw, alto * (1 - 2 * margen) / ch)
    nuevo = im.resize((max(1, round(cw * k)), max(1, round(ch * k))), Image.LANCZOS)
    lienzo = Image.new("RGB", (ancho, alto), fondo)
    lienzo.paste(nuevo, ((ancho - nuevo.width) // 2, (alto - nuevo.height) // 2))
    lienzo.save(destino)
    return lienzo.size


def semillas_para(n, base=None):
    s = [int(x) for x in (base or SEMILLAS)]
    while len(s) < n:
        s.append(s[-1] + 1)
    return s[:n]


def plan_trabajos(n_variantes, n_refs, extra="", semillas=None):
    """Plan sin GPU: lista de trabajos con prompt, semilla e imagenes de entrada.
    Cada imagen de entrada es {'ref': <archivo en el pod relativo a /root/in>} o {'gen': <id de un trabajo anterior>}."""
    if n_variantes < 1:
        raise SystemExit("--variantes debe ser >= 1")
    sem = semillas_para(n_variantes, semillas)
    jobs = []
    for i in range(n_variantes):
        v = i + 1
        jobs.append({"id": f"v{v}_frente", "variante": v, "vista": "frente", "seed": sem[i],
                     "prompt": prompt_frente(n_refs, extra),
                     "imagenes": [{"ref": "vistas/frente.png"}] + [{"ref": f"refs/ref{k + 1}.png"} for k in range(n_refs)]})
        for vista in ("lado", "espalda"):
            jobs.append({"id": f"v{v}_{vista}", "variante": v, "vista": vista, "seed": sem[i], "prompt": prompt_vista(vista, extra),
                         "imagenes": [{"gen": f"v{v}_frente"}, {"ref": f"vistas/{vista}.png"}]})
    return jobs


def segundos_estimados(jobs):
    rondas = 2 if any(any("gen" in im for im in j["imagenes"]) for j in jobs) else 1
    return (ARRANQUE_S + 30 + RECARGA_S * (rondas - 1) + len(jobs) * SEG_IMG) * MARGEN


def hoja_concepto(out, n, destino, ancho=1000):
    """Una fila por variante con frente | lado | espalda, <= `ancho` px de ancho, con el numero de variante."""
    from PIL import Image, ImageDraw, ImageFont
    gap = 4
    celda_w = (ancho - gap * 4) // 3
    celda_h = round(celda_w * ALTO / ANCHO)
    hoja = Image.new("RGB", (gap * 4 + celda_w * 3, gap + n * (celda_h + gap)), (40, 40, 40))
    try:
        fuente = ImageFont.load_default(size=26)
    except TypeError:
        fuente = ImageFont.load_default()
    d = ImageDraw.Draw(hoja)
    for i in range(n):
        y = gap + i * (celda_h + gap)
        for j, vista in enumerate(VISTAS):
            f = os.path.join(out, f"v{i + 1}_{vista}.png")
            if not os.path.exists(f):
                continue
            im = Image.open(f).convert("RGB")
            im.thumbnail((celda_w, celda_h))
            hoja.paste(im, (gap + j * (celda_w + gap) + (celda_w - im.width) // 2, y))
        d.rectangle([gap, y, gap + 34, y + 34], fill=(0, 0, 0))
        d.text((gap + 8, y + 2), str(i + 1), fill=(255, 255, 80), font=fuente)
    hoja.save(destino, quality=88)
    return destino


def cmd_concepto(args):
    yaml = yamlmini.yaml()
    vistas = encontrar_vistas(os.path.abspath(os.path.expanduser(args.vistas)))
    refs = listar_referencias(args.referencias)
    if not refs:
        log("AVISO: sin --referencias: el frente solo usa la vista del original y el texto")
    if len(refs) > 3:
        log(f"AVISO: {len(refs)} referencias; el editor rinde mejor con <= 3 (cada imagen extra alarga la generacion)")
    nvar = args.variantes
    extra = args.prompt or ""
    jobs = plan_trabajos(nvar, len(refs), extra)
    out = os.path.abspath(os.path.expanduser(args.salida))
    os.makedirs(out, exist_ok=True)
    pend = [j for j in jobs if args.forzar or not os.path.exists(os.path.join(out, j["id"] + ".png"))]
    for j in jobs[:3]:
        log(f"prompt {j['id']}: {j['prompt']}")
    log(f"{len(jobs)} imagen(es) en total, {len(pend)} por generar ({nvar} variante(s) x frente/lado/espalda) a {ANCHO}x{ALTO}")
    if args.dry_run or pend:
        c = _candidatas(args, min_gb=40, solo48=True)
        seg = segundos_estimados(pend or jobs)
        _revisar_presupuesto(args, c, seg)
        if args.dry_run:
            log("dry-run: candidatas " + ", ".join(f"{n} ${pr:.2f}" for _, n, pr, _ in c))
            return 0
    resultados = {}
    if pend:
        stage = tempfile.mkdtemp(prefix="assetgen-concepto-")
        bajado = tempfile.mkdtemp(prefix="assetgen-conceptoout-")
        try:
            os.makedirs(os.path.join(stage, "vistas")); os.makedirs(os.path.join(stage, "refs"))
            for v in VISTAS:
                preparar_imagen(vistas[v], os.path.join(stage, "vistas", v + ".png"))
            for k, r in enumerate(refs):
                preparar_imagen(r, os.path.join(stage, "refs", f"ref{k + 1}.png"))
            existentes = [j for j in jobs if j not in pend and j["vista"] == "frente"]
            if existentes:        # un frente ya generado sirve de entrada para lado/espalda pendientes
                os.makedirs(os.path.join(stage, "gen"))
                for j in existentes:
                    shutil.copy(os.path.join(out, j["id"] + ".png"), os.path.join(stage, "gen", j["id"] + ".png"))
            plan = {"model": MODELO, "steps": PASOS, "width": ANCHO, "height": ALTO, "jobs": pend}
            with open(os.path.join(stage, "plan.json"), "w") as f:
                json.dump(plan, f, indent=1, ensure_ascii=False)
            archivos = ["plan.json", "vistas"] + (["refs"] if refs else []) + (["gen"] if existentes else [])
            t = {}
            with Sesion("concepto", IMG_IMAGEN, c, max(args.disco, 100), out, max_min=args.max_min,
                        presupuesto=args.presupuesto, espera_ssh=args.timeout_arranque * 60,
                        comando=f"remaster concepto n={len(pend)}") as s:
                t0 = s.t_crear
                s.subir(stage, archivos, "/root/in")
                s.subir_archivo(os.path.join(RAIZ_REPO, "scripts", "gen_concepto.py"), "/root/gen_concepto.py")
                # prepare_weights.sh de la imagen publicada puede no traer aun el caso qwen_edit: se sube el del repo (sin rebuild)
                s.subir_archivo(os.path.join(RAIZ_REPO, "scripts", "prepare_weights.sh"), "/root/prepare_weights.sh")
                t["ssh_listo"] = s.hitos.get("ssh_listo")
                log("descargando pesos (prepare_weights.sh qwen_edit)")
                s.ssh("bash /root/prepare_weights.sh qwen_edit")
                t["pesos_listos"] = round(time.time() - t0, 1)
                log(f"pesos listos a los {t['pesos_listos']} s del pod")
                tg = time.time()
                s.ssh(f"mkdir -p /root/out/concepto && cd /root && {VENV}/bin/python /root/gen_concepto.py /root/in/plan.json --in /root/in --out /root/out/concepto")
                t["generacion"] = round(time.time() - tg, 1)
                s.bajar("/root/out/concepto", bajado)
                t["total_pod"] = round(time.time() - t0, 1)
                gpu, precio = s.gpu, s.precio_h
            for f in os.listdir(bajado):
                if f.endswith(".png"):
                    shutil.move(os.path.join(bajado, f), os.path.join(out, f))
            try:
                with open(os.path.join(bajado, "stats.json")) as f:
                    resultados = json.load(f)
            except Exception:
                resultados = {}
            resultados.update({"gpu": gpu, "precio_h": precio, "tiempos_s": t, "costo_usd": s.costo_total})
        finally:
            shutil.rmtree(stage, ignore_errors=True)
            shutil.rmtree(bajado, ignore_errors=True)
    faltan = [j["id"] for j in jobs if not os.path.exists(os.path.join(out, j["id"] + ".png"))]
    if faltan:
        raise SystemExit(f"Faltan imagenes: {', '.join(faltan)}")
    datos = {"modelo": MODELO, "lora": LORA, "licencia": LICENCIA, "pasos": PASOS, "tamano": f"{ANCHO}x{ALTO}",
             "referencias": [os.path.basename(r) for r in refs], "prompt_extra": extra, "variantes": nvar,
             "trabajos": [{k: j[k] for k in ("id", "vista", "seed", "prompt")} | {"entradas": [im.get("ref") or "gen:" + im["gen"] for im in j["imagenes"]]} for j in jobs],
             **{k: resultados[k] for k in ("gpu", "precio_h", "tiempos_s", "costo_usd", "imagenes") if k in resultados}}
    with open(os.path.join(out, "concepto.yaml"), "w") as f:
        yaml.safe_dump(datos, f, allow_unicode=True, sort_keys=False)
    try:
        hoja = hoja_concepto(out, nvar, os.path.join(out, "hoja_concepto.jpg"))
    except ImportError:
        hoja = "(sin Pillow no hay hoja)"
    sel = os.path.join(out, "seleccion.yaml")
    if not os.path.exists(sel):
        with open(sel, "w") as f:
            f.write("# variante de concepto elegida (mira hoja_concepto.jpg; 1 = primera fila)\n")
            yaml.safe_dump({"concepto": 1}, f, allow_unicode=True, sort_keys=False)
    log(f"listo: {nvar} variante(s) en {out}\n  hoja: {hoja}   seleccion: {sel}")
    return 0
