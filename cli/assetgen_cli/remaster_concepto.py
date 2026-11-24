"""`assetgen remaster concepto`: paso 5b del modo remaster. A partir de las vistas del original (frente/lado/espalda) y de
imagenes de referencia de estilo genera N variantes de imagen de concepto del remake (frente, lado, espalda) con un editor
de imagenes con IA (Qwen-Image-Edit-2511 + LoRA Lightning 8 pasos, Apache 2.0) en un pod de RunPod (imagen assetgen-imagen).
Agnostico: nada de ningun juego va en el codigo; el diseno y el estilo llegan por las imagenes y por --prompt.

Flujo por variante (misma semilla en sus 3 vistas):
  frente  = editar [vista frente del original, referencias de estilo] -> personaje de pie en pose A, fondo blanco
  lado    = editar [frente generado, vista lado del original]         -> misma persona de perfil
  espalda = editar [frente generado, vista espalda del original]      -> misma persona de espaldas

Modo refinado (--base): en vez del original, la entrada es un concepto ya generado (v<n>_frente|lado|espalda.png de una ronda anterior).
  frente  = editar [frente base, referencias de MATERIAL/realismo]     -> misma persona, mas realista, brazos a ~45 grados
  lado    = editar [frente refinado, lado base]                        -> misma persona de perfil
  espalda = editar [frente refinado, espalda base]                     -> misma persona de espaldas

Modo cabeza (--cabeza, requiere --base): primeros planos de la cabeza a partir de las 3 vistas base, para dar mas detalle de rostro
a un paso posterior de 3D. Se recorta la cabeza de cada vista base (cuadrado) y se genera, a 1024x1024:
  frente  = editar [recorte frente base]                   -> retrato de cabeza y hombros de la misma persona, boca cerrada
  lado    = editar [frente generado, recorte lado base]    -> misma persona de perfil, mismo encuadre
  espalda = editar [frente generado, recorte espalda base] -> misma persona de espaldas, mismo encuadre"""
import json, os, shutil, tempfile, time

from . import yamlmini
from .comun import _candidatas, _revisar_presupuesto
from .presets import IMG_IMAGEN
from .sesion import Sesion, log

RAIZ_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VISTAS = ("frente", "lado", "espalda")
EXTS = (".png", ".jpg", ".jpeg", ".webp")
SEMILLAS = [11, 222, 3333, 44444]
SEMILLAS_REFINAR = [5, 66, 777, 8888]   # distintas de la ronda base para que el refinado no repita su ruido
SEMILLAS_CABEZA = [21, 303, 4004, 50505]  # propias del modo cabeza
LADO_CABEZA = 1024                      # lado del recorte/salida cuadrada del modo cabeza
FRAC_CABEZA = 0.26                      # lado del recorte = esta fraccion del alto de la figura (tocado a parte alta del pecho)
LADO_REF = 512                          # lado mayor de las referencias de material (recortes de detalle): menos tokens de imagen
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

ESTILO = ("realistic AAA video game remake character, high-detail PBR materials, physically based lighting, "
          "photoreal skin, fabric and leather")
POSE_A = ("standing straight in a neutral A-pose: both arms straight and clearly spread away from the torso at about 40 degrees, "
          "with a visible gap between each arm and the body, open relaxed hands with separated fingers (not fists), "
          "legs straight and parallel, shoulder-width apart, feet flat on the ground, neutral expression, "
          "nothing held in the hands (no props, no weapon)")
FORMATO = ("full body from head to feet fully in frame, orthographic look with no perspective, on a plain pure white (#FFFFFF) "
           "background: replace any gray or black background of the pictures with pure white, no gradient, no floor, "
           "soft even lighting, no cast shadow on the background")


REFINAR_POSE = ("standing straight in a neutral A-pose: both arms straight and clearly spread away from the torso at about 45 degrees "
                "(wider than in Picture 1), with a clear visible gap between each arm and the body, open relaxed hands with separated "
                "fingers (not fists), legs straight and parallel, shoulder-width apart, feet flat on the ground, neutral expression, "
                "nothing held in the hands (no props, no weapon)")
# papel de cada referencia segun su orden (--referencias cara brazalete tela); con otro numero se usa un texto generico
ROLES_REF = ("photoreal skin, face and beard: realistic pores, subtle skin tone variation, individual beard hairs, natural eyes",
             "leather and metal: embossed worn leather with stitching, scratched metal and the skin of the arm",
             "fabric: woven brocade with relief, fold shadows, wear and fine thread detail, for the baggy trousers and the sleeves")


def _lista_a_texto(nums):
    nums = [str(n) for n in nums]
    return nums[0] if len(nums) == 1 else ", ".join(nums[:-1]) + " and " + nums[-1]


def prompt_frente(n_refs, extra=""):
    """Imagen 1 = vista frente del original; imagenes 2..n_refs+1 = referencias de estilo (opcionales: con 0 el estilo sale solo del texto)."""
    s = (f"Redraw the character in Picture 1 as a {ESTILO}. Keep exactly the design, outfit pieces, colors, silhouette and body "
         "proportions of the character in Picture 1 (head size, limb lengths, clothing layers, accessories and their placement), "
         "but ignore its low resolution, its flat look, its crouched pose and its closed fists. ")
    if n_refs:
        refs = _lista_a_texto(range(2, n_refs + 2))
        s += (f"Take the visual style, face, materials, color palette, realism and rendering quality from the style reference "
              f"picture{'s' if n_refs > 1 else ''} {refs}, but do not copy their pose or their camera. ")
    if extra:
        s += extra.strip().rstrip(".") + ". "
    s += f"The character is {POSE_A}, facing the camera, orthographic front view, {FORMATO}."
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
         f"same colors, materials, proportions and rendering style as Picture 1 ({ESTILO}). {guia}; do not copy its low quality, its flat gray "
         "look, its crouched pose or its staggered, bent legs. ")
    if extra:
        s += extra.strip().rstrip(".") + ". "
    s += f"The character is {POSE_A}, same A-pose as Picture 1, standing still (not walking), {FORMATO}."
    return s


def _roles_ref(n_refs):
    if n_refs == len(ROLES_REF):
        partes = [f"Picture {k + 2} is a material reference for {ROLES_REF[k]}" for k in range(n_refs)]
        return "; ".join(partes) + ". "
    if not n_refs:
        return ""
    refs = _lista_a_texto(range(2, n_refs + 2))
    return f"Picture{'s' if n_refs > 1 else ''} {refs} {'are' if n_refs > 1 else 'is a'} material and realism reference{'s' if n_refs > 1 else ''}. "


def prompt_refinar_frente(n_refs, extra=""):
    """Modo refinado. Imagen 1 = frente ya generado (se conserva); imagenes 2.. = referencias SOLO de material y realismo."""
    s = ("Picture 1 is a character concept. Keep EXACTLY the same person, the same design, the same colors and the same clothes as in "
         "Picture 1, the same full body framing and proportions, on a plain pure white background. Only raise the realism of the materials. ")
    if n_refs:
        s += (_roles_ref(n_refs) + "These pictures are material and realism reference only: do not copy their clothing design, their person, "
              "their hair, their pose or their camera. The face reference shows a person without a hood and with loose hair: do not copy "
              "that, keep the hood of Picture 1. ")
    s += ("Make the skin, face and beard photorealistic, the leather and metal realistic with wear, and the fabric of the baggy trousers "
          "and the sleeves textured with natural folds, while keeping the blue hood with the gold band exactly as in Picture 1. ")
    if extra:
        s += extra.strip().rstrip(".") + ". "
    s += f"The character is {REFINAR_POSE}, facing the camera, orthographic front view, {FORMATO}."
    return s


def prompt_refinar_vista(vista, extra=""):
    """Modo refinado. Imagen 1 = frente refinado; imagen 2 = la misma vista (lado/espalda) de la base, solo para la silueta y las piezas."""
    mirada = "the exact side (profile) view" if vista == "lado" else "the exact back view, seen from behind"
    desde = "the side" if vista == "lado" else "behind"
    s = (f"Show the same person and the same outfit as in Picture 1, seen from {mirada}. Same character, same face, same clothes, same colors, "
         f"same realistic materials (photoreal skin, leather, metal and textured fabric), same proportions and same rendering quality as "
         f"Picture 1. Picture 2 is an earlier version of the same character seen from {desde}: use it only to see how the outfit pieces and "
         "silhouette look from this side; do not copy its flatter materials, and do not copy its arms, which are too close to the body. ")
    if extra:
        s += extra.strip().rstrip(".") + ". "
    s += f"The character is {REFINAR_POSE}, same A-pose as Picture 1, standing still (not walking), {FORMATO}."
    return s


FORMATO_CABEZA = ("head and shoulders portrait from the top of the head to the upper chest with both shoulders visible, head centered and "
                  "filling about 55 to 65 percent of the image height, orthographic look with no perspective, on a plain pure white (#FFFFFF) "
                  "background: no gradient, no shadow, soft even lighting")
IDENTIDAD_CABEZA = ("exactly the same person: same face, facial hair (if any), hair, hood or headwear, clothes and colors, same realistic rendering style "
                    "(realistic AAA video game remake character, high-detail PBR materials, photoreal skin)")


def prompt_cabeza_frente(extra=""):
    """Modo cabeza. Imagen 1 = recorte de la cabeza de la vista frente base (baja resolucion)."""
    s = ("Picture 1 is a low resolution crop of a character's head and upper chest. Redraw it as a sharp, highly detailed close-up portrait of "
         f"{IDENTIDAD_CABEZA}. Add detail and resolution (realistic skin pores, individual hair strands, crisp eyes and fabric weave) "
         "but do not copy the blur of Picture 1. ")
    if extra:
        s += extra.strip().rstrip(".") + ". "
    s += ("Front view, looking straight at the camera, head upright and symmetrical, mouth closed, serious neutral expression, "
          f"eyes open looking forward, sharp detailed face. Framing: {FORMATO_CABEZA}.")
    return s


def prompt_cabeza_vista(vista, extra=""):
    """Modo cabeza. Imagen 1 = retrato frente generado; imagen 2 = recorte de la misma vista (lado/espalda) de la base."""
    if vista == "lado":
        mirada = "the exact side (profile) view, facing sideways"
        guia = ("Picture 2 is a low resolution crop of the same head seen from the side: use it only to see the silhouette of the "
                "hood or headwear in profile; do not copy its blur")
    else:
        mirada = "the exact back view, seen from behind"
        guia = ("Picture 2 is a low resolution crop of the same head seen from behind: use it only to see how the hood or cloth falls at "
                "the back; do not copy its blur")
    s = (f"Show {IDENTIDAD_CABEZA} as in Picture 1, seen from {mirada}. Keep the same framing, scale and level of detail as Picture 1, with the head "
         f"upright. {guia}. ")
    if extra:
        s += extra.strip().rstrip(".") + ". "
    s += f"The person has the mouth closed and a serious neutral expression. Framing: {FORMATO_CABEZA}."
    return s


def encontrar_base(carpeta, variante=1):
    """{'frente','lado','espalda'} de un concepto ya generado: v<variante>_<vista>.png o, si no, <vista>.png en esa carpeta."""
    if not os.path.isdir(carpeta):
        raise SystemExit(f"--base no es una carpeta: {carpeta}")
    out = {}
    for v in VISTAS:
        for nombre in (f"v{variante}_{v}", v):
            for ext in EXTS:
                f = os.path.join(carpeta, nombre + ext)
                if os.path.isfile(f):
                    out[v] = f
                    break
            if v in out:
                break
    faltan = [v for v in VISTAS if v not in out]
    if faltan:
        raise SystemExit(f"en {carpeta} faltan las vistas base: {', '.join(faltan)} (espero v{variante}_frente|lado|espalda.png o frente|lado|espalda.png)")
    return out


def preparar_referencia(origen, destino, lado=LADO_REF):
    """Recorte de material: se conserva tal cual (sin recentrar ni tocar el fondo), solo se reduce a `lado` px como maximo."""
    from PIL import Image
    im = Image.open(origen).convert("RGB")
    k = min(1.0, lado / max(im.size))
    if k < 1:
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    im.save(destino)
    return im.size


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


def recortar_cabeza(origen, destino, lado=LADO_CABEZA):
    """Recorte cuadrado de la cabeza (tocado/cabello a parte alta del pecho) de una vista de cuerpo entero sobre fondo uniforme.
    La figura es lo que difiere del fondo (color de las esquinas). Lado del recorte = FRAC_CABEZA x alto de la figura; arriba = tope - 2 %
    del alto; centrado en x en el centro de la cabeza (filas del primer ~10 % de la figura). Si se sale de la imagen se rellena con el
    color de fondo. Escala a `lado` con LANCZOS. Devuelve (caja del recorte (x0, y0, x1, y1) en la imagen origen, tamano final)."""
    from PIL import Image, ImageChops, ImageFilter
    im = Image.open(origen).convert("RGB")
    w, h = im.size
    esq = [im.getpixel(p) for p in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))]
    fondo = tuple(sorted(c[i] for c in esq)[len(esq) // 2] for i in range(3))
    dif = ImageChops.difference(im, Image.new("RGB", im.size, fondo)).convert("L").point(lambda v: 255 if v > 18 else 0)
    dif = dif.filter(ImageFilter.MedianFilter(5))          # quita pixeles sueltos de ruido (los generadores dejan motas en el fondo)
    caja = dif.getbbox()
    if not caja:
        raise SystemExit(f"no se detecta ninguna figura distinta del fondo en {origen}")
    x0, y0, x1, y1 = caja
    alto = y1 - y0
    banda = dif.crop((0, y0, w, y0 + max(1, round(0.10 * alto)))).getbbox()      # filas del primer ~10 % de la figura
    cx = (banda[0] + banda[2]) / 2 if banda else (x0 + x1) / 2
    cl = max(1, round(FRAC_CABEZA * alto))
    cy0 = y0 - round(0.02 * alto)
    cx0 = round(cx - cl / 2)
    lienzo = Image.new("RGB", (cl, cl), fondo)
    lienzo.paste(im.crop((max(0, cx0), max(0, cy0), min(w, cx0 + cl), min(h, cy0 + cl))), (max(0, -cx0), max(0, -cy0)))
    lienzo = lienzo.resize((lado, lado), Image.LANCZOS)
    lienzo.save(destino)
    return (cx0, cy0, cx0 + cl, cy0 + cl), lienzo.size


def semillas_para(n, base=None):
    s = [int(x) for x in (base or SEMILLAS)]
    while len(s) < n:
        s.append(s[-1] + 1)
    return s[:n]


def plan_trabajos(n_variantes, n_refs, extra="", semillas=None, refinar=False, cabeza=False):
    """Plan sin GPU: lista de trabajos con prompt, semilla e imagenes de entrada.
    Cada imagen de entrada es {'ref': <archivo en el pod relativo a /root/in>} o {'gen': <id de un trabajo anterior>}.
    refinar=True: la entrada es el concepto base (carpeta base/ en el pod) y las referencias son solo de material/realismo.
    cabeza=True: la entrada son los recortes de cabeza (carpeta cabeza/), sin referencias; salida cuadrada."""
    if n_variantes < 1:
        raise SystemExit("--variantes debe ser >= 1")
    sem = semillas_para(n_variantes, semillas or (SEMILLAS_CABEZA if cabeza else SEMILLAS_REFINAR if refinar else None))
    if cabeza:
        n_refs = 0
        carp = "cabeza"
        pf = lambda: prompt_cabeza_frente(extra)
        pv = prompt_cabeza_vista
    else:
        carp = "base" if refinar else "vistas"
        pf = (lambda: prompt_refinar_frente(n_refs, extra)) if refinar else (lambda: prompt_frente(n_refs, extra))
        pv = prompt_refinar_vista if refinar else prompt_vista
    jobs = []
    for i in range(n_variantes):
        v = i + 1
        jobs.append({"id": f"v{v}_frente", "variante": v, "vista": "frente", "seed": sem[i], "prompt": pf(),
                     "imagenes": [{"ref": f"{carp}/frente.png"}] + [{"ref": f"refs/ref{k + 1}.png"} for k in range(n_refs)]})
        for vista in ("lado", "espalda"):
            jobs.append({"id": f"v{v}_{vista}", "variante": v, "vista": vista, "seed": sem[i], "prompt": pv(vista, extra),
                         "imagenes": [{"gen": f"v{v}_frente"}, {"ref": f"{carp}/{vista}.png"}]})
    return jobs


def segundos_estimados(jobs):
    rondas = 2 if any(any("gen" in im for im in j["imagenes"]) for j in jobs) else 1
    return (ARRANQUE_S + 30 + RECARGA_S * (rondas - 1) + len(jobs) * SEG_IMG) * MARGEN


def hoja_concepto(out, n, destino, ancho=1000, base=None, cuadrada=False):
    """Una fila por variante con frente | lado | espalda, <= `ancho` px de ancho, con el numero de variante.
    base = {'frente','lado','espalda': ruta}: fila inicial de comparacion (etiqueta 'B') con el concepto de partida.
    cuadrada=True (modo cabeza): celdas cuadradas."""
    from PIL import Image, ImageDraw, ImageFont
    gap = 4
    celda_w = (ancho - gap * 4) // 3
    celda_h = celda_w if cuadrada else round(celda_w * ALTO / ANCHO)
    filas = ([("B", {v: base[v] for v in VISTAS if v in base})] if base else []) + \
            [(str(i + 1), {v: os.path.join(out, f"v{i + 1}_{v}.png") for v in VISTAS}) for i in range(n)]
    hoja = Image.new("RGB", (gap * 4 + celda_w * 3, gap + len(filas) * (celda_h + gap)), (40, 40, 40))
    try:
        fuente = ImageFont.load_default(size=26)
    except TypeError:
        fuente = ImageFont.load_default()
    d = ImageDraw.Draw(hoja)
    for i, (etiqueta, rutas) in enumerate(filas):
        y = gap + i * (celda_h + gap)
        for j, vista in enumerate(VISTAS):
            f = rutas.get(vista)
            if not f or not os.path.exists(f):
                continue
            im = Image.open(f).convert("RGB")
            im.thumbnail((celda_w, celda_h))
            hoja.paste(im, (gap + j * (celda_w + gap) + (celda_w - im.width) // 2, y))
        d.rectangle([gap, y, gap + 34, y + 34], fill=(0, 0, 0))
        d.text((gap + 8, y + 2), etiqueta, fill=(255, 255, 80), font=fuente)
    hoja.save(destino, quality=88)
    return destino


def cmd_concepto(args):
    yaml = yamlmini.yaml()
    base_dir = getattr(args, "base", None)
    refinar = bool(base_dir)
    cabeza = bool(getattr(args, "cabeza", False))
    if cabeza and not refinar:
        raise SystemExit("--cabeza necesita --base (carpeta con el concepto del que se recorta la cabeza)")
    if refinar:
        vistas = encontrar_base(os.path.abspath(os.path.expanduser(base_dir)), getattr(args, "base_variante", 1))
        if cabeza:
            log("modo cabeza (--cabeza): primeros planos de la cabeza recortada de las 3 vistas base")
        else:
            log("modo refinado (--base): se conserva el concepto base y las referencias solo aportan material/realismo")
    elif getattr(args, "vistas", None):
        vistas = encontrar_vistas(os.path.abspath(os.path.expanduser(args.vistas)))
    else:
        raise SystemExit("hace falta --vistas (original) o --base (concepto ya generado a refinar)")
    refs = listar_referencias(args.referencias)
    if cabeza and refs:
        log("AVISO: --cabeza ignora --referencias (la entrada son solo los recortes de la cabeza)")
        refs = []
    if not refs and not refinar:
        log("sin --referencias: el estilo sale solo del texto del prompt (la vista del original solo da el diseno)")
    if refinar and not refs and not cabeza:
        log("AVISO: --base sin --referencias: solo se re-renderiza la base (sin recortes de material el realismo apenas cambia)")
    if len(refs) > 3:
        log(f"AVISO: {len(refs)} referencias; el editor rinde mejor con <= 3 (cada imagen extra alarga la generacion)")
    nvar = args.variantes
    extra = args.prompt or ""
    jobs = plan_trabajos(nvar, len(refs), extra, refinar=refinar, cabeza=cabeza)
    ancho, alto = (LADO_CABEZA, LADO_CABEZA) if cabeza else (ANCHO, ALTO)
    out = os.path.abspath(os.path.expanduser(args.salida))
    os.makedirs(out, exist_ok=True)
    pend = [j for j in jobs if args.forzar or not os.path.exists(os.path.join(out, j["id"] + ".png"))]
    for j in jobs[:3]:
        log(f"prompt {j['id']}: {j['prompt']}")
    log(f"{len(jobs)} imagen(es) en total, {len(pend)} por generar ({nvar} variante(s) x frente/lado/espalda) a {ancho}x{alto}")
    if args.dry_run or pend:
        c = _candidatas(args, min_gb=40, solo48=True)
        seg = segundos_estimados(pend or jobs)
        _revisar_presupuesto(args, c, seg)
        if args.dry_run:
            log("dry-run: candidatas " + ", ".join(f"{n} ${pr:.2f}" for _, n, pr, _ in c))
            return 0
    recortes = {}
    if cabeza:        # recortes de la cabeza base: se guardan en la salida (fila 'B' de la hoja) y se suben al pod
        rc_dir = os.path.join(out, "recortes_base")
        os.makedirs(rc_dir, exist_ok=True)
        for v in VISTAS:
            recortes[v] = os.path.join(rc_dir, v + ".png")
            recortar_cabeza(vistas[v], recortes[v])
    resultados = {}
    if pend:
        stage = tempfile.mkdtemp(prefix="assetgen-concepto-")
        bajado = tempfile.mkdtemp(prefix="assetgen-conceptoout-")
        try:
            carp = "cabeza" if cabeza else "base" if refinar else "vistas"
            os.makedirs(os.path.join(stage, carp)); os.makedirs(os.path.join(stage, "refs"))
            for v in VISTAS:
                if cabeza:
                    shutil.copy(recortes[v], os.path.join(stage, carp, v + ".png"))
                elif refinar:        # el concepto base ya esta encuadrado a la proporcion de salida: se sube tal cual
                    shutil.copy(vistas[v], os.path.join(stage, carp, v + ".png"))
                else:
                    preparar_imagen(vistas[v], os.path.join(stage, carp, v + ".png"))
            for k, r in enumerate(refs):
                (preparar_referencia if refinar else preparar_imagen)(r, os.path.join(stage, "refs", f"ref{k + 1}.png"))
            existentes = [j for j in jobs if j not in pend and j["vista"] == "frente"]
            if existentes:        # un frente ya generado sirve de entrada para lado/espalda pendientes
                os.makedirs(os.path.join(stage, "gen"))
                for j in existentes:
                    shutil.copy(os.path.join(out, j["id"] + ".png"), os.path.join(stage, "gen", j["id"] + ".png"))
            plan = {"model": MODELO, "steps": PASOS, "width": ancho, "height": alto, "jobs": pend}
            with open(os.path.join(stage, "plan.json"), "w") as f:
                json.dump(plan, f, indent=1, ensure_ascii=False)
            archivos = ["plan.json", carp] + (["refs"] if refs else []) + (["gen"] if existentes else [])
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
    datos = {"modelo": MODELO, "lora": LORA, "licencia": LICENCIA, "pasos": PASOS, "tamano": f"{ancho}x{alto}",
             "modo": "cabeza" if cabeza else "refinado" if refinar else "concepto", "base": [os.path.basename(vistas[v]) for v in VISTAS] if refinar else None,
             "referencias": [os.path.basename(r) for r in refs], "prompt_extra": extra, "variantes": nvar,
             "trabajos": [{k: j[k] for k in ("id", "vista", "seed", "prompt")} | {"entradas": [im.get("ref") or "gen:" + im["gen"] for im in j["imagenes"]]} for j in jobs],
             **{k: resultados[k] for k in ("gpu", "precio_h", "tiempos_s", "costo_usd", "imagenes") if k in resultados}}
    with open(os.path.join(out, "concepto.yaml"), "w") as f:
        yaml.safe_dump(datos, f, allow_unicode=True, sort_keys=False)
    try:
        hoja = hoja_concepto(out, nvar, os.path.join(out, "hoja_concepto.jpg"), base=recortes if cabeza else vistas if refinar else None, cuadrada=cabeza)
    except ImportError:
        hoja = "(sin Pillow no hay hoja)"
    sel = os.path.join(out, "seleccion.yaml")
    if not os.path.exists(sel):
        with open(sel, "w") as f:
            f.write("# variante de concepto elegida (mira hoja_concepto.jpg; 1 = primera fila)\n")
            yaml.safe_dump({"concepto": 1}, f, allow_unicode=True, sort_keys=False)
    log(f"listo: {nvar} variante(s) en {out}\n  hoja: {hoja}   seleccion: {sel}")
    return 0
