"""`assetgen remaster analizar`: extrae de un modelo original las reglas que debe respetar su remake (restricciones.yaml) y
renderiza referencias en reposo (frente, lado, espalda + vistas.jpg). Local, sin GPU, sin costo. Agnostico: nada de ningun
juego va en el codigo; lo especifico llega por archivos (layout, paletas, pose, esqueleto) que pone el usuario."""
import json, os, shutil, tempfile

from . import remaster_3dmigoto as MIG, remaster_esqueleto as ESQ, remaster_gltf as GLTF, remaster_restricciones as RES, remaster_vistas as VIS
from .remaster_comun import fmt_eje, vector_eje
from .sesion import log


def detectar_formato(original, pedido="auto"):
    if pedido and pedido != "auto":
        return pedido
    if os.path.isdir(original):
        nombres = os.listdir(original)
        if any(n.lower().endswith(".vb") for n in nombres):
            return "3dmigoto"
        for ext, f in ((".glb", "gltf"), (".gltf", "gltf"), (".fbx", "fbx")):
            if any(n.lower().endswith(ext) for n in nombres):
                return f
        raise SystemExit("no reconozco el contenido de la carpeta (espero .vb/.ib, .glb/.gltf o .fbx); usa --formato")
    ext = os.path.splitext(original)[1].lower()
    if ext in (".glb", ".gltf"):
        return "gltf"
    if ext == ".fbx":
        return "fbx"
    if ext in (".vb", ".ib"):
        return "3dmigoto"
    raise SystemExit(f"extension '{ext}' no reconocida; usa --formato gltf|fbx|3dmigoto")


def _archivo_modelo(original, formato):
    """Si `original` es carpeta de gltf/fbx, elige el unico archivo del formato (error si hay varios)."""
    if not os.path.isdir(original):
        return original
    exts = (".glb", ".gltf") if formato == "gltf" else (".fbx",)
    c = sorted(f for f in os.listdir(original) if f.lower().endswith(exts))
    if len(c) != 1:
        raise SystemExit(f"la carpeta tiene {len(c)} archivos {'/'.join(exts)}; pasa --original con el archivo exacto")
    return os.path.join(original, c[0])


def _textura_para_render(texturas):
    if not texturas:
        return None
    if os.path.isdir(texturas):
        c = sorted(f for f in os.listdir(texturas) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        return os.path.join(texturas, c[0]) if c else None
    return texturas


def analizar(args):
    original = os.path.abspath(args.original)
    if not os.path.exists(original):
        raise SystemExit(f"no existe --original: {args.original}")
    salida = os.path.abspath(args.salida)
    os.makedirs(salida, exist_ok=True)
    formato = detectar_formato(original, args.formato)
    log(f"remaster analizar: {os.path.basename(original)} [{formato}]")
    geom = None
    esq_ext = ESQ.leer(os.path.abspath(args.esqueleto)) if args.esqueleto else None
    if formato == "gltf":
        archivo = _archivo_modelo(original, formato)
        mod = GLTF.analizar(archivo)
    elif formato == "fbx":
        archivo = _archivo_modelo(original, formato)
        mod = ESQ.medir_con_blender(archivo)
    elif formato == "3dmigoto":
        if not args.layout:
            raise SystemExit("3dmigoto necesita --layout (yaml con el formato del vertice; ver cli/ejemplos/layout_3dmigoto.yaml)")
        lay = MIG.leer_layout(os.path.abspath(args.layout))
        paletas = MIG.cargar_paletas(os.path.abspath(args.paletas)) if args.paletas else None
        pose = MIG.cargar_pose(os.path.abspath(args.pose)) if args.pose else None
        textos = os.path.abspath(args.texturas) if args.texturas else None
        mod, geom = MIG.analizar(original, lay, paletas, pose, textos)
        archivo = original
    else:
        raise SystemExit(f"formato '{formato}' no soportado (gltf|fbx|3dmigoto)")
    if esq_ext:
        if mod.get("esqueleto"):
            mod["avisos"].append("el original ya traia esqueleto; se usa el de --esqueleto")
        mod["esqueleto"] = esq_ext
    if args.texturas and formato != "3dmigoto":
        mod["avisos"].append("--texturas solo se usa con 3dmigoto; el original ya declara sus materiales")

    ejes = RES.inferir_ejes(mod, args.arriba, args.frente)
    d = RES.construir(mod, ejes)
    ruta_yaml = os.path.join(salida, "restricciones.yaml")
    with open(ruta_yaml, "w", encoding="utf-8") as f:
        f.write(RES.a_yaml(d))
    medidas = {k: v for k, v in mod.items() if k != "muestra_puntos"}
    with open(os.path.join(salida, "medidas.json"), "w") as f:
        json.dump(medidas, f, indent=1, ensure_ascii=False)
    log(f"restricciones.yaml: {len(mod['piezas'])} pieza(s), pesos max {max([p['max_pesos'] for p in mod['piezas']] + [0])}, "
        f"eje arriba {fmt_eje(*ejes['arriba'])} ({ejes['origen_arriba']}), frente {fmt_eje(*ejes['frente'])} ({ejes['origen_frente']})")
    for a in mod["avisos"] + ejes["notas"]:
        log("AVISO: " + a)

    if args.sin_renders:
        return 0
    carpeta = os.path.join(salida, "renders")
    hoja_jpg = os.path.join(salida, "vistas.jpg")
    tmp = None
    try:
        if geom is not None:
            tmp = tempfile.mkdtemp(prefix="assetgen-geom-")
            gj = os.path.join(tmp, "geom.json")
            VIS.escribir_geom(geom, gj, fmt_eje(*ejes["arriba"]), fmt_eje(*ejes["frente"]))
            rutas = VIS.renderizar(carpeta, hoja_jpg, geom=gj, frente_canonico=(0, 0, 1), textura=_textura_para_render(args.textura_render), tam=args.tam)
        else:
            if ejes["arriba"] != (1, 1):
                log(f"AVISO: el render supone Y arriba para {formato}; con --arriba {fmt_eje(*ejes['arriba'])} las vistas pueden salir de lado")
            fr = vector_eje(fmt_eje(*ejes["frente"]))
            rutas = VIS.renderizar(carpeta, hoja_jpg, entrada=archivo, frente_canonico=fr, tam=args.tam)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    log(f"renders: {carpeta} ({args.tam} px) y vistas.jpg")
    return 0


def cmd_remaster(args):
    if args.rcmd == "concepto":
        from .remaster_concepto import cmd_concepto
        return cmd_concepto(args)
    return analizar(args)
