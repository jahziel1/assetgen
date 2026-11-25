"""hornear.py -- UV (un atlas para todas las partes) y horneado desde la malla alta hacia las mallas del juego.

    blender -b -P hornear.py -- uv       --trab DIR [--margen 0.004] [--densidad-cabeza 1.6] [--angulo 66] [--partes cabeza,cuerpo]
    blender -b -P hornear.py -- hornear  --trab DIR [--res 4096] [--extrusion 0.02] [--rayo 0.04] [--ao 1] [--muestras 16]
    blender -b -P hornear.py -- exportar --trab DIR --salida DIR [--nombre principe_juego]

Entradas en --trab: altas.blend (alto_<parte> con material y texturas), retopo_<parte>.blend (juego_<parte>, de retopologia.py).
Cada parte se hornea por separado contra SU malla alta (la cabeza no ve el cuerpo ni al reves) a imagenes con el mismo diseno de atlas;
despues se componen por mascara de cobertura exacta y se rellena el margen de las islas (dilatacion) en numpy, para que dos partes
no se pisen el margen. Color en sRGB, normal (espacio tangente) y AO en no-color. Usa GPU Metal si esta disponible."""
import json
import math
import os
import random
import sys
import time

import bmesh
import bpy
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "..", "cli"))
from assetgen_cli import retopo_geom as RG  # noqa: E402

a = sys.argv[sys.argv.index("--") + 1:]
CMD = a[0] if a else ""


def arg(n, d=None):
    return a[a.index(n) + 1] if n in a else d


T0 = time.time()
PARTES = arg("--partes", "cabeza,cuerpo").split(",")


def log(*m):
    print("[%6.1fs]" % (time.time() - T0), *m, flush=True)


def activar(ob, *otros):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in otros:
        o.select_set(True)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)


def traer(ruta, nombres):
    with bpy.data.libraries.load(ruta, link=False) as (src, dst):
        dst.objects = [n for n in src.objects if n in nombres]
    for o in dst.objects:
        bpy.context.scene.collection.objects.link(o)
    return dst.objects


def gpu():
    """Cycles en GPU Metal si hay; devuelve el nombre del dispositivo usado."""
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    try:
        pr = bpy.context.preferences.addons["cycles"].preferences
        pr.compute_device_type = "METAL"
        pr.get_devices()
        usados = []
        for d in pr.devices:
            d.use = d.type == "METAL"
            if d.use:
                usados.append(d.name)
        if not usados:
            raise RuntimeError("sin dispositivos METAL")
        sc.cycles.device = "GPU"
        log("dispositivo de horneado: GPU Metal", usados)
        return "GPU Metal: " + ", ".join(usados)
    except Exception as e:      # noqa: BLE001
        sc.cycles.device = "CPU"
        log("GPU no disponible, CPU:", e)
        return "CPU"


# ------------------------------------------------------------------------------------------------------------------------- uv
def uvs(me):
    n = len(me.loops)
    uv = np.empty(n * 2, np.float32); me.uv_layers[0].data.foreach_get("uv", uv)
    return uv.reshape(-1, 2)


def estadisticas_uv(obs, res):
    out = {}
    random.seed(3)
    for nombre, ob in obs.items():
        me = ob.data
        co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
        uv = uvs(me)
        a3 = a2 = 0.0
        dist = []
        muestras = set(random.sample(range(len(me.polygons)), min(4000, len(me.polygons))))
        for p in me.polygons:
            ls = list(range(p.loop_start, p.loop_start + p.loop_total))
            vs = [me.loops[l].vertex_index for l in ls]
            for k in range(1, len(vs) - 1):
                idx = [0, k, k + 1]
                p3 = [co[vs[i]] for i in idx]; pu = [uv[ls[i]] for i in idx]
                A3 = 0.5 * float(np.linalg.norm(np.cross(p3[1] - p3[0], p3[2] - p3[0])))
                A2 = 0.5 * abs(float((pu[1][0] - pu[0][0]) * (pu[2][1] - pu[0][1]) - (pu[2][0] - pu[0][0]) * (pu[1][1] - pu[0][1])))
                a3 += A3; a2 += A2
                if p.index in muestras and A3 > 1e-10:
                    d = RG.distorsion_triangulo([list(map(float, q)) for q in p3], [list(map(float, q)) for q in pu])
                    if math.isfinite(d):
                        dist.append(d)
        dist.sort()
        out[nombre] = {"area_3d_m2": round(a3, 4), "area_uv": round(a2, 4), "px_por_m": round(RG.densidad_texel(a3, a2, res), 0),
                       "px_por_mm": round(RG.densidad_texel(a3, a2, res) / 1000, 2),
                       "estiramiento_mediana": round(dist[len(dist) // 2], 2) if dist else None,
                       "estiramiento_p95": round(dist[int(len(dist) * 0.95)], 2) if dist else None,
                       "u": [round(float(uv[:, 0].min()), 3), round(float(uv[:, 0].max()), 3)],
                       "v": [round(float(uv[:, 1].min()), 3), round(float(uv[:, 1].max()), 3)]}
    return out


def cmd_uv():
    trab = arg("--trab"); marg = float(arg("--margen", 0.004)); kcab = float(arg("--densidad-cabeza", 1.6))
    ang = float(arg("--angulo", 66)); res = int(arg("--res", 4096))
    bpy.ops.wm.open_mainfile(filepath=os.path.join(trab, "altas.blend"))
    obs = {}
    for p in PARTES:
        o = traer(os.path.join(trab, "retopo_%s.blend" % p), ["juego_" + p])[0]
        while o.data.uv_layers:
            o.data.uv_layers.remove(o.data.uv_layers[0])
        o.data.uv_layers.new(name="UVMap")
        obs[p] = o
    lista = list(obs.values())
    activar(lista[0], *lista)
    ts = bpy.context.scene.tool_settings
    ts.use_uv_select_sync = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    t0 = time.time()
    bpy.ops.uv.smart_project(angle_limit=math.radians(ang), margin_method="FRACTION", island_margin=0.0, area_weight=0.0,
                             correct_aspect=True, scale_to_bounds=False)
    bpy.ops.uv.average_islands_scale()
    bpy.ops.object.mode_set(mode="OBJECT")
    log("smart_project + densidad uniforme %.1fs" % (time.time() - t0))
    if "cabeza" in obs and kcab != 1.0:     # la cabeza (cara) recibe mas pixeles por metro
        me = obs["cabeza"].data
        uv = uvs(me); c = uv.mean(axis=0); uv = (uv - c) * kcab + c
        me.uv_layers[0].data.foreach_set("uv", uv.ravel())
    activar(lista[0], *lista)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    t0 = time.time()
    bpy.ops.uv.pack_islands(udim_source="CLOSEST_UDIM", rotate=True, rotate_method="ANY", scale=True, merge_overlap=False,
                            margin_method="FRACTION", margin=marg, pin=False, shape_method="CONCAVE")
    bpy.ops.object.mode_set(mode="OBJECT")
    t_pack = time.time() - t0
    st = estadisticas_uv(obs, res)
    # islas por componente de UV (aprox: componentes conexas por UV compartido)
    for p, o in obs.items():
        bm = bmesh.new(); bm.from_mesh(o.data); capa = bm.loops.layers.uv[0]
        bm.verts.ensure_lookup_table()
        vistos = set(); cuenta = 0
        for f0 in bm.faces:
            if f0.index in vistos:
                continue
            cuenta += 1; pila = [f0]; vistos.add(f0.index)
            while pila:
                g = pila.pop()
                for e in g.edges:
                    for h in e.link_faces:
                        if h.index in vistos:
                            continue
                        ok = all((next(l[capa].uv for l in g.loops if l.vert == v) - next(l[capa].uv for l in h.loops if l.vert == v)).length < 1e-5
                                 for v in e.verts)
                        if ok:
                            vistos.add(h.index); pila.append(h)
        st[p]["islas"] = cuenta
        bm.free()
    total_uv = sum(v["area_uv"] for v in st.values())
    st["_global"] = {"ocupacion_uv": round(total_uv, 3), "margen_uv": marg, "margen_px": round(marg * res, 1), "tiempo_empaquetado_s": round(t_pack, 1),
                     "angulo": ang, "densidad_cabeza_relativa": kcab, "res": res}
    log("UV:", json.dumps(st))
    json.dump(st, open(os.path.join(trab, "uv_stats.json"), "w"), indent=1)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(trab, "uv.blend"), compress=True)


# ------------------------------------------------------------------------------------------------------------------------- hornear
def material_hornear(ob, imagen):
    """Material con un nodo de imagen activo (destino del horneado) y emision blanca para la mascara."""
    m = bpy.data.materials.new("hornear_" + ob.name); m.use_nodes = True
    nt = m.node_tree
    nodo = nt.nodes.new("ShaderNodeTexImage"); nodo.image = imagen; nodo.name = "destino"
    nt.nodes.active = nodo
    ob.data.materials.clear(); ob.data.materials.append(m)
    return m, nodo


def nueva_imagen(nombre, res, color, srgb, flotante=False):
    im = bpy.data.images.new(nombre, res, res, alpha=False, float_buffer=flotante)
    im.colorspace_settings.name = "sRGB" if srgb else "Non-Color"
    px = np.empty(res * res * 4, np.float32); px.reshape(-1, 4)[:] = color
    im.pixels.foreach_set(px)
    return im


def pixeles(im):
    res = im.size[0]
    px = np.empty(res * res * 4, np.float32); im.pixels.foreach_get(px)
    return px.reshape(res, res, 4)


def hornear_parte(p, tipo, res, extr, rayo, muestras, margen_px, con_fuente=True, **kw):
    """Hornea `tipo` de alto_<p> a juego_<p> en una imagen nueva (margen 0); devuelve array (res,res,4)."""
    bajo = bpy.data.objects["juego_" + p]; alto = bpy.data.objects["alto_" + p]
    sc = bpy.context.scene
    sc.cycles.samples = muestras
    sc.render.bake.margin = margen_px
    sc.render.bake.use_clear = True
    fondo = {"COLOR": (0, 0, 0, 1), "NORMAL": (0.5, 0.5, 1, 1), "AO": (1, 1, 1, 1), "MASK": (0, 0, 0, 1)}[kw["etq"]]
    im = nueva_imagen("h_%s_%s" % (p, kw["etq"]), res, fondo, kw["etq"] == "COLOR")
    m, nodo = material_hornear(bajo, im)
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    if con_fuente:
        alto.select_set(True)
    bajo.select_set(True); bpy.context.view_layer.objects.active = bajo
    t0 = time.time()
    if kw["etq"] == "MASK":
        # emision blanca en el propio objeto: cobertura exacta de las UV
        nt = m.node_tree
        em = nt.nodes.new("ShaderNodeEmission"); em.inputs["Color"].default_value = (1, 1, 1, 1)
        sal = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
        nt.links.new(em.outputs[0], sal.inputs["Surface"])
        bpy.ops.object.bake(type="EMIT", margin=0, use_selected_to_active=False, use_clear=True)
    else:
        kws = dict(type=tipo, margin=margen_px, margin_type="EXTEND", use_selected_to_active=True, cage_extrusion=extr,
                   max_ray_distance=rayo, use_clear=True)
        if tipo == "DIFFUSE":
            kws["pass_filter"] = {"COLOR"}
        if tipo == "NORMAL":
            kws["normal_space"] = "TANGENT"
        bpy.ops.object.bake(**kws)
    dt = time.time() - t0
    arr = pixeles(im).copy()
    bpy.data.images.remove(im)
    return arr, dt


def dilatar(img, cubierto, pasos, defecto=None):
    """Rellena `pasos` pixeles alrededor de lo cubierto con la media de los vecinos cubiertos (4-conexo)."""
    img = img.copy(); cub = cubierto.copy()
    for _ in range(pasos):
        suma = np.zeros_like(img); n = np.zeros(cub.shape, np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            c = np.roll(cub, (dy, dx), axis=(0, 1)); v = np.roll(img, (dy, dx), axis=(0, 1))
            suma += v * c[..., None]; n += c
        nuevo = (~cub) & (n > 0)
        if not nuevo.any():
            break
        img[nuevo] = suma[nuevo] / n[nuevo][:, None]
        cub = cub | nuevo
    return img, cub


def guardar_png(arr, ruta, srgb, res):
    im = bpy.data.images.new("tmp_" + os.path.basename(ruta), res, res, alpha=False)
    im.colorspace_settings.name = "sRGB" if srgb else "Non-Color"
    out = arr.copy(); out[..., 3] = 1.0
    im.pixels.foreach_set(out.ravel())
    im.filepath_raw = ruta; im.file_format = "PNG"
    sc = bpy.context.scene
    sc.render.image_settings.color_depth = "8"; sc.render.image_settings.color_mode = "RGB"
    im.save()
    bpy.data.images.remove(im)


def cmd_hornear():
    trab = arg("--trab"); res = int(arg("--res", 4096)); extr = float(arg("--extrusion", 0.03)); rayo = float(arg("--rayo", 0.06))
    muestras = int(arg("--muestras", 16)); con_ao = arg("--ao", "1") == "1"; margen_px = int(arg("--margen-px", 16))
    bpy.ops.wm.open_mainfile(filepath=os.path.join(trab, "uv.blend"))
    disp = gpu()
    tiempos = {"dispositivo": disp}
    mascaras, datos = {}, {}
    for p in PARTES:
        m, dt = hornear_parte(p, "EMIT", res, extr, rayo, 1, 0, con_fuente=False, etq="MASK")
        mascaras[p] = m[..., 0] > 0.5; tiempos["mascara_" + p] = round(dt, 1)
        log("mascara", p, "px", int(mascaras[p].sum()), "%.1fs" % dt)
    pasos = [("COLOR", "DIFFUSE"), ("NORMAL", "NORMAL")] + ([("AO", "AO")] if con_ao else [])
    for etq, tipo in pasos:
        for p in PARTES:
            arr, dt = hornear_parte(p, tipo, res, extr, rayo, muestras if etq != "AO" else max(32, muestras), 0, etq=etq)
            datos[(etq, p)] = arr; tiempos["%s_%s" % (etq.lower(), p)] = round(dt, 1)
            log("hornead", etq, p, "%.1fs" % dt)
    # composicion: la primera parte gana si hubiese solape de islas (se cuenta)
    solape = int(np.logical_and.reduce([mascaras[p] for p in PARTES]).sum()) if len(PARTES) > 1 else 0
    # texeles donde el rayo no impacto (color casi negro): se tratan como no cubiertos y se rellenan por dilatacion
    valido, huecos = {}, {}
    for p in PARTES:
        c = datos[("COLOR", p)][..., :3].max(axis=2)
        valido[p] = mascaras[p] & (c >= 0.004)
        huecos[p] = int((mascaras[p] & ~valido[p]).sum())
    cub = np.zeros((res, res), bool)
    fondos = {"COLOR": (0, 0, 0, 1), "NORMAL": (0.5, 0.5, 1, 1), "AO": (1, 1, 1, 1)}
    final = {}
    for etq, _ in pasos:
        img = np.zeros((res, res, 4), np.float32); img[:] = fondos[etq]
        cub = np.zeros((res, res), bool)
        for p in reversed(PARTES):
            img[mascaras[p]] = datos[(etq, p)][mascaras[p]]
            cub |= valido[p]
        t0 = time.time()
        img, _c = dilatar(img, cub, margen_px + 8)
        if etq == "NORMAL":
            n = img[..., :3] * 2 - 1
            n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-6)
            img[..., :3] = n * 0.5 + 0.5
        final[etq] = img
        log("compuesto", etq, "dilatado %.1fs" % (time.time() - t0))
    nombres = {"COLOR": ("juego_color.png", True), "NORMAL": ("juego_normal.png", False), "AO": ("juego_ao.png", False)}
    for etq, (nom, srgb) in nombres.items():
        if etq in final:
            guardar_png(final[etq], os.path.join(trab, nom), srgb, res)
    cobertura = float(np.mean(np.logical_or.reduce([mascaras[p] for p in PARTES])))
    info = {"res": res, "extrusion_m": extr, "rayo_m": rayo, "muestras": muestras, "margen_px": margen_px, "tiempos_s": tiempos,
            "cobertura_atlas": round(cobertura, 3), "px_cubiertos": {p: int(mascaras[p].sum()) for p in PARTES}, "solape_px": solape,
            "px_sin_impacto_rellenados": huecos, "ao": con_ao}
    json.dump(info, open(os.path.join(trab, "hornear.json"), "w"), indent=1)
    log("horneado listo", json.dumps(info))


# ------------------------------------------------------------------------------------------------------------------------- exportar
def cmd_exportar():
    trab, salida = arg("--trab"), arg("--salida"); nombre = arg("--nombre", "principe_juego")
    os.makedirs(salida, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=os.path.join(trab, "uv.blend"))
    for o in list(bpy.data.objects):
        if o.name.startswith("alto_"):
            bpy.data.objects.remove(o)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m)
    for im in list(bpy.data.images):
        bpy.data.images.remove(im)
    col = bpy.data.images.load(os.path.join(trab, "juego_color.png")); col.colorspace_settings.name = "sRGB"; col.pack()
    nor = bpy.data.images.load(os.path.join(trab, "juego_normal.png")); nor.colorspace_settings.name = "Non-Color"; nor.pack()
    m = bpy.data.materials.new("mat_juego"); m.use_nodes = True
    nt = m.node_tree; bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    t1 = nt.nodes.new("ShaderNodeTexImage"); t1.image = col
    t2 = nt.nodes.new("ShaderNodeTexImage"); t2.image = nor
    nm = nt.nodes.new("ShaderNodeNormalMap"); nm.space = "TANGENT"
    nt.links.new(t1.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(t2.outputs["Color"], nm.inputs["Color"]); nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = 0.85; bsdf.inputs["Metallic"].default_value = 0.0
    for o in bpy.data.objects:
        if o.type == "MESH":
            o.data.materials.clear(); o.data.materials.append(m)
            o.name = {"juego_cabeza": "cabeza", "juego_cuerpo": "cuerpo"}.get(o.name, o.name)
    for o in bpy.data.objects:
        o.select_set(o.type == "MESH")
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(salida, nombre + ".blend"), compress=True)
    bpy.ops.export_scene.gltf(filepath=os.path.join(salida, nombre + ".glb"), export_format="GLB", use_selection=True,
                              export_image_format="AUTO", export_tangents=True, export_apply=False)
    # resumen: retopologia + UV + horneado + vertices reales tras el corte por UV (lo que ve el motor; limite del juego 65535 por pieza)
    res = {"partes": {}}
    for o in bpy.data.objects:
        if o.type != "MESH":
            continue
        me = o.data
        lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", lv)
        uv = uvs(me)
        par = np.concatenate([lv[:, None].astype(np.float64), np.round(uv.astype(np.float64), 6)], axis=1)
        ini = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("loop_start", ini)
        fin = np.append(ini[1:], len(lv))
        res["partes"][o.name] = {"vertices_malla": len(me.vertices), "vertices_con_costuras_uv": int(len(np.unique(par, axis=0))),
                                 "quads_o_caras": len(me.polygons), "triangulos": int((fin - ini - 2).sum())}
    for p in PARTES:
        f = os.path.join(trab, "retopo_%s.json" % p)
        if os.path.exists(f):
            res.setdefault("remallado", {})[p] = json.load(open(f))
    for k, f in (("uv", "uv_stats.json"), ("horneado", "hornear.json")):
        if os.path.exists(os.path.join(trab, f)):
            res[k] = json.load(open(os.path.join(trab, f)))
    res["triangulos_total"] = sum(v["triangulos"] for v in res["partes"].values())
    json.dump(res, open(os.path.join(salida, "retopo.json"), "w"), indent=1)
    log("exportado", nombre, "triangulos", res["triangulos_total"], {k: v["vertices_con_costuras_uv"] for k, v in res["partes"].items()})


{"uv": cmd_uv, "hornear": cmd_hornear, "exportar": cmd_exportar}.get(CMD, lambda: log("comando desconocido", CMD))()
