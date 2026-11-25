"""retopologia.py -- malla del juego a partir de un modelo de >1M de triangulos (capas, aristas no-manifold, solapes), por partes.

    blender -b -P retopologia.py -- separar  --entrada alto.blend --trab DIR [--material-cabeza mat_cabeza]
    blender -b -P retopologia.py -- medir    --trab DIR                                  # huecos y cuellos de botella (dedos, manga/torso)
    blender -b -P retopologia.py -- remallar --trab DIR --parte cabeza|cuerpo --voxel 0.002 --caras 3500 [--decimar 0.3]
    blender -b -P retopologia.py -- unir     --trab DIR [--margen 0.004] [--densidad-cabeza 1.6]   # UV: un solo atlas para todas las partes

Sin bpy.ops sobre la malla grande: la separacion por material es numpy (sin `separate`). El remallado por voxeles (cierra capas y
solapes) y QuadriFlow solo se aplican a cada parte por separado, nunca a la malla entera. Las partes se guardan en DIR/*.blend."""
import json
import math
import os
import sys
import time

import bpy
import bmesh
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "..", "cli"))
from assetgen_cli import retopo_geom as RG  # noqa: E402

a = sys.argv[sys.argv.index("--") + 1:]
CMD = a[0] if a else ""


def arg(n, d=None):
    return a[a.index(n) + 1] if n in a else d


T0 = time.time()


def log(*m):
    print("[%6.1fs]" % (time.time() - T0), *m, flush=True)


def activar(ob):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)


def arrays(me):
    """(vertices Nx3, triangulos Mx3, uv por bucle (3M)x2 o None, indice de material M)."""
    n, m = len(me.vertices), len(me.polygons)
    co = np.empty(n * 3, np.float32); me.vertices.foreach_get("co", co)
    lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", lv)
    mi = np.empty(m, np.int32); me.polygons.foreach_get("material_index", mi)
    uv = None
    if me.uv_layers:
        uv = np.empty(len(me.loops) * 2, np.float32); me.uv_layers[0].data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
    return co.reshape(-1, 3), lv.reshape(-1, 3), uv, mi


def malla_de(nombre, co, tri, uv=None, mat=None):
    """Malla nueva desde arrays (sin from_pydata, que es lento con >1M)."""
    me = bpy.data.meshes.new(nombre)
    me.vertices.add(len(co)); me.vertices.foreach_set("co", co.astype(np.float32).ravel())
    me.loops.add(tri.size); me.loops.foreach_set("vertex_index", tri.astype(np.int32).ravel())
    me.polygons.add(len(tri))
    me.polygons.foreach_set("loop_start", np.arange(0, tri.size, 3, dtype=np.int32))
    me.update(calc_edges=True)
    if uv is not None:
        capa = me.uv_layers.new(name="UVMap"); capa.data.foreach_set("uv", uv.astype(np.float32).ravel())
    if mat is not None:
        me.materials.append(mat)
    me.update()
    return me


def caras_de(me):
    """Lista de tuplas de indices de vertice por poligono (triangulos o quads), con numpy."""
    lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", lv)
    ini = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("loop_start", ini)
    fin = np.append(ini[1:], len(lv))
    if len(ini) and np.all(fin - ini == 3):
        return [tuple(t) for t in lv.reshape(-1, 3).tolist()]
    if len(ini) and np.all(fin - ini == 4):
        return [tuple(t) for t in lv.reshape(-1, 4).tolist()]
    return [tuple(lv[i:j].tolist()) for i, j in zip(ini.tolist(), fin.tolist())]


def stats_topologia(me):
    """Aristas abiertas / no-manifold, componentes y Euler (retopo_geom.topologia) mas vertices no-manifold (bmesh)."""
    t = RG.topologia(caras_de(me))
    bm = bmesh.new(); bm.from_mesh(me)
    t["vertices_no_manifold"] = sum(1 for v in bm.verts if not v.is_manifold)
    bm.free()
    return t


# --------------------------------------------------------------------------------------------------------------------- separar
def cmd_separar():
    entrada, trab = arg("--entrada"), arg("--trab")
    nombre_cab = arg("--material-cabeza", "mat_cabeza")
    os.makedirs(trab, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=entrada)
    ob = next(o for o in bpy.data.objects if o.type == "MESH")
    me = ob.data
    assert all(p.loop_total == 3 for p in me.polygons[:1000]), "se esperaba una malla triangulada"
    co, tri, uv, mi = arrays(me)
    mats = list(me.materials)
    log("alto:", len(co), "verts", len(tri), "tris", [m.name for m in mats])
    uvl = uv.reshape(-1, 3, 2) if uv is not None else None
    res = {}
    for k, mat in enumerate(mats):
        sel = mi == k
        tv = tri[sel]
        uniq, inv = np.unique(tv, return_inverse=True)
        nombre = "cabeza" if mat.name == nombre_cab else "cuerpo"
        m2 = malla_de("alto_" + nombre, co[uniq], inv.reshape(-1, 3), uvl[sel].reshape(-1, 2) if uvl is not None else None, mat)
        o2 = bpy.data.objects.new("alto_" + nombre, m2)
        bpy.context.scene.collection.objects.link(o2)
        o2.data.polygons.foreach_set("use_smooth", np.ones(len(m2.polygons), bool))
        res[nombre] = {"verts": len(m2.vertices), "tris": len(m2.polygons), "material": mat.name}
        log(nombre, res[nombre])
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(me)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(trab, "altas.blend"), compress=True)
    json.dump(res, open(os.path.join(trab, "separar.json"), "w"), indent=1)
    log("guardado altas.blend")


# --------------------------------------------------------------------------------------------------------------------- medir
def cmd_medir():
    """Para cada tamano de voxel: remalla una copia de la parte y mide cuanto se aleja de la malla alta (vertice -> vertice alto mas
    cercano). Un voxel que funde un hueco (dedos, manga/torso, daga/pierna) crea membranas: vertices lejos de toda la malla alta."""
    from mathutils import kdtree
    trab, parte = arg("--trab"), arg("--parte")
    tams = [float(x) for x in arg("--voxeles", "0.003,0.002").split(",")]
    bpy.ops.wm.open_mainfile(filepath=os.path.join(trab, "altas.blend"))
    alto = bpy.data.objects["alto_" + parte]
    co = np.empty(len(alto.data.vertices) * 3, np.float32); alto.data.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
    kd = kdtree.KDTree(len(co))
    for i, p in enumerate(co.tolist()):
        kd.insert(p, i)
    kd.balance()
    log("kd listo", len(co), "bbox", co.min(0).round(3).tolist(), co.max(0).round(3).tolist())
    res = []
    for vox in tams:
        w = alto.copy(); w.data = alto.data.copy(); bpy.context.scene.collection.objects.link(w)
        while w.data.uv_layers:
            w.data.uv_layers.remove(w.data.uv_layers[0])
        w.data.materials.clear(); activar(w)
        t0 = time.time()
        w.data.remesh_voxel_size = vox; w.data.remesh_voxel_adaptivity = 0.0
        bpy.ops.object.voxel_remesh()
        dt = time.time() - t0
        vv = np.empty(len(w.data.vertices) * 3, np.float32); w.data.vertices.foreach_get("co", vv); vv = vv.reshape(-1, 3)
        paso = max(1, len(vv) // 150000)
        d = np.array([kd.find(tuple(p))[2] for p in vv[::paso].tolist()], np.float32)
        umbral = 0.004 + vox * 1.5
        lejos = vv[::paso][d > umbral]
        info = {"voxel_m": vox, "verts": len(vv), "caras": len(w.data.polygons), "s": round(dt, 1),
                "dist_p50_mm": round(float(np.percentile(d, 50)) * 1e3, 2), "dist_p99_mm": round(float(np.percentile(d, 99)) * 1e3, 2),
                "dist_max_mm": round(float(d.max()) * 1e3, 1), "umbral_mm": round(umbral * 1e3, 1), "lejos": int(len(lejos)),
                "lejos_pct": round(100.0 * len(lejos) / len(d), 3)}
        if len(lejos):   # agrupar por celdas de 4 cm para localizar donde estan
            celdas = {}
            for p in lejos.tolist():
                celdas.setdefault(tuple(int(math.floor(x / 0.04)) for x in p), []).append(p)
            top = sorted(celdas.items(), key=lambda kv: -len(kv[1]))[:8]
            info["donde"] = [{"n": len(v), "centro_m": np.mean(v, axis=0).round(3).tolist()} for k, v in top]
        log(info)
        res.append(info)
        bpy.data.objects.remove(w)
    json.dump(res, open(os.path.join(trab, "medir_%s.json" % parte), "w"), indent=1)


# --------------------------------------------------------------------------------------------------------------------- remallar
def componentes_cerradas(bm):
    """Reparte las caras en islas; devuelve [(volumen firmado, area, caras)] ordenadas por area."""
    vistos, islas = set(), []
    for f0 in bm.faces:
        if f0.index in vistos:
            continue
        pila, isla = [f0], []
        vistos.add(f0.index)
        while pila:
            g = pila.pop(); isla.append(g)
            for e in g.edges:
                for h in e.link_faces:
                    if h.index not in vistos:
                        vistos.add(h.index); pila.append(h)
        vol = 0.0; area = 0.0
        for f in isla:
            v = [x.co for x in f.verts]
            vol += v[0].dot(v[1].cross(v[2])) / 6.0
            area += f.calc_area()
        islas.append((vol, area, isla))
    islas.sort(key=lambda t: -t[1])
    return islas


def engrosar(me, e):
    """Malla con cada lamina engrosada +-e a lo largo de la normal de vertice (tres copias unidas): sin depender de la orientacion de las
    normales, las laminas mas finas que el voxel quedan con grosor y el remallado ya no abre huecos donde el signo del campo es ambiguo."""
    co, tri, _, _ = arrays(me)
    me.calc_normals() if hasattr(me, "calc_normals") else None
    nv = np.empty(len(me.vertices) * 3, np.float32); me.vertex_normals.foreach_get("vector", nv); nv = nv.reshape(-1, 3)
    n = len(co)
    co3 = np.concatenate([co, co + nv * e, co - nv * e])
    tri3 = np.concatenate([tri, tri + n, tri + 2 * n])
    return malla_de("engrosada", co3, tri3)


def voxelizar(alto, vox, nombre, grosor=0.0):
    """Copia de `alto` remallada por voxeles (cierra capas/solapes), solo islas exteriores, normales hacia fuera, triangulada.
    `grosor` > 0: antes se engrosa cada lamina +-grosor (ver `engrosar`)."""
    if grosor > 0:
        w = bpy.data.objects.new(nombre, engrosar(alto.data, grosor))
    else:
        w = alto.copy(); w.data = alto.data.copy(); w.name = nombre
    bpy.context.scene.collection.objects.link(w)
    while w.data.uv_layers:
        w.data.uv_layers.remove(w.data.uv_layers[0])
    w.data.materials.clear()
    activar(w)
    t0 = time.time()
    w.data.remesh_voxel_size = vox
    w.data.remesh_voxel_adaptivity = 0.0
    bpy.ops.object.voxel_remesh()
    t_vox = time.time() - t0
    log("%s: voxel %.2f mm -> %d verts %d caras (%.0fs)" % (nombre, vox * 1e3, len(w.data.vertices), len(w.data.polygons), t_vox))
    bm = bmesh.new(); bm.from_mesh(w.data)
    islas = componentes_cerradas(bm)
    amax = islas[0][1]
    borrar, quitadas = [], []
    for i, (v, ar, fs) in enumerate(islas):
        if i and (v <= 0 or ar < 0.02 * amax):     # cavidades (volumen negativo) y migas
            borrar += fs; quitadas.append((round(v, 6), round(ar, 4), len(fs)))
    if borrar:
        bmesh.ops.delete(bm, geom=borrar, context="FACES")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(w.data); bm.free(); w.data.update()
    w.data.polygons.foreach_set("use_smooth", np.ones(len(w.data.polygons), bool)); w.data.update()
    log("%s: islas %d, quitadas %d, euler %d" % (nombre, len(islas), len(quitadas), stats_topologia(w.data)["euler"]))
    return w, {"voxel_m": vox, "tiempo_voxel_s": round(t_vox, 1), "islas": len(islas), "islas_quitadas": len(quitadas)}


def decimar(w, ratio):
    activar(w)
    m = w.modifiers.new("dec", "DECIMATE"); m.ratio = ratio
    bpy.ops.object.modifier_apply(modifier="dec")


def quadriflow(w, caras):
    activar(w)
    t = time.time()
    r = bpy.ops.object.quadriflow_remesh(target_faces=caras, use_mesh_symmetry=False, use_preserve_sharp=False,
                                         use_preserve_boundary=False, smooth_normals=False, mode="FACES", seed=1)
    log("quadriflow", r, "%.0fs" % (time.time() - t), "->", len(w.data.vertices), "verts", len(w.data.polygons), "caras")
    if "FINISHED" not in r:
        raise SystemExit("QuadriFlow rechazo la malla")
    return time.time() - t


def cerrar(w):
    """Tras QuadriFlow: quita islas minusculas y tapa los huecos que deja (bordes abiertos); devuelve (islas quitadas, huecos tapados)."""
    bm = bmesh.new(); bm.from_mesh(w.data)
    islas = componentes_cerradas(bm)
    quit_ = [f for i, (_, _, fs) in enumerate(islas) if i and len(fs) < 0.02 * len(bm.faces) for f in fs]
    if quit_:
        bmesh.ops.delete(bm, geom=quit_, context="FACES")
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bordes = [e for e in bm.edges if e.is_boundary]
    n0 = len(bm.faces)
    if bordes:
        bmesh.ops.holes_fill(bm, edges=bordes, sides=16)
        resto = [e for e in bm.edges if e.is_boundary]
        if resto:
            try:
                bmesh.ops.triangle_fill(bm, edges=resto, use_beauty=True)
            except Exception as e:      # noqa: BLE001
                log("triangle_fill fallo:", e)
        resto = [e for e in bm.edges if e.is_boundary]
        if resto:
            try:
                bmesh.ops.edgenet_fill(bm, edges=resto, mat_nr=0, use_smooth=True, sides=32)
            except Exception as e:      # noqa: BLE001
                log("edgenet_fill fallo:", e)
    tapados = len(bm.faces) - n0
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(w.data); bm.free(); w.data.update()
    w.data.polygons.foreach_set("use_smooth", np.ones(len(w.data.polygons), bool)); w.data.update()
    return len(islas) - 1 if quit_ else 0, tapados


def cmd_remallar():
    """Remalla una parte. --voxel = tamano de la malla fina (referencia de forma); --voxel-proxy (opcional) = voxel mas grueso solo
    para que QuadriFlow acepte la malla (con capas muy finas el voxel fino deja micro-asas que Blender 5.3 rechaza); los vertices
    del resultado se proyectan despues sobre la malla fina (shrinkwrap) y se alisan --suavizar iteraciones."""
    trab, parte = arg("--trab"), arg("--parte")
    vox = float(arg("--voxel")); caras = int(arg("--caras"))
    proxy = float(arg("--voxel-proxy", 0)); dec = float(arg("--decimar", 0.3)); suav = int(arg("--suavizar", 2))
    bpy.ops.wm.open_mainfile(filepath=os.path.join(trab, "altas.blend"))
    alto = bpy.data.objects["alto_" + parte]
    grosor = float(arg("--grosor", 0))
    fino, inf = voxelizar(alto, vox, "fino", grosor)
    inf["grosor_m"] = grosor
    if proxy:
        w, inf_p = voxelizar(alto, proxy, "trabajo", grosor); inf["proxy"] = inf_p
    else:
        w = fino.copy(); w.data = fino.data.copy(); w.name = "trabajo"; bpy.context.scene.collection.objects.link(w)
    if dec < 1:
        decimar(w, dec)
        log("decimado ->", len(w.data.polygons), "caras")
    t_qf = quadriflow(w, caras)
    if proxy:
        activar(w)
        m = w.modifiers.new("sw", "SHRINKWRAP"); m.target = fino; m.wrap_method = "NEAREST_SURFACEPOINT"
        bpy.ops.object.modifier_apply(modifier="sw")
        if suav:
            m = w.modifiers.new("sm", "SMOOTH"); m.factor = 0.5; m.iterations = suav
            bpy.ops.object.modifier_apply(modifier="sm")
            m = w.modifiers.new("sw", "SHRINKWRAP"); m.target = fino; m.wrap_method = "NEAREST_SURFACEPOINT"
            bpy.ops.object.modifier_apply(modifier="sw")
        log("proyectado sobre la malla fina")
    quitadas, tapados = cerrar(w)
    log("cerrar: islas quitadas", quitadas, "caras de tapa", tapados)
    w.name = "juego_" + parte; w.data.name = "juego_" + parte
    topo = stats_topologia(w.data)
    log("topologia:", topo)
    bm = bmesh.new(); bm.from_mesh(w.data)
    ar = np.array([f.calc_area() for f in bm.faces]); bm.free()
    topo["area_media_cm2"] = round(float(ar.mean()) * 1e4, 3)
    topo["caras_degeneradas"] = int((ar < 0.05 * ar.mean()).sum())     # area < 5 % de la media: pliegues/colapsos por la proyeccion
    log("caras degeneradas (<5%% del area media): %d de %d" % (topo["caras_degeneradas"], len(ar)))
    for o in list(bpy.data.objects):
        if o is not w:
            bpy.data.objects.remove(o)
    out = {"parte": parte, "voxel_m": vox, "voxel_proxy_m": proxy or None, "objetivo_caras": caras, "decimar": dec,
           "tiempo_quadriflow_s": round(t_qf, 1), "fino": inf, "vertices": len(w.data.vertices), "caras": len(w.data.polygons),
           "tris": int(sum(len(p.vertices) - 2 for p in w.data.polygons)), "caras_tapa": tapados, "topologia": topo}
    json.dump(out, open(os.path.join(trab, "retopo_%s.json" % parte), "w"), indent=1)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(trab, "retopo_%s.blend" % parte), compress=True)


{"separar": cmd_separar, "medir": cmd_medir, "remallar": cmd_remallar}.get(CMD, lambda: log("comando desconocido", CMD))()
