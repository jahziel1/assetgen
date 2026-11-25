"""envolver_original.py -- remaster SOBRE la malla original: se subdivide la malla del juego (topologia, pesos y piezas intactos) y se
envuelve sobre un modelo nuevo de alta (proyeccion + suavizado); despues UV en un atlas y horneado de color/normal/AO desde el nuevo.

    blender -b -P envolver_original.py -- preparar --datos analisis_original --salida DIR [--objetivo 30000] [--finura "clave=0.4,..."]
    blender -b -P envolver_original.py -- envolver --base DIR/base.npz --alto nuevo.glb --salida DIR [--dmax 0.03]
        [--especiales "clave=0.10,clave2=0.06"] [--conservar clave1,clave2] [--suavizado 6] [--objeto principe_alto]
    blender -b -P envolver_original.py -- uv       --salida DIR [--margen 0.004] [--angulo 66] [--res 4096]
    blender -b -P envolver_original.py -- hornear  --salida DIR --alto nuevo.glb [--res 4096] [--extrusion 0.02] [--rayo 0.05] [--muestras 16]
    blender -b -P envolver_original.py -- exportar --salida DIR [--nombre principe_remaster]

`preparar`: lee original_bind.glb (grupos `hNN`) y reconstruye la pieza de cada cara repitiendo el orden de `exportar_bind.py`
(entrada/ y entrada_parches/); subdivide por longitud de arista (sin suavizado; pesos interpolados <= 3, pieza heredada).
`envolver`: cada posicion (las costuras entre piezas comparten posicion y se proyectan una sola vez) se lleva al punto del nuevo
a lo largo de la normal (+/-) o al mas cercano, con distancia maxima por pieza; se suavizan los desplazamientos; informa distancia
superficie a superficie (nueva->envuelta y envuelta->nueva) por pieza. `--conservar`: piezas que no se proyectan (se quedan con la forma
del original subdividida, ancladas a sus costuras).
Escribe en DIR: base.npz, envuelto.npz, envoltura.json, envuelto.blend, uv.blend, juego_*.png, principe_remaster.blend/.glb, remaster.json."""
import json
import math
import os
import sys
import time

import bpy
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "..", "cli"))
from assetgen_cli import envoltura_geom as EG  # noqa: E402

a = sys.argv[sys.argv.index("--") + 1:]
CMD = a[0] if a else ""
T0 = time.time()


def arg(n, d=None):
    return a[a.index(n) + 1] if n in a else d


def log(*m):
    print("[%6.1fs]" % (time.time() - T0), *m, flush=True)


def activar(ob, *otros):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in otros:
        o.select_set(True)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)


def arrays_malla(me):
    n = len(me.vertices)
    co = np.empty(n * 3, np.float64); me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


def caras_tri(me):
    me.calc_loop_triangles()
    t = np.empty(len(me.loop_triangles) * 3, np.int32); me.loop_triangles.foreach_get("vertices", t)
    return t.reshape(-1, 3)


# ----------------------------------------------------------------------------------------------------------------- preparar
def piezas_por_cara(datos, caras_glb):
    """Pieza (nombre de dibujo) de cada triangulo del glb: repite el orden de exportar_bind.py (vb unicos, piezas en orden alfabetico
    de entrada/ y despues entrada_parches/). Comprueba que coincide con las caras del glb."""
    import glob
    import hashlib
    dt = np.dtype([("p", "<f4", 3), ("n", "<f4", 3), ("i", "<i2", 4), ("w", "<f4", 3), ("uv", "<f4", 2)])
    vbs = {}
    for carpeta in ("entrada", "entrada_parches"):
        for vbf in sorted(glob.glob(os.path.join(datos, carpeta, "*.vb"))):
            nombre = os.path.basename(vbf)[:-3]
            raw = open(vbf, "rb").read()
            V = np.frombuffer(raw[:len(raw) // 52 * 52], dtype=dt)
            I = np.frombuffer(open(vbf[:-3] + ".ib", "rb").read(), dtype="<u2").reshape(-1, 3).astype(np.int64)
            e = vbs.setdefault(hashlib.md5(raw).hexdigest(), {"n": len(V), "tris": [], "usado": np.zeros(len(V), bool), "piezas": []})
            e["piezas"].append(nombre); e["tris"].append(I); e["usado"][np.unique(I)] = True
    off, tris, pieza = 0, [], []
    for e in vbs.values():
        mapa = -np.ones(e["n"], np.int64); idx = np.where(e["usado"])[0]; mapa[idx] = np.arange(len(idx)) + off; off += len(idx)
        for nm, t in zip(e["piezas"], e["tris"]):
            tris.extend(map(tuple, mapa[t].tolist())); pieza.extend([nm] * len(t))
    if len(tris) != len(caras_glb) or any(sorted(x) != sorted(y) for x, y in zip(tris, caras_glb)):
        raise SystemExit("las caras del glb no coinciden con las piezas reconstruidas (%d vs %d)" % (len(tris), len(caras_glb)))
    return pieza


def cmd_preparar():
    datos, salida = arg("--datos"), arg("--salida"); objetivo = int(arg("--objetivo", 30000))
    os.makedirs(salida, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=os.path.join(datos, "original_bind.glb"))
    ob = next(o for o in bpy.context.scene.objects if o.type == "MESH" and o.vertex_groups)
    me = ob.data
    P = arrays_malla(me)
    caras = [tuple(p.vertices) for p in me.polygons]
    assert all(len(c) == 3 for c in caras)
    nombres_g = {g.index: int(g.name[1:]) for g in ob.vertex_groups}
    pesos = []
    for v in me.vertices:
        w = {nombres_g[g.group]: g.weight for g in v.groups if g.weight > 1e-6}
        pesos.append(w)
    pieza_n = piezas_por_cara(datos, caras)
    nombres = sorted(set(pieza_n))
    pieza = [nombres.index(x) for x in pieza_n]
    fin = {}
    for t in (arg("--finura", "") or "").split(","):
        if "=" in t:
            k, v = t.split("="); fin.update({i: float(v) for i, nm in enumerate(nombres) if k in nm})
    grupo0 = EG.agrupar_por_posicion(P.tolist(), 1e-4)
    factor = EG.factor_por_grupo(grupo0, caras, pieza, fin)
    (P2, C2, W2, Z2, padres), lmax = EG.subdividir_hasta(P.tolist(), caras, pesos, pieza, objetivo, factor=factor)
    log("subdivision: %d -> %d vertices, %d -> %d triangulos, lmax %.4f m" % (len(P), len(P2), len(caras), len(C2), lmax))
    n = len(P2)
    hueso = -np.ones((n, 3), np.int32); peso = np.zeros((n, 3), np.float32)
    for i, w in enumerate(W2):
        for k, (h, x) in enumerate(sorted(w.items(), key=lambda t: -t[1])[:3]):
            hueso[i, k] = h; peso[i, k] = x
    pa = np.array([(-1, -1) if p is None else p for p in padres], np.int32)
    np.savez(os.path.join(salida, "base.npz"), P=np.array(P2), caras=np.array(C2, np.int32), pieza=np.array(Z2, np.int32),
             nombres=np.array(nombres), hueso=hueso, peso=peso, padres=pa, n_orig=len(P), caras_orig=np.array(caras, np.int32),
             pieza_orig=np.array(pieza, np.int32), P_orig=P)
    por_pieza = {nombres[i]: {"triangulos_original": int(sum(1 for z in pieza if z == i)), "triangulos": int(sum(1 for z in Z2 if z == i))}
                 for i in range(len(nombres))}
    json.dump({"vertices": n, "triangulos": len(C2), "lmax_m": lmax, "por_pieza": por_pieza,
               "original": {"vertices": len(P), "triangulos": len(caras)}},
              open(os.path.join(salida, "subdivision.json"), "w"), indent=1)
    log(json.dumps(por_pieza))


# ----------------------------------------------------------------------------------------------------------------- envolver
def cargar_alto(ruta, objeto):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if ruta.endswith(".blend"):
        bpy.ops.wm.open_mainfile(filepath=ruta)
    else:
        bpy.ops.import_scene.gltf(filepath=ruta)
    ob = bpy.data.objects[objeto] if objeto else max((o for o in bpy.context.scene.objects if o.type == "MESH"),
                                                      key=lambda o: len(o.data.polygons))
    ob.data.materials  # noqa: B018
    return ob


def bvh_de(ob):
    from mathutils.bvhtree import BVHTree
    dg = bpy.context.evaluated_depsgraph_get()
    return BVHTree.FromObject(ob, dg)


def bvh_de_arrays(P, caras):
    from mathutils.bvhtree import BVHTree
    return BVHTree.FromPolygons([tuple(p) for p in P.tolist()], [tuple(c) for c in caras.tolist()])


def normales_vertice(P, caras):
    fn = np.cross(P[caras[:, 1]] - P[caras[:, 0]], P[caras[:, 2]] - P[caras[:, 0]])
    N = np.zeros_like(P)
    for k in range(3):
        np.add.at(N, caras[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


def distancias_a(bvh, puntos):
    d = np.empty(len(puntos))
    for i, p in enumerate(puntos.tolist()):
        r = bvh.find_nearest(p)
        d[i] = r[3] if r[0] is not None else 1e9
    return d


def muestras_malla(P, caras):
    """Vertices + centroides de cara, con la cara a la que pertenece (los vertices -> la primera cara que los usa)."""
    cen = P[caras].mean(axis=1)
    return cen


def cmd_envolver():
    from mathutils import Vector  # noqa: F401
    base = np.load(os.path.join(arg("--base")), allow_pickle=False)
    salida = arg("--salida"); dmax_base = float(arg("--dmax", 0.03)); iters = int(arg("--suavizado", 6))
    P, caras, pieza, nombres = base["P"], base["caras"], base["pieza"], [str(x) for x in base["nombres"]]
    especiales = {}
    for t in (arg("--especiales", "") or "").split(","):
        if "=" in t:
            k, v = t.split("="); especiales[k] = float(v)
    conservar = [t for t in (arg("--conservar", "") or "").split(",") if t]
    dmax_p = EG.dmax_por_pieza(nombres, dmax_base, especiales)
    ob = cargar_alto(arg("--alto"), arg("--objeto"))
    log("alto: %s %d tris" % (ob.name, len(ob.data.polygons)))
    bvh = bvh_de(ob)
    n = len(P)
    grupo = np.array(EG.agrupar_por_posicion(P.tolist(), 1e-4)); ng = int(grupo.max()) + 1
    # posicion, normal y dmax por grupo; piezas por grupo
    Nv = normales_vertice(P, caras)
    pos_g = np.zeros((ng, 3)); nor_g = np.zeros((ng, 3)); dm_g = np.zeros(ng); cnt = np.zeros(ng)
    piezas_g = [set() for _ in range(ng)]
    pieza_v = -np.ones(n, np.int32)
    for k in range(3):
        pieza_v[caras[:, k]] = pieza
    for v in range(n):
        g = grupo[v]
        if cnt[g] == 0:
            pos_g[g] = P[v]; nor_g[g] = Nv[v]
        else:
            nor_g[g] += Nv[v] if np.dot(Nv[v], nor_g[g]) >= 0 else -Nv[v]
        cnt[g] += 1
    for k in range(3):
        for c, z in zip(caras[:, k], pieza):
            piezas_g[grupo[c]].add(int(z))
    for g in range(ng):
        dm_g[g] = max(dmax_p[nombres[z]] for z in piezas_g[g]) if piezas_g[g] else dmax_base
    nor_g /= np.maximum(np.linalg.norm(nor_g, axis=1, keepdims=True), 1e-12)
    # proyeccion por grupo
    delta = np.zeros((ng, 3)); conf = np.ones(ng); modo = np.zeros(ng, np.int8)    # 0 = sin impacto, 1 = normal, 2 = mas cercano
    d_antes = np.zeros(ng)
    t0 = time.time()
    for g in range(ng):
        p = Vector(pos_g[g].tolist()); nrm = Vector(nor_g[g].tolist()); dm = dm_g[g]
        cerca = bvh.find_nearest(p)
        dn = cerca[3] if cerca[0] is not None else 1e9
        d_antes[g] = dn
        mejor = None
        for s in (1.0, -1.0):
            h = bvh.ray_cast(p, nrm * s, dm)
            if h[0] is not None and abs(h[1].dot(nrm)) >= 0.35:
                if mejor is None or h[3] < mejor[1]:
                    mejor = (h[0] - p, h[3])
        if mejor is not None and mejor[1] <= max(1.6 * dn, dn + 0.005) and mejor[1] <= dm:
            delta[g] = mejor[0]; modo[g] = 1
        elif cerca[0] is not None:
            d = cerca[0] - p
            delta[g] = EG.limitar_longitud((d.x, d.y, d.z), dm); modo[g] = 2
            conf[g] = 1.0 if dn <= dm else 0.3
        else:
            conf[g] = 0.1
    log("proyeccion %.1fs: normal %d, cercano %d, fuera de distancia %d de %d grupos" % (
        time.time() - t0, (modo == 1).sum(), (modo == 2).sum(), (d_antes > dm_g).sum(), ng))
    # piezas conservadas: sin desplazamiento propio; las costuras con otras piezas mantienen la proyeccion
    fijos = set()
    if conservar:
        idx_cons = {i for i, nm in enumerate(nombres) if any(c in nm for c in conservar)}
        for g in range(ng):
            if piezas_g[g] and piezas_g[g] <= idx_cons:
                delta[g] = 0.0; conf[g] = 1.0
            elif piezas_g[g] & idx_cons:
                fijos.add(g)
    vec = EG.vecinos_de_grupos(caras.tolist(), grupo.tolist(), ng)
    D = EG.suavizar_desplazamientos([tuple(d) for d in delta], vec, iteraciones=iters, lam=0.5, fijos=fijos, confianza=conf.tolist())
    D = np.array([EG.limitar_longitud(d, dm_g[g] * 1.05) for g, d in enumerate(D)])
    # ajuste fino: varias vueltas de «acercar al punto mas cercano del nuevo» (limitado y a medias) + un suavizado; baja el error residual
    # que deja el suavizado. Las piezas conservadas no se ajustan.
    n_aj = int(arg("--ajuste", 3)); paso_max = float(arg("--ajuste-max", 0.015))
    libres = np.array([not (piezas_g[g] and conservar and piezas_g[g] <= idx_cons) for g in range(ng)]) if conservar else np.ones(ng, bool)
    for _ in range(n_aj):
        pos = pos_g + D
        E = np.zeros((ng, 3))
        for g in range(ng):
            if not libres[g] or g in fijos:
                continue
            r = bvh.find_nearest(Vector(pos[g].tolist()))
            if r[0] is not None:
                E[g] = EG.limitar_longitud((r[0].x - pos[g, 0], r[0].y - pos[g, 1], r[0].z - pos[g, 2]), min(paso_max, dm_g[g]))
        D = D + 0.5 * E
        D = np.array(EG.suavizar_desplazamientos([tuple(d) for d in D], vec, iteraciones=1, lam=0.3, fijos=fijos))
    P2 = P + D[grupo]
    # ---- metricas superficie a superficie
    P_o = P
    bvh_n = bvh_de_arrays(P2, caras)
    def por_pieza(Pm, bvh_a):
        pts, etq = [], []
        cen = Pm[caras].mean(axis=1)
        for k, z in enumerate(pieza.tolist()):
            pass
        # puntos: vertices (por pieza de su cara) y centroides
        vv = np.concatenate([caras[:, 0], caras[:, 1], caras[:, 2]]); zz = np.concatenate([pieza, pieza, pieza])
        pts = np.concatenate([Pm[vv], cen]); etq = np.concatenate([zz, pieza])
        d = distancias_a(bvh_a, pts)
        out = {nm: EG.resumen_distancias(d[etq == i].tolist()) for i, nm in enumerate(nombres)}
        out["_total"] = EG.resumen_distancias(d.tolist())
        return out
    antes = por_pieza(P_o, bvh)
    despues = por_pieza(P2, bvh)
    log("distancia nueva<-envuelta (m) p50/p95 total: antes %s / despues %s" % (
        (antes["_total"]["p50"], antes["_total"]["p95"]), (despues["_total"]["p50"], despues["_total"]["p95"])))
    # inversa: puntos del nuevo (centroides de una muestra de triangulos) -> envuelta; se atribuye a la pieza del triangulo mas cercano
    me_a = ob.data
    me_a.calc_loop_triangles()
    nt = len(me_a.loop_triangles)
    rng = np.random.default_rng(7)
    sel = rng.choice(nt, size=min(60000, nt), replace=False)
    co = arrays_malla(me_a); tv = np.empty(nt * 3, np.int32); me_a.loop_triangles.foreach_get("vertices", tv); tv = tv.reshape(-1, 3)
    mw = np.array(ob.matrix_world)
    pts = (co[tv[sel]].mean(axis=1) @ mw[:3, :3].T + mw[:3, 3])
    d_inv = np.empty(len(pts)); z_inv = np.empty(len(pts), np.int32)
    for i, p in enumerate(pts.tolist()):
        r = bvh_n.find_nearest(p)
        d_inv[i] = r[3]; z_inv[i] = pieza[r[2]]
    inversa = {nm: EG.resumen_distancias(d_inv[z_inv == i].tolist()) for i, nm in enumerate(nombres)}
    inversa["_total"] = EG.resumen_distancias(d_inv.tolist())
    inversa["_fraccion_a_mas_de_3cm"] = round(float((d_inv > 0.03).mean()), 4)
    inversa["_fraccion_a_mas_de_6cm"] = round(float((d_inv > 0.06).mean()), 4)
    # caras invertidas / calidad
    mal = EG.caras_invertidas(P_o.tolist(), P2.tolist(), caras.tolist())
    mal_por = {nm: int(sum(1 for k in mal if pieza[k] == i)) for i, nm in enumerate(nombres)}
    def lado(Pm):
        e = np.concatenate([np.linalg.norm(Pm[caras[:, k]] - Pm[caras[:, (k + 1) % 3]], axis=1) for k in range(3)])
        return e
    e0, e1 = lado(P_o), lado(P2)
    res = {"dmax_por_pieza_m": dmax_p, "conservadas": conservar, "suavizado_iter": iters,
           "grupos_posicion": ng, "vertices": n, "triangulos": int(len(caras)),
           "proyeccion": {"normal": int((modo == 1).sum()), "mas_cercano": int((modo == 2).sum()), "sin_impacto": int((modo == 0).sum())},
           "nueva_a_envuelta_antes": antes, "nueva_a_envuelta": despues, "inversa_nuevo_a_envuelta": inversa,
           "caras_invertidas": len(mal), "caras_invertidas_por_pieza": mal_por,
           "aristas_m": {"original_p50": round(float(np.percentile(e0, 50)), 5), "envuelta_p50": round(float(np.percentile(e1, 50)), 5),
                         "envuelta_p99": round(float(np.percentile(e1, 99)), 5), "envuelta_max": round(float(e1.max()), 5)},
           "desplazamiento_m": {"p50": round(float(np.percentile(np.linalg.norm(D, axis=1), 50)), 5),
                                "p95": round(float(np.percentile(np.linalg.norm(D, axis=1), 95)), 5),
                                "max": round(float(np.linalg.norm(D, axis=1).max()), 5)}}
    # orientacion: flujo de las normales respecto al centro de cada pieza (positivo = hacia fuera); se voltea la pieza con flujo negativo
    pp = P2[caras]; fn2 = np.cross(pp[:, 1] - pp[:, 0], pp[:, 2] - pp[:, 0]); cen2 = pp.mean(axis=1)
    flujo, frac_fuera = np.zeros(len(caras)), {}
    voltear = np.zeros(len(nombres), bool)
    for i, nm in enumerate(nombres):
        m = pieza == i
        c0 = P2[np.unique(caras[m])].mean(axis=0)
        fl = ((cen2[m] - c0) * fn2[m]).sum(axis=1)
        frac_fuera[nm] = round(float(fl.sum() / max(np.abs(fl).sum(), 1e-12)), 3)
        voltear[i] = frac_fuera[nm] < 0
    res["orientacion"] = {"flujo_ponderado_por_pieza_(+1 = hacia fuera)": frac_fuera, "piezas_volteadas": [nm for nm, v in zip(nombres, voltear) if v]}
    log("orientacion (flujo, +1 = hacia fuera)", json.dumps(frac_fuera))
    json.dump(res, open(os.path.join(salida, "envoltura.json"), "w"), indent=1)
    np.savez(os.path.join(salida, "envuelto.npz"), P=P2, D=D[grupo], grupo=grupo, voltear=voltear)
    for nm in nombres:
        log("%-16s dmax %.3f antes p50/p95 %.4f/%.4f -> %.4f/%.4f | inversa p50/p95 %s/%s | invertidas %d" % (
            nm, dmax_p[nm], antes[nm].get("p50", 0), antes[nm].get("p95", 0), despues[nm].get("p50", 0), despues[nm].get("p95", 0),
            inversa[nm].get("p50"), inversa[nm].get("p95"), mal_por[nm]))
    log("inversa total", json.dumps(inversa["_total"]), "a>3cm", inversa["_fraccion_a_mas_de_3cm"], "a>6cm", inversa["_fraccion_a_mas_de_6cm"])


# ----------------------------------------------------------------------------------------------------------------- malla de juego
def construir(salida):
    """Malla envuelta como objeto Blender: caras volteadas hacia fuera, grupos hNN, atributo `pieza` por cara, normales suaves
    promediadas en las costuras entre piezas (mismo grupo de posicion y normales parecidas)."""
    base = np.load(os.path.join(salida, "base.npz")); env = np.load(os.path.join(salida, "envuelto.npz"))
    P, caras, pieza = env["P"], base["caras"].copy(), base["pieza"]
    nombres = [str(x) for x in base["nombres"]]
    vol = env["voltear"][pieza]
    caras[vol] = caras[vol][:, [0, 2, 1]]
    me = bpy.data.meshes.new("principe")
    me.from_pydata(P.tolist(), [], caras.tolist()); me.update()
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))      # sin esto las normales personalizadas salen planas
    me.attributes.new("pieza", "INT", "FACE").data.foreach_set("value", pieza.astype(np.int32))
    ob = bpy.data.objects.new("principe", me); bpy.context.scene.collection.objects.link(ob)
    hueso, peso = base["hueso"], base["peso"]
    grupos = {}
    for v in range(len(P)):
        for k in range(3):
            h = int(hueso[v, k])
            if h >= 0 and peso[v, k] > 0:
                vg = grupos.get(h) or grupos.setdefault(h, ob.vertex_groups.new(name="h%02d" % h))
                vg.add([v], float(peso[v, k]), "REPLACE")
    # normales: por vertice (de sus caras) y promedio con los vertices del mismo grupo de posicion que miran parecido
    N = normales_vertice(P, caras)
    g = env["grupo"]
    orden = np.argsort(g, kind="stable"); gs = g[orden]
    cortes = np.flatnonzero(np.diff(gs)) + 1
    N2 = N.copy()
    for idx in np.split(orden, cortes):
        if len(idx) > 1:
            for v in idx:
                m = idx[(N[idx] @ N[v]) > 0.5]
                s_ = N[m].sum(axis=0); nn = np.linalg.norm(s_)
                if nn > 1e-9:
                    N2[v] = s_ / nn
    me.normals_split_custom_set_from_vertices(N2.tolist())
    return ob, nombres


def estadisticas_uv(ob, res):
    me = ob.data
    co = arrays_malla(me)
    me.calc_loop_triangles()
    tv = np.empty(len(me.loop_triangles) * 3, np.int32); me.loop_triangles.foreach_get("vertices", tv); tv = tv.reshape(-1, 3)
    tl = np.empty(len(me.loop_triangles) * 3, np.int32); me.loop_triangles.foreach_get("loops", tl); tl = tl.reshape(-1, 3)
    uv = np.empty(len(me.loops) * 2, np.float64); me.uv_layers[0].data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
    p = co[tv]; a3 = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    u = uv[tl]; a2 = 0.5 * np.abs((u[:, 1, 0] - u[:, 0, 0]) * (u[:, 2, 1] - u[:, 0, 1]) - (u[:, 2, 0] - u[:, 0, 0]) * (u[:, 1, 1] - u[:, 0, 1]))
    ok = (a3 > 1e-10) & (a2 > 1e-14)
    # estiramiento: razon de densidad local (px/m) frente a la mediana
    dens = np.sqrt(a2[ok] / a3[ok]); med = np.median(dens)
    # razon de aspecto de cada triangulo UV vs 3D por las longitudes de arista
    l3 = np.stack([np.linalg.norm(p[:, k] - p[:, (k + 1) % 3], axis=1) for k in range(3)], axis=1)
    l2 = np.stack([np.linalg.norm(u[:, k] - u[:, (k + 1) % 3], axis=1) for k in range(3)], axis=1)
    r = (l2[ok] / np.maximum(l3[ok], 1e-9)) / np.maximum(dens[:, None], 1e-12)
    aniso = r.max(axis=1) / np.maximum(r.min(axis=1), 1e-9)
    return {"area_3d_m2": round(float(a3.sum()), 4), "area_uv": round(float(a2.sum()), 4), "px_por_m": round(float(np.sqrt(a2.sum() / a3.sum()) * res), 0),
            "densidad_relativa_p5_p95": [round(float(np.percentile(dens, 5) / med), 2), round(float(np.percentile(dens, 95) / med), 2)],
            "anisotropia_mediana": round(float(np.median(aniso)), 2), "anisotropia_p95": round(float(np.percentile(aniso, 95)), 2),
            "u": [round(float(uv[:, 0].min()), 3), round(float(uv[:, 0].max()), 3)], "v": [round(float(uv[:, 1].min()), 3), round(float(uv[:, 1].max()), 3)]}


def cmd_uv():
    salida = arg("--salida"); marg = float(arg("--margen", 0.004)); ang = float(arg("--angulo", 66)); res = int(arg("--res", 4096))
    dens = {}
    for t in (arg("--densidad", "000056=1.6") or "").split(","):
        if "=" in t:
            k, v = t.split("="); dens[k] = float(v)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ob, nombres = construir(salida)
    ob.data.uv_layers.new(name="UVMap")
    activar(ob)
    ts = bpy.context.scene.tool_settings
    ts.use_uv_select_sync = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    t0 = time.time()
    bpy.ops.uv.smart_project(angle_limit=math.radians(ang), margin_method="FRACTION", island_margin=0.0, area_weight=0.0,
                             correct_aspect=True, scale_to_bounds=False)
    bpy.ops.uv.average_islands_scale()
    bpy.ops.object.mode_set(mode="OBJECT")
    me = ob.data
    pieza = np.empty(len(me.polygons), np.int32); me.attributes["pieza"].data.foreach_get("value", pieza)
    uv = np.empty(len(me.loops) * 2, np.float32); me.uv_layers[0].data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
    lp = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", lp)
    fl = np.repeat(np.arange(len(me.polygons)), 3)      # triangulos: 3 esquinas por cara
    for clave, k in dens.items():
        if k == 1.0:
            continue
        for i, nm in enumerate(nombres):
            if clave in nm:
                m = pieza[fl] == i
                c = uv[m].mean(axis=0); uv[m] = (uv[m] - c) * k + c
    me.uv_layers[0].data.foreach_set("uv", uv.ravel())
    activar(ob)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.pack_islands(udim_source="CLOSEST_UDIM", rotate=True, rotate_method="ANY", scale=True, merge_overlap=False,
                            margin_method="FRACTION", margin=marg, pin=False, shape_method="CONCAVE")
    bpy.ops.object.mode_set(mode="OBJECT")
    st = estadisticas_uv(ob, res)
    # numero de islas (componentes de caras que comparten esquinas UV en sus aristas)
    st["densidad_relativa_pedida"] = dens; st["margen_uv"] = marg; st["margen_px"] = round(marg * res, 1); st["angulo"] = ang; st["res"] = res
    st["tiempo_s"] = round(time.time() - t0, 1)
    log("UV:", json.dumps(st))
    json.dump(st, open(os.path.join(salida, "uv_stats.json"), "w"), indent=1)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(salida, "uv.blend"), compress=True)


# ----------------------------------------------------------------------------------------------------------------- hornear
def gpu():
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    try:
        pr = bpy.context.preferences.addons["cycles"].preferences
        pr.compute_device_type = "METAL"; pr.get_devices()
        usados = []
        for d in pr.devices:
            d.use = d.type == "METAL"
            if d.use:
                usados.append(d.name)
        if not usados:
            raise RuntimeError("sin dispositivos METAL")
        sc.cycles.device = "GPU"
        return "GPU Metal: " + ", ".join(usados)
    except Exception as e:      # noqa: BLE001
        sc.cycles.device = "CPU"
        return "CPU (%s)" % e


def nueva_imagen(nombre, res, color, srgb):
    im = bpy.data.images.new(nombre, res, res, alpha=False)
    im.colorspace_settings.name = "sRGB" if srgb else "Non-Color"
    px = np.empty(res * res * 4, np.float32); px.reshape(-1, 4)[:] = color
    im.pixels.foreach_set(px)
    return im


def pixeles(im):
    res = im.size[0]
    px = np.empty(res * res * 4, np.float32); im.pixels.foreach_get(px)
    return px.reshape(res, res, 4)


def hornear_una(bajo, alto, etq, res, extr, rayo, muestras):
    sc = bpy.context.scene
    sc.cycles.samples = muestras
    sc.render.bake.use_clear = True
    fondo = {"COLOR": (0, 0, 0, 1), "NORMAL": (0.5, 0.5, 1, 1), "AO": (1, 1, 1, 1), "MASK": (0, 0, 0, 1)}[etq]
    im = nueva_imagen("h_" + etq, res, fondo, etq == "COLOR")
    m = bpy.data.materials.new("hornear_" + etq); m.use_nodes = True
    nt = m.node_tree
    nodo = nt.nodes.new("ShaderNodeTexImage"); nodo.image = im; nt.nodes.active = nodo
    bajo.data.materials.clear(); bajo.data.materials.append(m)
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    if etq != "MASK":
        alto.select_set(True)
    bajo.select_set(True); bpy.context.view_layer.objects.active = bajo
    t0 = time.time()
    if etq == "MASK":
        em = nt.nodes.new("ShaderNodeEmission"); em.inputs["Color"].default_value = (1, 1, 1, 1)
        sal = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
        nt.links.new(em.outputs[0], sal.inputs["Surface"])
        bpy.ops.object.bake(type="EMIT", margin=0, use_selected_to_active=False, use_clear=True)
    else:
        tipo = {"COLOR": "DIFFUSE", "NORMAL": "NORMAL", "AO": "AO"}[etq]
        kws = dict(type=tipo, margin=0, use_selected_to_active=True, cage_extrusion=extr, max_ray_distance=rayo, use_clear=True)
        if tipo == "DIFFUSE":
            kws["pass_filter"] = {"COLOR"}
        if tipo == "NORMAL":
            kws["normal_space"] = "TANGENT"
        bpy.ops.object.bake(**kws)
    arr = pixeles(im).copy()
    bpy.data.images.remove(im)
    return arr, time.time() - t0


def dilatar(img, cubierto, pasos):
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
    bpy.context.scene.render.image_settings.color_depth = "8"; bpy.context.scene.render.image_settings.color_mode = "RGB"
    im.save()
    bpy.data.images.remove(im)


def cmd_hornear():
    salida = arg("--salida"); res = int(arg("--res", 4096)); extr = float(arg("--extrusion", 0.02)); rayo = float(arg("--rayo", 0.05))
    muestras = int(arg("--muestras", 16)); margen_px = int(arg("--margen-px", 16)); min_ao = float(arg("--min-ao", 0))
    bpy.ops.wm.open_mainfile(filepath=os.path.join(salida, "uv.blend"))
    bajo = bpy.data.objects["principe"]
    n_antes = set(bpy.data.objects.keys())
    alto = cargar_alto_en_escena(arg("--alto"), arg("--objeto"))
    disp = gpu(); log("dispositivo", disp)
    mask, dt_m = hornear_una(bajo, alto, "MASK", res, extr, rayo, 1)
    mascara = mask[..., 0] > 0.5
    log("mascara px", int(mascara.sum()), "%.1fs" % dt_m)
    datos, tiempos = {}, {"mascara": round(dt_m, 1)}
    for etq in ("COLOR", "NORMAL", "AO"):
        datos[etq], dt = hornear_una(bajo, alto, etq, res, extr, rayo, muestras if etq != "AO" else max(32, muestras))
        tiempos[etq.lower()] = round(dt, 1); log("hornead", etq, "%.1fs" % dt)
    valido = mascara & (datos["COLOR"][..., :3].max(axis=2) >= 0.004)
    if min_ao > 0:
        valido &= datos["AO"][..., 0] >= min_ao
    sin_impacto = int((mascara & ~valido).sum())
    # segunda pasada con jaula y rayo mayores solo para los texeles que no dieron en el nuevo (zonas donde la envoltura queda lejos)
    extr2, rayo2 = float(arg("--extrusion2", 0.06)), float(arg("--rayo2", 0.16)); sin_impacto2 = sin_impacto
    if sin_impacto > 0 and extr2 > 0:
        for etq in ("COLOR", "NORMAL", "AO"):
            d2, dt = hornear_una(bajo, alto, etq, res, extr2, rayo2, muestras if etq != "AO" else max(32, muestras))
            tiempos[etq.lower() + "_2"] = round(dt, 1)
            if etq == "COLOR":
                v2 = mascara & ~valido & (d2[..., :3].max(axis=2) >= 0.004)
            datos[etq][v2] = d2[v2]
        valido = valido | v2
        sin_impacto2 = int((mascara & ~valido).sum())
        log("segunda pasada: recuperados %d de %d texeles" % (int(v2.sum()), sin_impacto))
    fondos = {"COLOR": (0, 0, 0, 1), "NORMAL": (0.5, 0.5, 1, 1), "AO": (1, 1, 1, 1)}
    final = {}
    for etq in ("COLOR", "NORMAL", "AO"):
        img = np.zeros((res, res, 4), np.float32); img[:] = fondos[etq]
        img[valido] = datos[etq][valido]
        img, _ = dilatar(img, valido, margen_px + 8)
        if etq == "NORMAL":
            nn = img[..., :3] * 2 - 1; nn /= np.maximum(np.linalg.norm(nn, axis=2, keepdims=True), 1e-6); img[..., :3] = nn * 0.5 + 0.5
        final[etq] = img
    for etq, (nom, srgb) in {"COLOR": ("juego_color.png", True), "NORMAL": ("juego_normal.png", False), "AO": ("juego_ao.png", False)}.items():
        guardar_png(final[etq], os.path.join(salida, nom), srgb, res)
    # ORM del visor/glTF: R = oclusion, G = rugosidad, B = metal
    orm = np.zeros((res, res, 4), np.float32); orm[..., 0] = final["AO"][..., 0]; orm[..., 1] = 0.85; orm[..., 2] = 0.0
    guardar_png(orm, os.path.join(salida, "juego_orm.png"), False, res)
    info = {"res": res, "extrusion_m": extr, "rayo_m": rayo, "muestras": muestras, "margen_px": margen_px, "tiempos_s": tiempos, "dispositivo": disp,
            "cobertura_atlas": round(float(mascara.mean()), 3), "px_cubiertos": int(mascara.sum()), "px_sin_impacto_primera_pasada": sin_impacto, "px_sin_impacto_rellenados": sin_impacto2, "extrusion2_m": extr2, "rayo2_m": rayo2,
            "fraccion_sin_impacto": round(sin_impacto2 / max(int(mascara.sum()), 1), 4)}
    json.dump(info, open(os.path.join(salida, "hornear.json"), "w"), indent=1)
    log("horneado listo", json.dumps(info))


def cargar_alto_en_escena(ruta, objeto):
    """Importa el alto (glb o blend) en la escena actual y lo devuelve; lo demas que traiga el glb (huesos, icosferas) se descarta."""
    antes = set(bpy.data.objects.keys())
    if ruta.endswith(".blend"):
        with bpy.data.libraries.load(ruta, link=False) as (src, dst):
            dst.objects = [n for n in src.objects if (objeto is None or n == objeto)]
        for o in dst.objects:
            bpy.context.scene.collection.objects.link(o)
    else:
        bpy.ops.import_scene.gltf(filepath=ruta)
    nuevos = [bpy.data.objects[k] for k in bpy.data.objects.keys() if k not in antes]
    mallas = [o for o in nuevos if o.type == "MESH"]
    alto = bpy.data.objects[objeto] if objeto else max(mallas, key=lambda o: len(o.data.polygons))
    for o in nuevos:
        if o is not alto:
            bpy.data.objects.remove(o)
    return alto


# ----------------------------------------------------------------------------------------------------------------- exportar
def cmd_exportar():
    salida = arg("--salida"); nombre = arg("--nombre", "principe_remaster")
    bpy.ops.wm.open_mainfile(filepath=os.path.join(salida, "uv.blend"))
    ob = bpy.data.objects["principe"]
    base = np.load(os.path.join(salida, "base.npz"))
    for o in list(bpy.data.objects):
        if o is not ob:
            bpy.data.objects.remove(o)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m)
    for im in list(bpy.data.images):
        bpy.data.images.remove(im)
    col = bpy.data.images.load(os.path.join(salida, "juego_color.png")); col.colorspace_settings.name = "sRGB"; col.pack()
    nor = bpy.data.images.load(os.path.join(salida, "juego_normal.png")); nor.colorspace_settings.name = "Non-Color"; nor.pack()
    orm = bpy.data.images.load(os.path.join(salida, "juego_orm.png")); orm.colorspace_settings.name = "Non-Color"; orm.pack()
    m = bpy.data.materials.new("mat_principe"); m.use_nodes = True
    nt = m.node_tree; bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    t1 = nt.nodes.new("ShaderNodeTexImage"); t1.image = col
    t2 = nt.nodes.new("ShaderNodeTexImage"); t2.image = nor
    t3 = nt.nodes.new("ShaderNodeTexImage"); t3.image = orm
    nm = nt.nodes.new("ShaderNodeNormalMap"); nm.space = "TANGENT"
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(t1.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(t2.outputs["Color"], nm.inputs["Color"]); nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(t3.outputs["Color"], sep.inputs["Color"])
    nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"]); nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
    ob.data.materials.clear(); ob.data.materials.append(m)
    # esqueleto sin jerarquia (un hueso por hueso global, en el centroide ponderado), como original_bind.glb
    from mathutils import Vector
    me = ob.data
    P = arrays_malla(me)
    ao_d = bpy.data.armatures.new("esqueleto"); ao = bpy.data.objects.new("esqueleto", ao_d); bpy.context.scene.collection.objects.link(ao)
    pesos = {}
    for v in me.vertices:
        for g in v.groups:
            pesos.setdefault(g.group, []).append((v.index, g.weight))
    activar(ao); bpy.ops.object.mode_set(mode="EDIT")
    for gi, lst in sorted(pesos.items()):
        vi = np.array([x for x, _ in lst]); wi = np.array([w for _, w in lst])
        c = (P[vi] * wi[:, None]).sum(0) / max(wi.sum(), 1e-9)
        b = ao_d.edit_bones.new(ob.vertex_groups[gi].name); b.head = Vector(c.tolist()); b.tail = Vector((c + [0, 0, 0.03]).tolist())
    bpy.ops.object.mode_set(mode="OBJECT")
    ob.parent = ao; md = ob.modifiers.new("Armature", "ARMATURE"); md.object = ao
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(salida, nombre + ".blend"), compress=True)
    for o in bpy.data.objects:
        o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=os.path.join(salida, nombre + ".glb"), export_format="GLB", use_selection=True,
                              export_image_format="AUTO", export_tangents=True, export_apply=False, export_skins=True, export_yup=True)
    log("exportado", nombre)


def cmd_segmentos():
    """Escribe un .blend con la malla (grupos hNN) y el atributo de vertice `segmento` (codigo de pesos_geom.SEGMENTOS) para estiramiento.py.
    --entrada .blend|.glb --salida estirar.blend --datos analisis_original (restricciones.yaml: huesos por pieza) [--hombro-x 0.2 --hombro-z 1.48 --z-pierna 0.85]"""
    import re
    from assetgen_cli import pesos_geom as PG
    entrada, salida_b, datos = arg("--entrada"), arg("--salida"), arg("--datos")
    txt = open(os.path.join(datos, "restricciones.yaml")).read()
    huesos = {m.group(1): json.loads(m.group(2)) for m in re.finditer(r'nombre: "(\w+)".*?huesos_usados: (\[[^\]]*\])', txt)}
    sueltas = {"cabeza": huesos["000056_dcc7b0ee"], "daga": huesos["000055_eb47a112"], "faldon": huesos["000075_a124f3bb"]}
    if entrada.endswith(".blend"):
        bpy.ops.wm.open_mainfile(filepath=entrada)
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True); bpy.ops.import_scene.gltf(filepath=entrada)
    for o in list(bpy.data.objects):
        if o.type == "MESH" and not o.vertex_groups:
            bpy.data.objects.remove(o)
    ob = next(o for o in bpy.data.objects if o.type == "MESH"); me = ob.data
    P = arrays_malla(me)
    nom = {g.index: int(g.name[1:]) for g in ob.vertex_groups}
    suma, cen = {}, {}
    for v in me.vertices:
        for g in v.groups:
            h = nom[g.group]; suma[h] = suma.get(h, 0.0) + g.weight; cen[h] = cen.get(h, 0.0) + g.weight * P[v.index]
    cen = {h: tuple(cen[h] / suma[h]) for h in suma}
    mapa = PG.mapa_hueso_segmento(sueltas, cen, float(arg("--hombro-x", 0.2)), float(arg("--hombro-z", 1.48)), z_pierna=float(arg("--z-pierna", 0.85)))
    seg = np.zeros(len(me.vertices), np.int32)
    for v in me.vertices:
        if v.groups:
            h = nom[max(v.groups, key=lambda g: g.weight).group]
            seg[v.index] = PG.SEGMENTOS.index(mapa.get(h, "cuerpo"))
    at = me.attributes.new("segmento", "INT", "POINT"); at.data.foreach_set("value", seg)
    bpy.ops.wm.save_as_mainfile(filepath=salida_b, compress=True)
    log("segmentos", {PG.SEGMENTOS[i]: int((seg == i).sum()) for i in range(len(PG.SEGMENTOS))})


def cmd_resumen():
    """remaster.json: une subdivision, envoltura, UV, horneado y estiramiento (--estira-remaster / --estira-original, de estiramiento.py)."""
    salida = arg("--salida")
    cargar = lambda f: json.load(open(f)) if os.path.exists(f) else None      # noqa: E731
    sub = cargar(os.path.join(salida, "subdivision.json")); env = cargar(os.path.join(salida, "envoltura.json"))
    base = np.load(os.path.join(salida, "base.npz")); nombres = [str(x) for x in base["nombres"]]; pieza = base["pieza"]
    rangos, ini = {}, 0
    for i, nm in enumerate(nombres):
        idx = np.flatnonzero(pieza == i)
        rangos[nm] = {"triangulos": int(len(idx)), "primer_triangulo": int(idx[0]), "contiguo": bool(idx[-1] - idx[0] + 1 == len(idx))}
    res = {"triangulos_total": int(len(pieza)), "vertices_malla": int(len(base["P"])), "piezas": rangos, "subdivision": sub, "envoltura": env,
           "uv": cargar(os.path.join(salida, "uv_stats.json")), "horneado": cargar(os.path.join(salida, "hornear.json")),
           "estiramiento": {"remaster": cargar(arg("--estira-remaster", "")), "original": cargar(arg("--estira-original", ""))}}
    json.dump(res, open(os.path.join(salida, "remaster.json"), "w"), indent=1)
    log("remaster.json escrito")


if CMD == "resumen":
    cmd_resumen()
elif CMD == "segmentos":
    cmd_segmentos()
elif CMD == "preparar":
    cmd_preparar()
elif CMD == "envolver":
    cmd_envolver()
elif CMD == "uv":
    cmd_uv()
elif CMD == "hornear":
    cmd_hornear()
elif CMD == "exportar":
    cmd_exportar()
else:
    raise SystemExit("comando desconocido: %s" % CMD)
