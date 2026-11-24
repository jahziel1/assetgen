"""cara_unir.py -- etapa 2 del ajuste de la cara base MakeHuman: limpiar la cara, ajustar la apertura de los parpados y unirla a la cabeza.

    blender -b -P cara_unir.py -- --fase cara  --dir DIR [--ap-factor-izq 1] [--ap-factor-dch 1] [--grosor-parpado 1]
    blender -b -P cara_unir.py -- --fase unir  --dir DIR --glb cabeza.glb [--margen-corte ...] [--sin-hoja]

`DIR` es la salida de cara_base.py (cara_ajuste.blend, landmarks.json). Fase `cara` (rapida, sin la cabeza completa): (1) cierra la boca
(suelda el labio superior con el inferior); (2) recorta la cara al ovalo facial (apertura morfologica = sin dientes ni rabitos, solo la
componente mayor) y suaviza el borde con Taubin; (3) escala la apertura de los parpados (`--ap-factor-*` multiplica la apertura actual,
el parpado superior asume el 80 %), sin que atraviesen el globo, y da grosor al borde con un anillo extruido hacia el globo;
escribe DIR/cara_limpia.blend y DIR/cara_limpia.json (aperturas geometricas).
Fase `unir`: importa la cabeza generada (y la suelda por posicion), borra las caras de la cabeza que quedan bajo la cara nueva (dentro del
contorno y a menos de --dist-corte de su superficie, mas las aberturas y la cuenca de los ojos), tapa los cortes pequenos del hueco (pelos de
la barba) y cose la cara con un faldon: el contorno de la cara se prolonga --faldon-mm hacia fuera y se hunde tras la superficie de la cabeza,
de modo que el borde del hueco lo tapa (costura de solape; el borde del hueco de una cabeza TRELLIS con barba de geometria es de varias capas
y de pelos sueltos y no admite una soldadura vertice a vertice limpia). Despues desenvuelve la cara (UV propias, material `cara_tex` 2048^2 con el color plano de la piel de la cabeza) y escribe cabeza_unida.blend,
cabeza_unida.glb, landmarks.json (seccion `unir`) y hoja_unida.jpg. Unidades: metros, Z arriba, frente en -Y."""
import json
import math
import os
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "..", "cli"))
import cara_comun as C  # noqa: E402
from assetgen_cli import cara_geom as G  # noqa: E402

a = sys.argv[sys.argv.index("--") + 1:]


def arg(n, d=None):
    return a[a.index(n) + 1] if n in a else d


FASE = arg("--fase")
DIR = arg("--dir")
if FASE not in ("cara", "unir") or not DIR:
    sys.exit("uso: --fase cara|unir --dir DIR [...]")
REPARTO_SUP = 0.8        # parte del cambio de apertura que asume el parpado superior


# ---------------------------------------------------------------- utilidades de malla
def bordes_lazos(bm):
    """Lazos de borde de un bmesh -> lista de listas de indices de vertice (ordenados)."""
    return G.lazo_ordenado([(e.verts[0].index, e.verts[1].index) for e in bm.edges if e.is_boundary])


def coords(bm):
    return np.array([v.co[:] for v in bm.verts], dtype=np.float64)


def anillos(bm, semillas, n):
    """Distancia (en aristas) a `semillas` hasta n pasos: dict vertice->distancia."""
    d = {i: 0 for i in semillas}
    fr = list(semillas)
    for k in range(1, n + 1):
        sig = []
        for i in fr:
            for e in bm.verts[i].link_edges:
                j = e.other_vert(bm.verts[i]).index
                if j not in d:
                    d[j] = k
                    sig.append(j)
        fr = sig
    return d


def lazo_de(bm, lazos, punto):
    """Lazo cuyo centroide esta mas cerca de `punto`."""
    V = coords(bm)
    return min(lazos, key=lambda l: np.linalg.norm(V[l].mean(axis=0) - np.asarray(punto)))


def orientar_hacia_fuera(bm, frente_y=-1.0):
    """Normales de las caras coherentes y con el lado visible hacia el frente (-Y) en la mayoria de las caras."""
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ny = np.mean([f.normal.y for f in bm.faces])
    if ny * frente_y < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces)


def cargar_lm():
    return json.load(open(os.path.join(DIR, "landmarks.json")))


# ================================================================= FASE CARA
def fase_cara():
    FAC = {"izq": float(arg("--ap-factor-izq", "1")), "dch": float(arg("--ap-factor-dch", "1"))}
    GROSOR = float(arg("--grosor-parpado", "1"))          # multiplicador del grosor del borde del parpado (1 = 1.6 mm)
    FRAC_ADELANTO = float(arg("--globo-adelanto", "0.0"))  # fraccion del hueco globo-parpados que se adelanta el globo (0 = no mover)
    MARGEN = float(arg("--margen-borde", "0.0"))          # del ancho de cara, hacia dentro (+) o fuera (-) del ovalo MediaPipe
    lm = cargar_lm()
    bpy.ops.wm.open_mainfile(filepath=os.path.join(DIR, "cara_ajuste.blend"))
    ob = bpy.data.objects["cara"]
    ob_gen = bpy.data.objects["cabeza_generada"]
    dg = bpy.context.evaluated_depsgraph_get()
    bvh_gen = BVHTree.FromObject(ob_gen, dg)
    m3d = {int(k): np.array(v) for k, v in lm["generada"]["m3d"].items()}
    ancho = lm["ajuste"]["ancho_cara_generada_m"]
    ojos = lm["ajuste"]["ojos"]
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.verts.ensure_lookup_table()
    info = {}

    # ---- 1. boca: cerrar la hendidura con una tira de triangulos entre los dos labios (sin fundir vertices: las densidades difieren) -------
    lazos = bordes_lazos(bm)
    c_boca = (m3d[61] + m3d[291]) / 2
    l_boca = lazo_de(bm, lazos, c_boca)
    V = coords(bm)
    xz = [(V[i][0], V[i][2]) for i in l_boca]
    i0 = min(range(len(l_boca)), key=lambda i: xz[i][0])
    i1 = max(range(len(l_boca)), key=lambda i: xz[i][0])
    n_l = len(l_boca)
    cad_a = [(i0 + k) % n_l for k in range(((i1 - i0) % n_l) + 1)]
    cad_b = [(i1 + k) % n_l for k in range(((i0 - i1) % n_l) + 1)][::-1]            # de i0 a i1 por el otro lado
    A = [l_boca[i] for i in cad_a]
    B = [l_boca[i] for i in cad_b]
    pares = G.emparejar_cadenas([V[i][0] for i in A], [V[i][0] for i in B])
    objetivo = {}
    for i, j in pares:                                         # cada vertice va hacia el punto medio con sus parejas (mitad del hueco)
        m = (V[A[i]] + V[B[j]]) / 2
        for v in (A[i], B[j]):
            objetivo.setdefault(v, []).append(m)
    for v, ms in objetivo.items():
        if v in (A[0], A[-1]):
            continue
        bm.verts[v].co = Vector(0.5 * V[v] + 0.5 * np.mean(ms, axis=0))
    tiras = 0
    for k in range(len(pares) - 1):
        (i, j), (i2, j2) = pares[k], pares[k + 1]
        if (i2, j2) == (i + 1, j + 1):
            cand = [[A[i], A[i2], B[j2], B[j]]]
        elif i2 == i + 1:
            cand = [[A[i], A[i2], B[j]]]
        else:
            cand = [[A[i], B[j2], B[j]]]
        for vs in cand:
            if len(set(vs)) >= 3:
                bm.faces.new([bm.verts[x] for x in dict.fromkeys(vs)])
                tiras += 1
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    print("boca: cadenas %d y %d vertices, %d caras de cierre" % (len(A), len(B), tiras))

    # ---- 2. recorte al ovalo, apertura morfologica, componente mayor -------------------------------------------------------------
    ovalo = [(m3d[i][0], m3d[i][2]) for i in C.OVALO if i in m3d]
    V = coords(bm)
    d = np.array([G.distancia_poligono(ovalo, (p[0], p[2])) for p in V])
    caras = [[v.index for v in f.verts] for f in bm.faces]
    activas = {k for k, f in enumerate(caras) if all(d[i] >= MARGEN * ancho for i in f)}
    # caras "virtuales" fuera de cada arista del borde exterior (para que la erosion trate el borde como frontera real)
    lazos = bordes_lazos(bm)
    centros_ojos = [np.array(ojos[k]["centro"]) for k in ("izq", "dch")]
    prop = [l for l in lazos if l != l_boca and not any(np.linalg.norm(V[l].mean(axis=0)[[0, 2]] - c[[0, 2]]) < 0.05 for c in centros_ojos)]
    l_ext = max(prop, key=len) if prop else max(lazos, key=len)
    ext_aristas = {(min(l_ext[i], l_ext[(i + 1) % len(l_ext)]), max(l_ext[i], l_ext[(i + 1) % len(l_ext)])) for i in range(len(l_ext))}
    virt = []
    nv = len(V)
    for k, e in enumerate(sorted(ext_aristas)):
        virt.append([e[0], e[1], nv + k])
    caras_v = caras + virt
    nuevas = G.abrir_region(caras_v, activas, 2)
    nuevas = {k for k in nuevas if k < len(caras)}
    mayor = G.componente_mayor(caras, nuevas)
    quitar = [bm.faces[k] for k in range(len(caras)) if k not in mayor]
    print("recorte al ovalo: caras %d -> %d (quitadas %d)" % (len(caras), len(mayor), len(quitar)))
    bmesh.ops.delete(bm, geom=quitar, context="FACES")
    sueltos = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=sueltos, context="VERTS")
    bm.verts.ensure_lookup_table()
    bm.verts.index_update()
    bm.faces.ensure_lookup_table()

    # ---- borde: Taubin + pegado a la superficie ------------------------------------------------------------------------------
    lazos = bordes_lazos(bm)
    V = coords(bm)
    l_ext = max([l for l in lazos if np.linalg.norm(V[l].mean(axis=0) - c_boca) > 0.04 and
                 not any(np.linalg.norm(V[l].mean(axis=0)[[0, 2]] - c[[0, 2]]) < 0.05 for c in centros_ojos)], key=len)
    suav = np.array(G.lazo_taubin(V[l_ext].tolist(), 15))
    dis = suav - V[l_ext]
    ring = anillos(bm, l_ext, 3)
    for k, vi in enumerate(l_ext):
        bm.verts[vi].co = Vector(suav[k])
    for vi, dd in ring.items():                                # el desplazamiento se reparte hacia dentro (0.5, 0.25, 0.1)
        if dd == 0:
            continue
        fac = (0.5, 0.25, 0.1)[dd - 1]
        # desplazamiento del vertice del borde mas cercano en el grafo = media de los desplazamientos de borde cercanos
        p = V[vi]
        dn = np.linalg.norm(V[l_ext] - p, axis=1)
        w = np.exp(-(dn / (0.012)) ** 2)
        dv = (dis * w[:, None]).sum(axis=0) / max(w.sum(), 1e-9)
        bm.verts[vi].co = Vector(p + fac * dv)
    V = coords(bm)
    movs = 0
    for vi in l_ext:                                          # pegado a la superficie generada (solo si es coherente con la cara)
        p = Vector(V[vi])
        loc, nor, ix, dist = bvh_gen.find_nearest(p, 0.006)
        if loc is not None:
            bm.verts[vi].co = p.lerp(loc, 0.8)
            movs += 1
    sn = np.array(G.lazo_taubin(coords(bm)[l_ext].tolist(), 4))
    for k, vi in enumerate(l_ext):
        bm.verts[vi].co = Vector(sn[k])
    print("borde: %d vertices, %d pegados a la superficie" % (len(l_ext), movs))

    # ---- 3. parpados: apertura y grosor -----------------------------------------------------------------------------------------
    lazos = bordes_lazos(bm)
    V = coords(bm)
    grosores = {}
    aperturas = {}
    adelantos = {}
    for etiqueta in ("izq", "dch"):
        c = np.array(ojos[etiqueta]["centro"])
        r = ojos[etiqueta]["radio"]
        lz = [l for l in lazos if np.linalg.norm(V[l].mean(axis=0)[[0, 2]] - c[[0, 2]]) < 0.04]
        if len(lz) != 1:
            sys.exit("no encuentro el lazo del ojo %s (%d candidatos)" % (etiqueta, len(lz)))
        lz = lz[0]
        # el globo se adelanta hasta quedar a ~1.6 mm tras el borde de los parpados (el proxy queda hundido en la cuenca: sin esto el borde de
        # los parpados cuelga ~7 mm por delante del globo y, al bajar los parpados, se forma un pozo oscuro encima del iris)
        og = bpy.data.objects["ojo_" + etiqueta]
        bvh0 = BVHTree.FromObject(og, bpy.context.evaluated_depsgraph_get())
        huecos = []
        for i in lz:
            vv = V[i] - c
            dn = np.linalg.norm(vv)
            if dn > 1e-9 and -vv[1] / dn >= 0.6:
                hit = bvh0.ray_cast(Vector(c), Vector(vv / dn))
                if hit[0] is not None:
                    huecos.append(dn - hit[3])
        delta = float(np.clip(FRAC_ADELANTO * (np.median(huecos) - 0.0016), 0.0, 0.009)) if huecos else 0.0
        if delta > 0:
            og.data.transform(Matrix.Translation((0.0, -delta, 0.0)))
            og.data.update()
            c = c + np.array([0.0, -delta, 0.0])
        adelantos[etiqueta] = delta
        xz = [(V[i][0], V[i][2]) for i in lz]
        isup, iinf = G.separar_parpados(xz)
        sup = [lz[i] for i in isup]
        inf = [lz[i] for i in iinf]
        xs_s, zs_s = [V[i][0] for i in sup], [V[i][2] for i in sup]
        xs_i, zs_i = [V[i][0] for i in inf], [V[i][2] for i in inf]
        ys_s, ys_i = [V[i][1] for i in sup], [V[i][1] for i in inf]
        x0, x1 = min(xs_s[0], xs_i[0]), max(xs_s[-1], xs_i[-1])
        W = x1 - x0
        xm = (x0 + x1) / 2
        gap = lambda x: max(0.0, G.interpolar_lineal(xs_s, zs_s, x) - G.interpolar_lineal(xs_i, zs_i, x))
        xp = np.linspace(x0, x1, 41)
        apert_antes = max(gap(x) for x in xp) / W
        f = FAC[etiqueta]
        h_up, h_lo = 0.46 * W, 0.32 * W
        proteg = set(lz)
        otros = {v for l in lazos if l is not lz for v in l}
        for v in bm.verts:
            if v.index in otros:
                continue
            p = V[v.index]
            nx = abs(p[0] - xm) / (W / 2)
            if nx > 1.6 or abs(p[2] - c[2]) > 1.1 * W:
                continue
            xc = min(max(p[0], x0), x1)
            zu, zl = G.interpolar_lineal(xs_s, zs_s, xc), G.interpolar_lineal(xs_i, zs_i, xc)
            yu, yl = G.interpolar_lineal(xs_s, ys_s, xc), G.interpolar_lineal(xs_i, ys_i, xc)
            dzs, dzi = G.desplazar_margenes(zu, zl, f, REPARTO_SUP)
            gx = G.suavizado(1.0 - max(0.0, nx - 1.0) / 0.6)
            if v.index in proteg:                              # el borde del parpado rueda exactamente lo que pide la apertura
                arriba = p[2] >= (zu + zl) / 2
                th, w = G.giro_parpado(p[1] - c[1], p[2] - c[2], dzs if arriba else dzi), 1.0
            elif p[2] >= zu - 1e-9:
                th, w = G.giro_parpado(yu - c[1], zu - c[2], dzs), G.suavizado(1.0 - (p[2] - zu) / h_up) * gx
            elif p[2] <= zl + 1e-9:
                th, w = G.giro_parpado(yl - c[1], zl - c[2], dzi), G.suavizado(1.0 - (zl - p[2]) / h_lo) * gx
            else:
                continue
            dy_, dz_ = p[1] - c[1], p[2] - c[2]
            rho = math.hypot(dy_, dz_)
            if rho < 1e-9 or w <= 0:
                continue
            phi = math.atan2(dz_, -dy_) - th * w            # el parpado gira alrededor del globo (conserva su distancia al centro)
            v.co.y = c[1] - rho * math.cos(phi)
            v.co.z = c[2] + rho * math.sin(phi)
        # que no atraviesen el globo: la distancia a la superficie REAL del globo (no a una esfera: el proxy de ojo no es esferico, su radio
        # varia de ~16 mm al frente a ~25 mm en el borde) debe ser >= 1.0 mm (1.6 mm en el borde del parpado); se saca hacia delante (-Y),
        # conservando x, z, para que la apertura no cambie
        bvh_ojo = BVHTree.FromObject(bpy.data.objects["ojo_" + etiqueta], bpy.context.evaluated_depsgraph_get())

        def radio_globo(p):
            d = Vector(p - c)
            if d.length < 1e-9:
                return 0.0
            hit = bvh_ojo.ray_cast(Vector(c), d.normalized())
            return hit[3] if hit[0] is not None else 0.0
        for v in bm.verts:
            p = np.array(v.co[:])
            vv = p - c
            dn = np.linalg.norm(vv)
            if dn < 1e-9 or dn > 2.0 * r or -vv[1] / dn < 0.6:       # solo la parte delantera del globo (a menos de ~53 grados del eje -Y):
                continue                                              # el ecuador y la parte trasera del proxy quedan dentro de la cabeza
            minimo = 0.0016 if v.index in proteg else 0.0010
            for _ in range(24):                                       # hasta 12 mm hacia delante
                if np.linalg.norm(np.array(v.co[:]) - c) >= radio_globo(np.array(v.co[:])) + minimo:
                    break
                v.co.y -= 0.0005
        # anillo extruido hacia el globo (grosor del borde del parpado): a `GROSOR`*1.6 mm tras el borde, sin pasar de la superficie del globo + 0.4 mm
        aristas = [e for e in bm.edges if e.is_boundary and e.verts[0].index in proteg and e.verts[1].index in proteg]
        ret = bmesh.ops.extrude_edge_only(bm, edges=aristas)
        nuevos = [g for g in ret["geom"] if isinstance(g, bmesh.types.BMVert)]
        grosor = 0.0016 * GROSOR
        for v in nuevos:
            vv = np.array(v.co[:]) - c
            dist = np.linalg.norm(vv)
            destino = max(radio_globo(np.array(v.co[:])) + 0.0004, dist - grosor)
            v.co = Vector(c + vv / dist * destino)
        grosores[etiqueta] = grosor
        bm.verts.index_update()
        bm.verts.ensure_lookup_table()
        V = coords(bm)
        # apertura geometrica resultante (sobre el borde delantero = el que mira al frente)
        sup2 = [bm.verts[i] for i in sup]
        inf2 = [bm.verts[i] for i in inf]
        gap2 = lambda x: max(0.0, G.interpolar_lineal([q.co.x for q in sup2], [q.co.z for q in sup2], x) -
                             G.interpolar_lineal([q.co.x for q in inf2], [q.co.z for q in inf2], x))
        aperturas[etiqueta] = dict(antes=apert_antes, despues=max(gap2(x) for x in xp) / W, ancho_m=W, factor=f)
        lazos = bordes_lazos(bm)
        V = coords(bm)
    print("globos adelantados (mm), huecos medianos globo-parpados antes:", {k: round(v * 1000, 2) for k, v in adelantos.items()})
    print("parpados:", json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in aperturas.items()}))
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    orientar_hacia_fuera(bm)

    # ---- guardar ------------------------------------------------------------------------------------------------------------
    me = ob.data
    nuevo_me = bpy.data.meshes.new("cara")
    bm.to_mesh(nuevo_me)
    for p in nuevo_me.polygons:
        p.use_smooth = True
    for m in me.materials:
        nuevo_me.materials.append(m)
    ob.data = nuevo_me
    bpy.data.meshes.remove(me)
    bpy.data.objects.remove(ob_gen)
    for nom in ("hoja_cam",):
        if nom in bpy.data.objects:
            bpy.data.objects.remove(bpy.data.objects[nom])
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(DIR, "cara_limpia.blend"), compress=True)
    lazos = bordes_lazos(bm)
    info = dict(globo_adelantado_m=adelantos, aperturas=aperturas, grosor_parpado_m=grosores, caras=len(bm.faces), vertices=len(bm.verts),
                lazos_borde=sorted(len(l) for l in lazos), boca_cerrada_caras=tiras)
    json.dump(info, open(os.path.join(DIR, "cara_limpia.json"), "w"), indent=1)
    print("CARA_LIMPIA", json.dumps(info))




# ================================================================= FASE UNIR
def dist_poligono_np(poly, P):
    """Distancia con signo (+ dentro) de los puntos P (n,2) al poligono cerrado poly (m,2), vectorizada."""
    poly = np.asarray(poly)
    q = np.roll(poly, -1, axis=0)
    ab = q - poly
    l2 = np.maximum((ab ** 2).sum(1), 1e-18)
    dmin = np.full(len(P), np.inf)
    dentro = np.zeros(len(P), dtype=bool)
    for i in range(len(poly)):
        t = np.clip(((P - poly[i]) @ ab[i]) / l2[i], 0, 1)
        d = np.linalg.norm(P - (poly[i] + t[:, None] * ab[i]), axis=1)
        dmin = np.minimum(dmin, d)
        cond = (poly[i][1] > P[:, 1]) != (q[i][1] > P[:, 1])
        xint = ab[i][0] * (P[:, 1] - poly[i][1]) / (ab[i][1] if ab[i][1] != 0 else 1e-18) + poly[i][0]
        dentro ^= cond & (P[:, 0] < xint)
    return np.where(dentro, dmin, -dmin)


def etiquetas_componentes(F, n):
    """Etiqueta de componente de cada cara triangular F (m,3), conectando caras que comparten una ARISTA (n = numero de vertices).
    Las piezas unidas solo por un vertice quedan como componentes distintas."""
    m = len(F)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e.sort(axis=1)
    key = e[:, 0].astype(np.int64) * n + e[:, 1]
    fid = np.tile(np.arange(m), 3)
    orden = np.argsort(key, kind="stable")
    key, fid = key[orden], fid[orden]
    igual = key[1:] == key[:-1]
    u, v = fid[:-1][igual], fid[1:][igual]
    lab = np.arange(m)
    for _ in range(5000):
        mn = np.minimum(lab[u], lab[v])
        nuevo = lab.copy()
        np.minimum.at(nuevo, u, mn)
        np.minimum.at(nuevo, v, mn)
        nuevo = nuevo[nuevo]
        if (nuevo == lab).all():
            break
        lab = nuevo
    return lab


def uv_frontal(P):
    uv = np.stack([P[:, 0], P[:, 2]], axis=1)
    return uv


def normalizar_uv(uv, pad=0.03):
    uv = np.asarray(uv, dtype=np.float64)
    mn, mx = uv.min(axis=0), uv.max(axis=0)
    esc = (1 - 2 * pad) / max(mx[0] - mn[0], mx[1] - mn[1])
    return (uv - (mn + mx) / 2) * esc + 0.5


def desenvolver(bm_cara, metodo):
    """UV por vertice de una malla de cara (sin costuras: isla unica) con el desenvuelto de Blender (`metodo`: CONFORMAL | ANGLE_BASED)."""
    me = bpy.data.meshes.new("uv_tmp")
    bm = bm_cara.copy()
    bm.to_mesh(me)
    bm.free()
    me.uv_layers.new(name="UVMap")
    ob = bpy.data.objects.new("uv_tmp", me)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.unwrap(method=metodo, fill_holes=True, margin=0.0)
    bpy.ops.object.mode_set(mode="OBJECT")
    uv = np.zeros((len(me.vertices), 2))
    capa = me.uv_layers[0].data
    for lp in me.loops:
        uv[lp.vertex_index] = capa[lp.index].uv
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(me)
    return uv


def elegir_uv(bm_cara):
    """Compara UV frontal, las de Blender (conforme, por angulos) y mezclas frontal/conforme (la conforme se alinea al frontal con Procrustes);
    puntua solo las caras que miran al frente (sin las paredes del borde de los parpados ni el faldon) y elige la de menos distorsion:
    0.5 * (anisotropia p90 - 1) + desviacion del area + penalizacion por triangulos invertidos."""
    P = coords(bm_cara)
    tri, frente = [], []
    for f in bm_cara.faces:
        vs = [v.index for v in f.verts]
        for k in range(1, len(vs) - 1):
            tri.append([vs[0], vs[k], vs[k + 1]])
            frente.append(f.normal.y < -0.25)
    nucleo = [t for t, fr in zip(tri, frente) if fr]
    front = normalizar_uv(uv_frontal(P))
    cand = {"frontal": front}
    for nom, met in (("conforme", "CONFORMAL"), ("angulos", "ANGLE_BASED")):
        uv = desenvolver(bm_cara, met)
        m, t = G.procrustes_2d(uv.tolist(), front.tolist(), reflexion=True)
        al = np.array([[m[0][0] * x + m[0][1] * y + t[0], m[1][0] * x + m[1][1] * y + t[1]] for x, y in uv])
        cand[nom] = normalizar_uv(al)
    for w in (0.25, 0.5, 0.75):
        cand["mezcla%.2f" % w] = normalizar_uv(w * cand["frontal"] + (1 - w) * cand["conforme"])
    tabla = {n: G.distorsion_uv(P.tolist(), uv.tolist(), nucleo) for n, uv in cand.items()}
    puntaje = {n: 0.5 * (t["aniso_p90"] - 1.0) + t["area_desv"] + 10.0 * t["invertidos"] / max(t["n"], 1) for n, t in tabla.items()}
    mejor = min(puntaje, key=puntaje.get)
    for n, t in tabla.items():
        print("UV %-11s aniso media %.3f p90 %.3f area_desv %.3f invertidos %d de %d%s" %
              (n, t["aniso_media"], t["aniso_p90"], t["area_desv"], t["invertidos"], t["n"], "  <- elegida" if n == mejor else ""))
    return mejor, cand[mejor], tabla


def debug_render(bh, aristas, ruta, R):
    """Dibujo de diagnostico: la cabeza (gris) con las aristas dadas en rojo, de frente y de 3/4."""
    me = bpy.data.meshes.new("dbg")
    bh.to_mesh(me)
    ob = bpy.data.objects.new("dbg", me)
    bpy.context.scene.collection.objects.link(ob)
    cu = bpy.data.curves.new("dbg_c", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = 0.0012
    for e in aristas:
        sp = cu.splines.new("POLY")
        sp.points.add(1)
        for k in range(2):
            sp.points[k].co = (*e.verts[k].co[:], 1.0)
    for o in bpy.data.objects:
        if o.name != "dbg":
            o.hide_render = True
    ob2 = bpy.data.objects.new("dbg_c", cu)
    bpy.context.scene.collection.objects.link(ob2)
    mr = bpy.data.materials.new("rojo")
    mr.diffuse_color = (1, 0, 0, 1)
    cu.materials.append(mr)
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.render.resolution_x = sc.render.resolution_y = 800
    sh = sc.display.shading
    sh.light = "STUDIO"
    sh.color_type = "MATERIAL"
    sh.show_cavity = True
    sc.world = bpy.data.worlds.new("w")
    sc.world.color = (0.4, 0.4, 0.4)
    cam = bpy.data.objects.new("c", bpy.data.cameras.new("c"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = float(arg("--diag-esc", "0.5"))
    cam.data.clip_end = 100
    for nom, az in (("f", 0), ("t", 40), ("s", 85)):
        t = math.radians(az)
        tgt = Vector([float(x) for x in arg("--diag-tgt", "0,0,0.1").split(",")])
        cam.location = tgt + Vector((math.sin(t) * 3, -math.cos(t) * 3, 0))
        cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = ruta + "_" + nom + ".png"
        bpy.ops.render.render(write_still=True)


def fase_unir():
    GLB = arg("--glb")
    if not GLB:
        sys.exit("falta --glb")
    D_CORTE = float(arg("--dist-corte", "0.045"))            # fraccion del ancho de cara: distancia maxima a la cara para borrar la cabeza
    MARGEN_MM = float(arg("--margen-corte-mm", "1.0"))        # el hueco queda este margen hacia dentro del contorno de la cara
    lm = cargar_lm()
    ancho = lm["ajuste"]["ancho_cara_generada_m"]
    bpy.ops.wm.open_mainfile(filepath=os.path.join(DIR, "cara_limpia.blend"))
    ob_cara = bpy.data.objects["cara"]
    ojos = [bpy.data.objects["ojo_izq"], bpy.data.objects["ojo_dch"]]
    pre = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=GLB)
    mallas = [o for o in bpy.data.objects if o.type == "MESH" and o.name not in pre]
    if len(mallas) > 1:
        for o in mallas:
            o.select_set(True)
        bpy.context.view_layer.objects.active = mallas[0]
        bpy.ops.object.join()
    ob_h = [o for o in bpy.data.objects if o.type == "MESH" and o.name not in pre][0]
    ob_h.name = "cabeza"
    for o in list(bpy.data.objects):                              # vacios del importador
        if o.type == "EMPTY":
            if ob_h.parent == o:
                ob_h.parent = None
            bpy.data.objects.remove(o)
    ob_h.data.name = "cabeza"
    if not np.allclose(np.array(ob_h.matrix_world), np.eye(4), atol=1e-6):
        bpy.context.view_layer.objects.active = ob_h
        ob_h.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if ob_h.data.has_custom_normals:
        bpy.context.view_layer.objects.active = ob_h
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
    mat_cabeza = ob_h.data.materials[0]
    print("cabeza importada: %d vertices, %d caras" % (len(ob_h.data.vertices), len(ob_h.data.polygons)))

    # ---- soldar la cabeza por posicion (la importacion duplica vertices en las costuras de UV; las UV viven en las esquinas) ---------
    bh = bmesh.new()
    bh.from_mesh(ob_h.data)
    bmesh.ops.remove_doubles(bh, verts=bh.verts, dist=1e-6)
    bh.verts.index_update()
    bh.edges.index_update()
    bh.faces.index_update()
    bh.verts.ensure_lookup_table()
    bh.faces.ensure_lookup_table()
    bh.edges.ensure_lookup_table()
    n_borde_orig = sum(1 for e in bh.edges if e.is_boundary)
    n_nomanif_orig = sum(1 for e in bh.edges if len(e.link_faces) > 2)
    for e in bh.edges:                                            # marca las aristas de borde originales (las etiquetas sobreviven a los borrados)
        e.tag = e.is_boundary
    Vh = coords(bh)
    Fh = np.array([[v.index for v in f.verts] for f in bh.faces])
    print("cabeza soldada: %d vertices, %d caras, %d aristas de borde y %d no-manifold previas" % (len(Vh), len(Fh), n_borde_orig, n_nomanif_orig))

    # ---- cara: borde, BVH, color de piel -------------------------------------------------------------------------------------
    bc = bmesh.new()
    bc.from_mesh(ob_cara.data)
    bc.verts.ensure_lookup_table()
    lazos_c = bordes_lazos(bc)
    Vc = coords(bc)
    ctr_ojos = [np.array(lm["ajuste"]["ojos"][k]["centro"]) for k in ("izq", "dch")]
    l_ext = max(lazos_c, key=len)
    R = Vc[l_ext]
    anillos_ojos = [l for l in lazos_c if l is not l_ext]
    bc_t = bc.copy()
    bmesh.ops.triangulate(bc_t, faces=bc_t.faces)
    bvh_c = BVHTree.FromBMesh(bc_t)
    # color plano de la piel: mediana de la textura de la cabeza alrededor de los pomulos (antes de cortar)
    m3d = {int(k): np.array(v) for k, v in lm["generada"]["m3d"].items()}
    nodo_tex = [n for n in mat_cabeza.node_tree.nodes if n.type == "TEX_IMAGE" and n.image.colorspace_settings.name == "sRGB"][0]
    img_h = nodo_tex.image
    tw, th = img_h.size
    px = np.empty(tw * th * 4, dtype=np.float32)
    img_h.pixels.foreach_get(px)
    px = px.reshape(th, tw, 4)
    uvl = bh.loops.layers.uv.active
    from mathutils import kdtree
    kd = kdtree.KDTree(len(Vh))
    for i, p in enumerate(Vh):
        kd.insert(Vector(p), i)
    kd.balance()
    por_punto = {}
    for k in (10, 151, 9, 8, 168, 6, 1, 50, 280, 205, 425, 116, 345, 234, 454):
        if k not in m3d:
            continue
        mu = []
        for co, i, d in kd.find_range(Vector(m3d[k]), 0.012):
            u, v = bh.verts[i].link_loops[0][uvl].uv
            mu.append(px[min(th - 1, max(0, int(v * th))), min(tw - 1, max(0, int(u * tw))), :3])
        if mu:
            por_punto[k] = np.median(np.array(mu), axis=0)
    # la piel: mediana de los puntos con color de piel (los de barba, pelo y capucha son mucho mas oscuros: brillo < 40 % del mas claro)
    claro = max((float(v.sum()) for v in por_punto.values()), default=1.0)
    skin_k = [k for k, v in por_punto.items() if float(v.sum()) >= 0.40 * claro and k not in (10, 151)]
    piel = np.median(np.array([por_punto[k] for k in skin_k]), axis=0) if skin_k else np.array([0.72, 0.5, 0.4])
    print("color plano de piel: mediana de %s; valores de imagen %s" % (skin_k, np.round(piel, 3)))
    if "--diag" in a:
        print("DIAG color por landmark", {k: np.round(v, 2).tolist() for k, v in por_punto.items()})
    del px

    # ---- borrar de la cabeza lo que queda bajo la cara -------------------------------------------------------------------------
    cent = Vh[Fh].mean(axis=1)
    dpol = dist_poligono_np(R[:, [0, 2]], cent[:, [0, 2]])
    # (A) caras de la cabeza dentro del contorno de la cara (en el plano frontal) y a menos de D1 de la superficie de la cara: son las que
    # sustituye. La distancia a la superficie (no la profundidad) acota tambien las laminas laterales que corren paralelas al eje Y.
    d1 = D_CORTE * ancho
    borde_cara = np.vstack([Vc[l] for l in lazos_c])
    cand = np.where(dpol >= MARGEN_MM / 1000.0)[0]
    borrar = []
    for fi in cand:
        loc, nor, ix, dist = bvh_c.find_nearest(Vector(cent[fi]), d1)
        if loc is None:
            continue
        borrar.append(fi)
    # (B) lo que hay dentro de las aberturas de los ojos (la cabeza tiene sus propios ojos: los globos nuevos van aparte)
    for l in anillos_ojos:
        poli = Vc[l][:, [0, 2]]
        dent = dist_poligono_np(poli, cent[:, [0, 2]]) >= -0.0025
        cy = Vc[l][:, 1].mean()
        # ademas, la zona elipsoidal del ojo (parpados y cuenca de la cabeza, que la cara ya no sigue al bajar los parpados)
        cx, cz = poli.mean(axis=0)
        W = poli[:, 0].max() - poli[:, 0].min()
        zona = (((cent[:, 0] - cx) / (0.85 * W)) ** 2 + ((cent[:, 2] - cz) / (0.60 * W)) ** 2) <= 1.0
        sel = np.where((dent | zona) & (cent[:, 1] > cy - 0.03) & (cent[:, 1] < cy + 0.045))[0]
        borrar.extend(sel.tolist())
    borrar = np.array(sorted(set(borrar)), dtype=int)
    en_poly = np.where(dpol >= MARGEN_MM / 1000.0)[0]
    lejos = np.setdiff1d(en_poly, borrar)
    print("caras de la cabeza bajo la cara (punto mas cercano en el interior, a menos de %.1f mm) o en las aberturas de los ojos: %d; "
          "dentro del contorno y no borradas (superficie trasera, barba lejana): %d" % (d1 * 1000, len(borrar), len(lejos)))
    bmesh.ops.delete(bh, geom=[bh.faces[i] for i in borrar], context="FACES")
    bh.verts.index_update()
    bh.verts.ensure_lookup_table()
    bh.edges.ensure_lookup_table()
    bh.faces.ensure_lookup_table()
    # restos sueltos: componentes pequenas dentro de la region de la cara
    bh.verts.index_update()
    Vh = coords(bh)
    Fh = np.array([[v.index for v in f.verts] for f in bh.faces])
    lab = etiquetas_componentes(Fh, len(Vh))
    cent = Vh[Fh].mean(axis=1)
    dpol2 = dist_poligono_np(R[:, [0, 2]], cent[:, [0, 2]])
    u_, c_ = np.unique(lab, return_counts=True)
    tam = dict(zip(u_.tolist(), c_.tolist()))
    mayor = max(tam, key=tam.get)
    quitar = set()
    for etiq, n_f in tam.items():
        if etiq == mayor:
            continue
        sel = np.where(lab == etiq)[0]
        if dpol2[sel].max() > -0.03 * ancho:                   # componente dentro (o pegada a) la region de la cara
            quitar.add(etiq)
    f_quitar = [bh.faces[i] for i in np.where(np.isin(lab, list(quitar)))[0]]
    bmesh.ops.delete(bh, geom=f_quitar, context="FACES")
    print("componentes de la cabeza tras borrar: %d; piezas sueltas quitadas en la region de la cara: %d (%d caras)" %
          (len(tam), len(quitar), len(f_quitar)))
    bh.verts.ensure_lookup_table()
    bh.edges.ensure_lookup_table()
    bh.faces.ensure_lookup_table()

    # ---- lazo del hueco: aristas de borde nuevas (que no eran borde antes) ---------------------------------------------------------
    def aristas_nuevas():
        return [e for e in bh.edges if e.is_boundary and not e.tag]
    bh.verts.index_update()
    for ronda in range(6):                                        # quitar pellizcos: vertices con mas de 2 aristas de borde nuevas
        nuevas = aristas_nuevas()
        cuenta = {}
        for e in nuevas:
            for v in e.verts:
                cuenta[v.index] = cuenta.get(v.index, 0) + 1
        malos = [i for i, c in cuenta.items() if c > 2]
        if not malos:
            break
        fq = set()
        for i in malos:
            for f in bh.verts[i].link_faces:
                if any(e.is_boundary and not e.tag for e in f.edges):
                    fq.add(f)
        bmesh.ops.delete(bh, geom=list(fq), context="FACES")
        bh.verts.index_update()
        bh.verts.ensure_lookup_table()
        bh.edges.ensure_lookup_table()
        bh.faces.ensure_lookup_table()
    nuevas = aristas_nuevas()
    lazos_h = G.lazo_ordenado([(e.verts[0].index, e.verts[1].index) for e in nuevas])
    print("hueco en la cabeza: %d aristas nuevas de borde, %d lazos, los mayores %s" % (len(nuevas), len(lazos_h), sorted(len(l) for l in lazos_h)[-6:]))
    if "--diag" in a:
        mid = np.array([(np.array(e.verts[0].co[:]) + np.array(e.verts[1].co[:])) / 2 for e in nuevas])
        dp = dist_poligono_np(R[:, [0, 2]], mid[:, [0, 2]])
        dc = np.array([(lambda r: r[3] if r[0] is not None else 1.0)(bvh_c.find_nearest(Vector(p), 1.0)) for p in mid])
        print("DIAG aristas nuevas: dist al poligono (mm) pct", np.round(np.percentile(dp * 1000, [5, 25, 50, 75, 95]), 1),
              "dist a la cara (mm) pct", np.round(np.percentile(dc * 1000, [5, 25, 50, 75, 95]), 1))
        print("DIAG fuera del poligono >5mm:", int((dp < -0.005).sum()), "dentro >5mm:", int((dp > 0.005).sum()), "lejos de cara >15mm:", int((dc > 0.015).sum()))
        print("DIAG v-y de aristas lejos de cara", np.round(np.percentile(mid[dc > 0.015][:, 1] * 1000, [5, 50, 95]), 0) if (dc > 0.015).any() else None)
    if "--diag" in a:
        for l in sorted(lazos_h, key=len)[::-1][:14]:
            Pl = np.array([bh.verts[i].co[:] for i in l])
            print("DIAG lazo n=%d centro(mm) %s extension(mm) %s" % (len(l), np.round(Pl.mean(0) * 1000, 0), np.round((Pl.max(0) - Pl.min(0)) * 1000, 0)))
    if "--diag" in a:
        deg = {}
        for e in nuevas:
            for v in e.verts:
                deg[v.index] = deg.get(v.index, 0) + 1
        print("DIAG grados de vertice de borde nuevo:", {k: list(deg.values()).count(k) for k in sorted(set(deg.values()))})
        pin = [i for i, c in deg.items() if c > 2]
        for i in pin[:6]:
            v = bh.verts[i]
            print("DIAG pellizco", np.round(np.array(v.co[:]) * 1000, 1), "caras", len(v.link_faces), "aristas", len(v.link_edges),
                  "nomanif", sum(1 for e in v.link_edges if len(e.link_faces) > 2))
    if "--diag-img" in a and "--diag-antes" in a:
        debug_render(bh, nuevas, arg("--diag-img"), R)

    # ---- lazo principal del hueco; los lazos pequenos (cortes de pelos de barba en el borde) se tapan -----------------------------
    principal = max(lazos_h, key=len)
    pequenos = [l for l in lazos_h if l is not principal]
    n_tapados = 0
    for l in pequenos:                                      # tapa cada lazo pequeno con un abanico de triangulos (orientacion coherente con el vecino)
        bh.verts.ensure_lookup_table()
        vs = [bh.verts[k] for k in l]
        c = bh.verts.new(np.mean([np.array(v.co[:]) for v in vs], axis=0).tolist())
        for q in range(len(vs)):
            va, vb = vs[q], vs[(q + 1) % len(vs)]
            e = bh.edges.get((va, vb))
            if e is None or not e.is_boundary:
                continue
            lp = e.link_loops[0]
            orden = [vb, va, c] if (lp.vert == va and lp.link_loop_next.vert == vb) else [va, vb, c]
            try:
                bh.faces.new(orden)
                n_tapados += 1
            except ValueError:
                pass
    if pequenos:
        bh.verts.index_update()
        bh.verts.ensure_lookup_table()
        bh.edges.ensure_lookup_table()
        bh.faces.ensure_lookup_table()
        nuevas = aristas_nuevas()
        lazos_h = G.lazo_ordenado([(e.verts[0].index, e.verts[1].index) for e in nuevas])
        principal = max(lazos_h, key=len)
    if "--diag" in a:
        for l in sorted(lazos_h, key=len)[::-1][:6]:
            Pl = np.array([bh.verts[i].co[:] for i in l])
            print("DIAG tras tapar lazo n=%d centro(mm) %s ext(mm) %s ciclo_cerrado=%s" % (len(l), np.round(Pl.mean(0) * 1000, 0), np.round((Pl.max(0) - Pl.min(0)) * 1000, 0),
                  any(e.other_vert(bh.verts[l[0]]).index == l[-1] for e in bh.verts[l[0]].link_edges)))
    print("lazos pequenos tapados: %d (%d caras); lazos de hueco restantes: %s" % (len(pequenos), n_tapados, sorted(len(l) for l in lazos_h)[-5:]))
    H_ids = list(principal)
    P_H = np.array([bh.verts[i].co[:] for i in H_ids])
    K = len(R)

    # ---- faldon de la cara: el contorno se prolonga SOBRE_MM hacia fuera y se hunde por detras de la superficie de la cabeza, de modo que el
    # borde (dentado) del hueco lo tapa desde delante y no queda ninguna rendija. La costura es de solape (no soldada vertice a vertice):
    # el borde del hueco de la cabeza es de varias capas y de pelos sueltos en la barba y no admite una soldadura limpia -------------------
    SOBRE = float(arg("--faldon-mm", "5.0")) / 1000.0
    HUND = float(arg("--faldon-hund-mm", "2.0")) / 1000.0
    bh.verts.index_update()
    bvh_h = BVHTree.FromBMesh(bh)
    ctr_c = Vc[l_ext].mean(axis=0)
    ext = np.zeros_like(R)
    for k in range(K):
        t = R[(k + 1) % K] - R[k - 1]
        n = np.array([t[2], 0.0, -t[0]])                       # normal en el plano XZ de la tangente
        n /= max(np.linalg.norm(n), 1e-12)
        if np.dot(n, R[k] - ctr_c) < 0:
            n = -n
        ext[k] = R[k] + SOBRE * n
    ext = np.array(G.lazo_taubin(ext.tolist(), 6))
    for k in range(K):
        loc, nor, ix, dist = bvh_h.ray_cast(Vector((ext[k][0], ext[k][1] - 0.5, ext[k][2])), Vector((0, 1, 0)))
        ext[k][1] = (loc[1] + HUND) if loc is not None else R[k][1] + HUND
    bc.verts.ensure_lookup_table()
    rim_v = [bc.verts[i] for i in l_ext]
    vf = [bc.verts.new(p.tolist()) for p in ext]
    for k in range(K):
        bc.faces.new([rim_v[k], rim_v[(k + 1) % K], vf[(k + 1) % K], vf[k]])
    bc.verts.index_update()
    bc.verts.ensure_lookup_table()
    bc.faces.ensure_lookup_table()
    orientar_hacia_fuera(bc)
    print("faldon de la cara: %d caras, %.1f mm hacia fuera, hundido %.1f mm tras la cabeza" % (K, SOBRE * 1000, HUND * 1000))

    # ---- UV de la cara (isla unica en 0..1) y material cara_tex ------------------------------------------------------------------
    metodo_uv, UVc, tabla_uv = elegir_uv(bc)
    TEX = int(arg("--tex", "2048"))
    img = bpy.data.images.new("cara_tex", TEX, TEX, alpha=False)
    img.pixels.foreach_set(np.tile(np.array([piel[0], piel[1], piel[2], 1.0], dtype=np.float32), TEX * TEX))
    img.pack()
    mat_cara = bpy.data.materials.new("cara_tex")
    mat_cara.use_nodes = True
    nt = mat_cara.node_tree
    tx = nt.nodes.new("ShaderNodeTexImage")
    tx.image = img
    nt.links.new(tx.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    nt.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.6

    # ---- unir a la cabeza ----------------------------------------------------------------------------------------------------------
    uvl = bh.loops.layers.uv.active
    cmap = {v.index: bh.verts.new(v.co) for v in bc.verts}
    n_cara = 0
    for f in bc.faces:
        nf = bh.faces.new([cmap[v.index] for v in f.verts])
        nf.material_index = 1
        for lp, v in zip(nf.loops, f.verts):
            lp[uvl].uv = tuple(UVc[v.index])
        n_cara += 1
    for f in bh.faces:
        f.smooth = True
    bh.verts.index_update()
    bh.edges.index_update()
    bh.faces.index_update()
    bh.verts.ensure_lookup_table()
    bh.edges.ensure_lookup_table()
    # aristas abiertas: las originales de la cabeza (tag), las del anillo de los ojos (esperadas) y el resto
    ids_ojos = {cmap[i].index for l in anillos_ojos for i in l}
    abiertas = [e for e in bh.edges if e.is_boundary]
    n_orig = sum(1 for e in abiertas if e.tag)
    n_ojos = sum(1 for e in abiertas if not e.tag and e.verts[0].index in ids_ojos and e.verts[1].index in ids_ojos)
    n_otras = len(abiertas) - n_orig - n_ojos
    n_nomanif = sum(1 for e in bh.edges if len(e.link_faces) > 2)
    print("aristas de borde en la malla unida: %d (originales de la cabeza %d, aberturas de los ojos %d esperadas, otras %d); no-manifold %d (antes %d)"
          % (len(abiertas), n_orig, n_ojos, n_otras, n_nomanif, n_nomanif_orig))
    me_h = ob_h.data
    bh.to_mesh(me_h)
    me_h.update()
    me_h.materials.append(mat_cara)
    bh.free()
    bpy.data.objects.remove(ob_cara)
    ob_cara = None
    for o in bpy.data.objects:
        if o.type == "MESH" and o.name not in ("cabeza", "ojo_izq", "ojo_dch"):
            bpy.data.objects.remove(o)

    # ---- hoja, blend, glb, json --------------------------------------------------------------------------------------------------
    info = json.load(open(os.path.join(DIR, "cara_limpia.json")))
    ancho_g = ancho
    ctr = np.mean([m3d[i] for i in (234, 454, 152, 10) if i in m3d], axis=0)
    if "--sin-hoja" not in a:
        import cara_hoja
        cara_hoja.hoja_unida(bpy.context.scene, ob_h, ojos, ctr, ancho_g, os.path.join(DIR, "hoja_unida.jpg"),
                             [np.array(lm["ajuste"]["ojos"][k]["centro"]) for k in ("izq", "dch")])
    bpy.ops.object.select_all(action="DESELECT")
    for o in [ob_h] + ojos:
        o.select_set(True)
    bpy.context.view_layer.objects.active = ob_h
    bpy.ops.export_scene.gltf(filepath=os.path.join(DIR, "cabeza_unida.glb"), export_format="GLB", use_selection=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(DIR, "cabeza_unida.blend"), compress=True)
    unir = dict(cara_limpia=info, uv=dict(metodo=metodo_uv, tabla=tabla_uv, textura=TEX), piel_rgb_imagen=[float(x) for x in piel],
                faldon_mm=SOBRE * 1000, hueco_aristas=len(nuevas), lazos_pequenos_tapados=len(pequenos),
                aristas_borde=dict(total=len(abiertas), originales_cabeza=n_orig, ojos=n_ojos, otras=n_otras),
                no_manifold=dict(antes=n_nomanif_orig, despues=n_nomanif), caras_cara=n_cara)
    lm["unir"] = unir
    json.dump(lm, open(os.path.join(DIR, "landmarks.json"), "w"), indent=1, default=float)
    print("UNIDA", json.dumps(dict(uv=metodo_uv, aristas_borde_otras=n_otras, ojos=info["aperturas"]), default=float))


if FASE == "cara":
    fase_cara()
else:
    fase_unir()
