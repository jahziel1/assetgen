"""cara_base.py -- ajusta la cabeza base MakeHuman (hm08, CC0) a una cabeza generada (imagen->3D) y coloca ojos reales.

    blender -b -P cara_base.py -- --glb cabeza.glb --salida DIR
        --lm-gen lm_gen.json --render-gen render_gen.json --lm-base lm_base.json --render-base render_base.json
        [--iris-rgb r,g,b | --retrato-lm lm_retrato.json] [--cache DIR] [--suavizado 8] [--ojo-prof-mm 0] [--sin-hoja]

Los *.json 2D salen de cara_landmarks2d.py (MediaPipe) sobre los renders de cara_render.py (misma camara ortografica de frente, para la
cabeza generada y para la base). Pasos: (1) cada landmark 2D se lleva a 3D con un rayo ortografico sobre la malla; (2) similitud de
Umeyama (robusta) base->generada; (3) deformacion RBF thin-plate con los landmarks; (4) los vertices de la CARA (poligono del ovalo
facial, con caida suave) se proyectan al punto mas cercano de la superficie generada, con el desplazamiento suavizado (Laplaciano) para
conservar el detalle fino de la base (labios, fosas nasales); los parpados y las cuencas (zona elipsoidal alrededor de cada ojo) NO se
proyectan; (5) los globos oculares MakeHuman se colocan en las cuencas (centro RBF, escala por la distancia entre comisuras) y el iris
de su textura se tinta con el color del retrato; los parpados que quedarian dentro del globo se empujan hacia fuera.
Escribe en DIR: cara_ajuste.blend, cara_base.glb (cara + ojos), landmarks.json, ojo_iris.png y hoja_cara.jpg. Imprime 'CARA {resumen}'.
Unidades de salida: metros, Z arriba, el frente en -Y (glTF importado en Blender)."""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "..", "cli"))
import cara_comun as C  # noqa: E402
from assetgen_cli import cara_geom as G  # noqa: E402
from assets_makehuman import asegurar  # noqa: E402

a = sys.argv[sys.argv.index("--") + 1:]


def arg(n, d=None):
    return a[a.index(n) + 1] if n in a else d


GLB, SALIDA = arg("--glb"), arg("--salida")
if not GLB or not SALIDA:
    sys.exit("uso: --glb cabeza.glb --salida DIR --lm-gen --render-gen --lm-base --render-base [--iris-rgb|--retrato-lm]")
os.makedirs(SALIDA, exist_ok=True)
N_SUAV = int(arg("--suavizado", "8"))
PROF_OJO = float(arg("--ojo-prof-mm", "0")) / 1000.0
LAM = float(arg("--rbf-lambda", "0.002"))

# ---- indices MediaPipe (478 puntos; "der/izq" del sujeto) -------------------------------------------------------------------------
OJO_ESQ = dict(ext=33, int=133, iris=468, sup=159, inf=145)       # ojo a la izquierda de la imagen (derecho del sujeto)
OJO_DCH = dict(ext=263, int=362, iris=473, sup=386, inf=374)
OVALO = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172,
         58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
AJUSTE = [33, 133, 263, 362, 468, 473,                       # comisuras y centro de iris
          107, 336, 105, 334,                                # cejas (interior y centro)
          6, 168, 195, 5, 1, 2, 98, 327, 129, 358, 49, 279,  # nariz: puente, punta, base, aletas
          61, 291, 0, 17, 13, 14, 37, 267, 84, 314,          # boca
          152, 148, 377, 175, 172, 397, 136, 365, 58, 288,   # menton y mandibula
          50, 280, 205, 425, 116, 345,                       # pomulos
          10, 151, 9, 8]                                     # frente
NOMBRES = {33: "ojo_der_comisura_ext", 133: "ojo_der_comisura_int", 263: "ojo_izq_comisura_ext", 362: "ojo_izq_comisura_int",
           468: "ojo_der_iris", 473: "ojo_izq_iris", 1: "nariz_punta", 2: "nariz_base", 6: "nariz_puente", 168: "nariz_entrecejo",
           98: "nariz_fosa_der", 327: "nariz_fosa_izq", 129: "nariz_ala_der", 358: "nariz_ala_izq", 61: "boca_comisura_der",
           291: "boca_comisura_izq", 0: "labio_sup", 17: "labio_inf", 152: "menton", 10: "frente_alta", 234: "lateral_der",
           454: "lateral_izq", 105: "ceja_der", 334: "ceja_izq"}


def cargar(p):
    return json.load(open(p))


lm_g, rg = cargar(arg("--lm-gen")), cargar(arg("--render-gen"))
lm_b, rb = cargar(arg("--lm-base")), cargar(arg("--render-base"))
if arg("--iris-rgb"):
    IRIS = [int(x) for x in arg("--iris-rgb").split(",")]
elif arg("--retrato-lm"):
    IRIS = cargar(arg("--retrato-lm"))["iris_rgb"]
else:
    IRIS = None

bpy.ops.wm.read_factory_settings(use_empty=True)

# ---- malla generada ---------------------------------------------------------------------------------------------------------------
bpy.ops.import_scene.gltf(filepath=GLB)
mallas = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if len(mallas) > 1:
    for o in mallas:
        o.select_set(True)
    bpy.context.view_layer.objects.active = mallas[0]
    bpy.ops.object.join()
ob_gen = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
ob_gen.name = "cabeza_generada"
dg = bpy.context.evaluated_depsgraph_get()
bvh_gen = BVHTree.FromObject(ob_gen, dg)

# ---- base MakeHuman + ojos (en metros, Blender) ---------------------------------------------------------------------------------------
rutas = asegurar(arg("--cache"))
base = C.leer_obj(rutas["base.obj"])
Vb_mh = base["v"]
Vb = C.a_blender(Vb_mh)
caras_cuerpo = [c for c, g in zip(base["caras"], base["grupo"]) if g == "body"]
ojos_obj = C.leer_obj(rutas["eyes_high-poly.obj"])
Ve = C.a_blender(C.ojos_sobre_base(C.leer_mhclo(rutas["eyes_high-poly.mhclo"]), Vb_mh))
caras_ojos = ojos_obj["caras"]
nb = len(Vb)
bvh_base = BVHTree.FromPolygons([Vector(v) for v in np.vstack([Vb, Ve]).tolist()],
                                [tuple(c) for c in caras_cuerpo] + [tuple(i + nb for i in c) for c in caras_ojos])


def a_rayo(lm, rend, i):
    x, y = lm["puntos"][i][0], lm["puntos"][i][1]
    c, s, res = rend["centro"], rend["escala"], lm["w"]
    return Vector((c[0] + (x / res - 0.5) * s, c[1] - 3.0, c[2] - (y / lm["h"] - 0.5) * s))


def elevar(lm, rend, bvh, i):
    o = a_rayo(lm, rend, i)
    loc, nor, idx, dist = bvh.ray_cast(o, Vector((0, 1, 0)))
    return (np.array(loc), idx) if loc is not None else (None, None)


idx_todos = sorted(set(AJUSTE) | set(OVALO) | {OJO_ESQ[k] for k in OJO_ESQ} | {OJO_DCH[k] for k in OJO_DCH})
P3g, P3b, HITb = {}, {}, {}
for i in idx_todos:
    pg, _ = elevar(lm_g, rg, bvh_gen, i)
    pb, ib = elevar(lm_b, rb, bvh_base, i)
    if pg is not None and pb is not None:
        P3g[i], P3b[i], HITb[i] = pg, pb, ib
print("landmarks elevados a 3D:", len(P3g), "de", len(idx_todos))

# ---- similitud robusta + RBF ----------------------------------------------------------------------------------------------------------
ids_aj = [i for i in AJUSTE if i in P3g]
src = [P3b[i].tolist() for i in ids_aj]
dst = [P3g[i].tolist() for i in ids_aj]
sim, usados = G.umeyama_robusta(src, dst)
ids_ok = [ids_aj[k] for k in usados]
descartados = [i for i in ids_aj if i not in ids_ok]
print("similitud: escala %.4f, usados %d/%d, descartados %s" % (sim[0], len(ids_ok), len(ids_aj), descartados))
_rs = {i: 1000 * float(np.linalg.norm(np.array(G.aplicar_similitud(sim, [P3b[i].tolist()])[0]) - P3g[i])) for i in ids_aj}
print("residuo tras similitud (mm):", {k: round(v, 1) for k, v in sorted(_rs.items(), key=lambda kv: -kv[1])[:20]})
R_sim = np.array(sim[1])
sim_pts = lambda P: np.array(G.aplicar_similitud(sim, np.asarray(P).tolist()))
_loo = {}
for _i in ids_aj:                                            # leave-one-out: un landmark que las demas no predicen (pelo delante) se descarta
    _m = G.rbf_ajustar([sim_pts([P3b[k]])[0].tolist() for k in ids_aj if k != _i], [P3g[k].tolist() for k in ids_aj if k != _i], LAM)
    _loo[_i] = 1000 * float(np.linalg.norm(np.array(G.rbf_evaluar(_m, sim_pts([P3b[_i]]).tolist())[0]) - P3g[_i]))
_umbral = max(30.0, 4 * float(np.median(list(_loo.values()))))
ids_rbf = [i for i in ids_aj if _loo[i] <= _umbral]
desc_rbf = [i for i in ids_aj if i not in ids_rbf]
print("leave-one-out RBF (mm):", {k: round(v, 1) for k, v in sorted(_loo.items(), key=lambda kv: -kv[1])[:8]}, "descartados RBF:", desc_rbf)
rbf = G.rbf_ajustar([sim_pts([P3b[i]])[0].tolist() for i in ids_rbf], [P3g[i].tolist() for i in ids_rbf], LAM)
mapa = lambda P: np.array(G.rbf_evaluar(rbf, sim_pts(P).tolist()))
res_rbf = [np.linalg.norm(mapa([P3b[i]])[0] - P3g[i]) for i in ids_rbf]
print("residuo RBF en landmarks: media %.2f mm, max %.2f mm" % (1000 * np.mean(res_rbf), 1000 * np.max(res_rbf)))

# ---- region de la cara en la base ---------------------------------------------------------------------------------------------------
ovalo = [(P3b[i][0], P3b[i][2]) for i in OVALO if i in P3b]
ancho_b = abs(P3b[454][0] - P3b[234][0])
ancho_g = abs(P3g[454][0] - P3g[234][0])
y_ref = max(P3b[234][1], P3b[454][1]) + 0.012           # no mas atras que los laterales de la cara
M_FUERA, M_DENTRO = 0.10 * ancho_b, 0.07 * ancho_b
Vcu = Vb[:13380]
dpol = np.array([G.distancia_poligono(ovalo, (p[0], p[2])) for p in Vcu])
# normales de vertice de la base (para descartar el reverso)
Fq = np.array(caras_cuerpo)
nf = np.cross(Vb[Fq[:, 1]] - Vb[Fq[:, 0]], Vb[Fq[:, 2]] - Vb[Fq[:, 0]])
nv = np.zeros_like(Vb)
for k in range(4):
    np.add.at(nv, Fq[:, k], nf)
nv /= np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-12)
en_parche = (dpol >= -(M_FUERA + 0.01 * ancho_b)) & (Vcu[:, 1] <= y_ref) & (nv[:13380, 1] < 0.35)
# solo la componente conexa mayor (sin restos sueltos, p. ej. un jiron del cuello)
_pad = list(range(len(Vcu)))


def _raiz(x):
    while _pad[x] != x:
        _pad[x] = _pad[_pad[x]]
        x = _pad[x]
    return x


for _f in caras_cuerpo:
    if all(_i < 13380 and en_parche[_i] for _i in _f):
        for _i in _f[1:]:
            _pad[_raiz(_i)] = _raiz(_f[0])
_tam = {}
for _i in np.where(en_parche)[0]:
    _tam[_raiz(_i)] = _tam.get(_raiz(_i), 0) + 1
_mayor = max(_tam, key=_tam.get)
en_parche = np.array([bool(en_parche[_i]) and _raiz(_i) == _mayor for _i in range(len(Vcu))])
idx_p = np.where(en_parche)[0]
remap = -np.ones(len(Vcu), dtype=int)
remap[idx_p] = np.arange(len(idx_p))
caras_p = [[int(remap[i]) for i in f] for f in caras_cuerpo if all(i < 13380 and en_parche[i] for i in f)]
usados_p = sorted({i for f in caras_p for i in f})
print("parche cara: %d vertices, %d caras" % (len(idx_p), len(caras_p)))
w_cara = np.array([G.suavizado((d + M_FUERA) / (M_FUERA + M_DENTRO)) for d in dpol[idx_p]])

P1 = mapa(Vb[idx_p])                                      # base deformada por similitud + RBF

# ---- ojos y parpados: los globos MakeHuman van a la cuenca y los parpados/cuenca viajan CON el globo -------------------------------
# Los vertices de cara dentro de la zona elipsoidal de cada ojo usan la misma transformacion local (similitud) que el globo, asi la relacion
# globo-parpado de la base (parpados abiertos con forma real, sin atravesar) se conserva; fuera de la zona mandan la RBF y la proyeccion.
n_ojo = len(Ve)
cen_faces = np.array([Ve[f].mean(axis=0) for f in caras_ojos])
lado_cara = cen_faces[:, 0] < 0                              # True: ojo del lado -X (izquierda de la imagen)
mascara_v = np.zeros(n_ojo, dtype=int)
for f, izq in zip(caras_ojos, lado_cara):
    mascara_v[f] = 1 if izq else 2
Ve_nuevo = Ve.copy()
info_ojos = {}
Vp_sim = sim_pts(Vb[idx_p])
excl = np.zeros(len(idx_p))
for lado, etiqueta, d_ in ((1, "izq", OJO_ESQ), (2, "dch", OJO_DCH)):
    sel = np.where(mascara_v == lado)[0]
    c_b = Ve[sel].mean(axis=0)
    r_b = np.linalg.norm(Ve[sel] - c_b, axis=1).mean()
    ge, gi, be, bi = P3g[d_["ext"]], P3g[d_["int"]], P3b[d_["ext"]], P3b[d_["int"]]
    W = float(np.linalg.norm(ge - gi))
    f = W / float(np.linalg.norm(sim_pts([be])[0] - sim_pts([bi])[0]))      # factor local respecto a la similitud global
    c_nuevo = mapa([c_b])[0] + np.array([0, PROF_OJO, 0])
    c_sim = sim_pts([c_b])[0]
    Ve_nuevo[sel] = c_nuevo + f * (sim_pts(Ve[sel]) - c_sim)
    info_ojos[etiqueta] = dict(sel=sel, centro=c_nuevo, radio=r_b * sim[0] * f, factor=f, ancho=W)
    # zona de exclusion: elipse alrededor del punto medio de las comisuras (deformado), en el plano frontal
    cm = mapa([(be + bi) / 2])[0]
    de = np.sqrt(((P1[:, 0] - cm[0]) / (0.80 * W)) ** 2 + ((P1[:, 2] - cm[2]) / (0.46 * W)) ** 2)
    e = 1.0 - np.array([G.suavizado((x - 1.0) / 0.5) for x in de])
    local = c_nuevo + f * (Vp_sim - c_sim)
    P1 = P1 + e[:, None] * (local - P1)
    excl = np.maximum(excl, e)

# ---- proyeccion al punto mas cercano de la generada, desplazamiento suavizado ---------------------------------------------------------
DMAX = 0.09 * ancho_g
desp = np.zeros_like(P1)
ok = np.zeros(len(P1))
nv_map = nv[idx_p] @ R_sim.T
for k, p in enumerate(P1):
    loc, nor, ix, dist = bvh_gen.find_nearest(Vector(p), DMAX)
    if loc is None:
        continue
    if float(np.dot(np.array(nor), nv_map[k])) < 0.35:      # superficie de otro lado (pelo, capucha): no proyectar
        continue
    desp[k] = np.array(loc) - p
    ok[k] = 1.0
# adyacencia en el parche
ea = []
for f in caras_p:
    for k in range(len(f)):
        ea.append((f[k], f[(k + 1) % len(f)]))
ea = np.array(ea)
ei = np.concatenate([ea[:, 0], ea[:, 1]])
ej = np.concatenate([ea[:, 1], ea[:, 0]])
grado = np.bincount(ei, minlength=len(P1)).astype(float)


def vecinos_medio(x):
    s = np.zeros_like(x)
    np.add.at(s, ei, x[ej])
    return s / np.maximum(grado, 1)[:, None] if x.ndim == 2 else s / np.maximum(grado, 1)


dn, mn = desp * ok[:, None], ok.copy()
for _ in range(N_SUAV):                                     # convolucion normalizada: el relleno viene de los vecinos validos
    dn = 0.5 * dn + 0.5 * vecinos_medio(dn)
    mn = 0.5 * mn + 0.5 * vecinos_medio(mn)
desp_s = np.where(mn[:, None] > 1e-3, dn / np.maximum(mn, 1e-3)[:, None], 0.0)
peso = w_cara * (1.0 - excl) * np.clip(mn * 1.5, 0, 1)
P2 = P1 + desp_s * peso[:, None]


# parpados que atravesarian el globo: empujar hacia fuera (solo la cara anterior)
empujados = 0
for etiqueta, d_ in info_ojos.items():
    c, r = d_["centro"], d_["radio"]
    v = P2 - c
    dist = np.linalg.norm(v, axis=1)
    dentro = (dist < 1.03 * r) & (v[:, 1] < 0.2 * r)
    P2[dentro] = c + v[dentro] / dist[dentro, None] * (1.03 * r)
    empujados += int(dentro.sum())
print("vertices empujados fuera de los globos:", empujados)

# ---- error de landmarks tras el ajuste ------------------------------------------------------------------------------------------------
def pos_final(i):
    """Posicion final del landmark de base i: baricentrica sobre la cara golpeada, con los vertices ya ajustados."""
    ix = HITb[i]
    f = list(caras_cuerpo[ix]) if ix < len(caras_cuerpo) else [j + nb for j in caras_ojos[ix - len(caras_cuerpo)]]
    if ix < len(caras_cuerpo):
        if not all(j < 13380 and en_parche[j] for j in f):
            return None
        pos = [P2[remap[j]] for j in f]
        orig = [Vb[j] for j in f]
    else:
        fl = caras_ojos[ix - len(caras_cuerpo)]
        pos = [Ve_nuevo[j] for j in fl]
        orig = [Ve[j] for j in fl]
    p = P3b[i]
    for tri in ((0, 1, 2), (0, 2, 3)) if len(orig) == 4 else ((0, 1, 2),):
        A, B, Cc = orig[tri[0]], orig[tri[1]], orig[tri[2]]
        v0, v1, v2 = B - A, Cc - A, p - A
        d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
        den = d00 * d11 - d01 * d01
        if abs(den) < 1e-18:
            continue
        v_ = (d11 * d20 - d01 * d21) / den
        w_ = (d00 * d21 - d01 * d20) / den
        u_ = 1 - v_ - w_
        if min(u_, v_, w_) >= -0.15:
            return u_ * pos[tri[0]] + v_ * pos[tri[1]] + w_ * pos[tri[2]]
    return None


err = {}
for i in ids_rbf:
    pf = pos_final(i)
    if pf is not None:
        err[i] = float(np.linalg.norm(pf - P3g[i]))
if err:
    ev = np.array(list(err.values()))
    print("error de landmarks (mm): media %.2f, mediana %.2f, p90 %.2f, max %.2f; %.2f%% del ancho de cara (media) n=%d"
          % (1000 * ev.mean(), 1000 * np.median(ev), 1000 * np.percentile(ev, 90), 1000 * ev.max(), 100 * ev.mean() / ancho_g, len(ev)))
# distancia de la cara a la superficie generada (fuera de la exclusion de ojos)
dsf = []
for k, p in enumerate(P2):
    if peso[k] > 0.9:
        loc, _, _, d = bvh_gen.find_nearest(Vector(p), 0.05)
        if loc is not None:
            dsf.append(d)
dsf = np.array(dsf)
print("distancia cara->superficie generada (zona proyectada): media %.2f mm, p90 %.2f mm, n=%d"
      % (1000 * dsf.mean(), 1000 * np.percentile(dsf, 90), len(dsf)) if len(dsf) else "sin muestras")

# ---- objetos Blender -----------------------------------------------------------------------------------------------------------------
sc = bpy.context.scene
# material de piel neutro
mat_piel = bpy.data.materials.new("piel_base")
mat_piel.diffuse_color = (0.72, 0.50, 0.40, 1)
mat_piel.use_nodes = True
mat_piel.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.72, 0.50, 0.40, 1)
mat_piel.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.6

me = bpy.data.meshes.new("cara")
me.from_pydata(P2.tolist(), [], [tuple(f) for f in caras_p])
me.update()
for p_ in me.polygons:
    p_.use_smooth = True
me.materials.append(mat_piel)
atr = me.attributes.new("peso_proyeccion", "FLOAT", "POINT")
atr.data.foreach_set("value", peso.astype(np.float32))
atr2 = me.attributes.new("exclusion_ojo", "FLOAT", "POINT")
atr2.data.foreach_set("value", excl.astype(np.float32))
ob_cara = bpy.data.objects.new("cara", me)
sc.collection.objects.link(ob_cara)

# textura de ojo tintada con el color del iris del retrato
img_ojo = bpy.data.images.load(rutas["eyes_brown_eye.png"])
ancho_t, alto_t = img_ojo.size
px = np.array(img_ojo.pixels[:], dtype=np.float32).reshape(alto_t, ancho_t, 4)
if IRIS:
    rgb8 = (np.clip(px[..., :3], 0, 1) * 255 + 0.5).astype(np.int32)
    sat = (rgb8.max(axis=2) - rgb8.min(axis=2)) / np.maximum(rgb8.max(axis=2), 1)
    m_iris = (sat >= 0.30) & (px[..., 3] > 0.5)
    ys, xs = np.where(m_iris)
    ref = G.estadistica_iris([tuple(int(c) for c in rgb8[y, x]) for y, x in zip(ys, xs)])
    # los pixeles del png vienen en sRGB (Blender los guarda lineales en .pixels): convertir ida y vuelta con gamma 2.2 no es necesario
    # porque se opera en el mismo espacio en que se lee el retrato (sRGB de 8 bits): usamos los valores de 8 bits tal cual.
    nuevo = px.copy()
    for y, x in zip(ys, xs):
        t = G.tintar_pixel(tuple(int(c) for c in rgb8[y, x]), ref, IRIS)
        nuevo[y, x, :3] = np.array(t, dtype=np.float32) / 255.0
    px = nuevo
    print("iris: referencia hsv %.2f %.2f %.2f -> objetivo rgb %s (%d pixeles tintados)" % (ref + (tuple(IRIS), len(ys))))
img_t = bpy.data.images.new("ojo_iris", ancho_t, alto_t, alpha=True)
img_t.colorspace_settings.name = "sRGB"
img_t.pixels.foreach_set(px.reshape(-1))
img_t.filepath_raw = os.path.join(SALIDA, "ojo_iris.png")
img_t.file_format = "PNG"
img_t.save()
mat_ojo = bpy.data.materials.new("ojo")
mat_ojo.use_nodes = True
nt = mat_ojo.node_tree
tx = nt.nodes.new("ShaderNodeTexImage")
tx.image = img_t
nt.links.new(tx.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
nt.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.15

objs_ojo = {}
for etiqueta, d_ in info_ojos.items():
    sel = d_["sel"]
    loc = {int(v): k for k, v in enumerate(sel)}
    fs, uvf = [], []
    for f, cv in zip(caras_ojos, ojos_obj["caras_vt"]):
        if f[0] in loc:
            fs.append(tuple(loc[i] for i in f))
            uvf.append(cv)
    m2 = bpy.data.meshes.new("ojo_" + etiqueta)
    m2.from_pydata(Ve_nuevo[sel].tolist(), [], fs)
    m2.update()
    capa = m2.uv_layers.new(name="UVMap")
    li = 0
    for cv in uvf:
        for k in cv:
            capa.data[li].uv = tuple(ojos_obj["vt"][k])
            li += 1
    for p_ in m2.polygons:
        p_.use_smooth = True
    m2.materials.append(mat_ojo)
    o2 = bpy.data.objects.new("ojo_" + etiqueta, m2)
    sc.collection.objects.link(o2)
    objs_ojo[etiqueta] = o2

# ---- landmarks.json ------------------------------------------------------------------------------------------------------------------
lm_out = dict(
    unidades="metros, Z arriba, frente en -Y (glTF importado en Blender); 2D en pixeles del render ortografico de frente (ver render)",
    render=dict(generada=rg, base=rb),
    indices_mediapipe=idx_todos, nombres={str(k): v for k, v in NOMBRES.items()},
    generada=dict(px={str(i): lm_g["puntos"][i][:2] for i in idx_todos}, m3d={str(i): P3g[i].tolist() for i in P3g}),
    base=dict(px={str(i): lm_b["puntos"][i][:2] for i in idx_todos}, m3d={str(i): P3b[i].tolist() for i in P3b}),
    base_ajustada_m3d={str(i): (pos_final(i).tolist() if pos_final(i) is not None else None) for i in ids_ok},
    ajuste=dict(similitud_escala=sim[0], usados_similitud=ids_ok, usados_rbf=ids_rbf, descartados_similitud=descartados, descartados_rbf=desc_rbf, rbf_lambda=LAM, suavizado_iter=N_SUAV,
                residuo_rbf_mm_media=1000 * float(np.mean(res_rbf)),
                error_mm={str(k): 1000 * v for k, v in err.items()},
                error_mm_media=1000 * float(np.mean(list(err.values()))) if err else None,
                error_pct_ancho_cara=100 * float(np.mean(list(err.values()))) / ancho_g if err else None,
                ancho_cara_generada_m=ancho_g, iris_rgb=IRIS,
                ojos={k: dict(centro=v["centro"].tolist(), radio=float(v["radio"]), factor=float(v["factor"]))
                      for k, v in info_ojos.items()},
                dist_a_superficie_mm_media=1000 * float(dsf.mean()) if len(dsf) else None))
json.dump(lm_out, open(os.path.join(SALIDA, "landmarks.json"), "w"), indent=1, default=float)

# ---- exportar glb (cara + ojos) -------------------------------------------------------------------------------------------------------
bpy.ops.object.select_all(action="DESELECT")
for o in [ob_cara] + list(objs_ojo.values()):
    o.select_set(True)
bpy.context.view_layer.objects.active = ob_cara
bpy.ops.export_scene.gltf(filepath=os.path.join(SALIDA, "cara_base.glb"), export_format="GLB", use_selection=True)

# ---- hoja de contacto ------------------------------------------------------------------------------------------------------------------
if "--sin-hoja" not in a:
    sys.path.insert(0, AQUI)
    import cara_hoja
    ctr = np.mean([P3g[i] for i in ids_rbf], axis=0)
    cara_hoja.hoja(sc, ob_gen, ob_cara, list(objs_ojo.values()), ctr, ancho_g, os.path.join(SALIDA, "hoja_cara.jpg"),
                   [info_ojos["izq"]["centro"], info_ojos["dch"]["centro"]])

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(SALIDA, "cara_ajuste.blend"), compress=True)
print("CARA", json.dumps(dict(ancho_cara_m=ancho_g, error_mm_media=lm_out["ajuste"]["error_mm_media"],
                              error_pct=lm_out["ajuste"]["error_pct_ancho_cara"], parche_verts=int(len(idx_p)), descartados=descartados), default=float))
