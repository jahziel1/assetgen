"""cara_textura.py -- etapa 3, lado Blender (Blender -b -P ... -- ...). La parte de imagen (numpy/opencv, venv) es cara_textura_img.py.

    --fase extraer --dir DIR [--tex 2048] [--franja-mm 40]
        Abre DIR/cabeza_unida.blend y escribe en DIR/_trabajo: texeles.npz (por texel UV de la cara: posicion 3D, normal, distancia al
        borde exterior, UV de la cabeza en el punto mas cercano de la cabeza), cabeza_tex.png (textura de la cabeza), puntos3d.json
        (landmarks 3D de la cara unida, con los bordes de los parpados ordenados, para alinear el retrato).
    --fase aplicar --dir DIR --tex-png cara_tex.png
        Pone cara_tex.png como textura de la cara, da brillo humedo a los ojos y guarda DIR/cabeza_final.blend y .glb.
    --fase hoja --dir DIR --glb-antes cabeza.glb --retrato retrato.png [--salida hoja_final.jpg]
        Hoja de contacto con textura y luz suave (Eevee): retrato | antes | final de frente; 3/4, perfil y ojos de la final.

Convenciones: la cara es el material 1 de la malla `cabeza` (cara_unir.py); los ojos son `ojo_izq`/`ojo_dch` (no se tocan sus UV)."""
import json
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "..", "cli"))
from assetgen_cli import cara_imagen as I  # noqa: E402

a = sys.argv[sys.argv.index("--") + 1:]


def arg(n, d=None):
    return a[a.index(n) + 1] if n in a else d


FASE = arg("--fase")
DIR = arg("--dir")
if FASE not in ("extraer", "aplicar", "hoja") or not DIR:
    sys.exit("uso: --fase extraer|aplicar|hoja --dir DIR [...]")
W = os.path.join(DIR, "_trabajo")
os.makedirs(W, exist_ok=True)
MAT_CARA = 1


def triangulos(me):
    """-> (poly_idx (t,), verts (t,3), loops (t,3)) de los triangulos de la malla (calc_loop_triangles)."""
    me.calc_loop_triangles()
    t = len(me.loop_triangles)
    poly = np.empty(t, np.int32)
    me.loop_triangles.foreach_get("polygon_index", poly)
    verts = np.empty(t * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", verts)
    loops = np.empty(t * 3, np.int32)
    me.loop_triangles.foreach_get("loops", loops)
    return poly, verts.reshape(t, 3), loops.reshape(t, 3)


def ordenar_lazo_frontal(bm, ids_borde):
    """Borde delantero del parpado: la arista opuesta a la de borde en los quads de la pared. -> indices en orden."""
    ar = []
    for f in bm.faces:
        if len(f.verts) == 4:
            vs = [v.index for v in f.verts]
            for i in range(4):
                if vs[i] in ids_borde and vs[(i + 1) % 4] in ids_borde:
                    ar.append((vs[(i + 2) % 4], vs[(i + 3) % 4]))
    adj = {}
    for p, q in ar:
        adj.setdefault(p, []).append(q)
        adj.setdefault(q, []).append(p)
    cur = next(iter(adj))
    orden, prev = [cur], None
    while True:
        sig = [x for x in adj[cur] if x != prev]
        if not sig or sig[0] == orden[0]:
            break
        prev, cur = cur, sig[0]
        orden.append(cur)
    return orden


def extraer():
    TEX = int(arg("--tex", "2048"))
    FRANJA = float(arg("--franja-mm", "40")) / 1000.0
    lm = json.load(open(os.path.join(DIR, "landmarks.json")))
    bpy.ops.wm.open_mainfile(filepath=os.path.join(DIR, "cabeza_unida.blend"))
    ob = bpy.data.objects["cabeza"]
    me = ob.data
    nv = len(me.vertices)
    V = np.empty(nv * 3, np.float64)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    NV = np.empty(nv * 3, np.float64)
    me.vertex_normals.foreach_get("vector", NV)
    NV = NV.reshape(-1, 3)
    uv_l = np.empty(len(me.loops) * 2, np.float64)
    me.uv_layers.active.data.foreach_get("uv", uv_l)
    uv_l = uv_l.reshape(-1, 2)
    mat_p = np.empty(len(me.polygons), np.int32)
    me.polygons.foreach_get("material_index", mat_p)
    poly, tv, tl = triangulos(me)
    es_cara = mat_p[poly] == MAT_CARA
    print("triangulos: cara %d, cabeza %d" % (es_cara.sum(), (~es_cara).sum()))

    # ---- distancia de cada vertice de la cara a su borde exterior (sin las aberturas de los ojos) ---------------------------------
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    centros = [np.array(lm["ajuste"]["ojos"][k]["centro"]) for k in ("izq", "dch")]
    segs = []
    borde_ojo = {k: set() for k in ("izq", "dch")}
    for e in bm.edges:
        caras = [f for f in e.link_faces if f.material_index == MAT_CARA]
        if len(caras) != 1:                       # borde del parche de la cara (con una sola cara de la cara a un lado)
            continue
        p, q = np.array(e.verts[0].co[:]), np.array(e.verts[1].co[:])
        cerca = [k for k, c in zip(("izq", "dch"), centros) if np.hypot((p[0] + q[0]) / 2 - c[0], (p[2] + q[2]) / 2 - c[2]) < 0.045]
        if cerca:
            borde_ojo[cerca[0]].update((e.verts[0].index, e.verts[1].index))
        else:
            segs.append((p, q))
    muestras = []
    for p, q in segs:
        n = max(2, int(np.linalg.norm(q - p) / 0.001))
        muestras.append(p[None] + (q - p)[None] * np.linspace(0, 1, n)[:, None])
    muestras = np.vstack(muestras)
    print("borde exterior de la cara: %d aristas, %d muestras" % (len(segs), len(muestras)))
    cv = np.unique(tv[es_cara])
    dv = np.full(nv, 1e9)
    for i in range(0, len(cv), 400):
        idx = cv[i:i + 400]
        dv[idx] = np.sqrt(((V[idx][:, None, :] - muestras[None, :, :]) ** 2).sum(-1)).min(axis=1)

    # ---- rasterizar la cara en UV ----------------------------------------------------------------------------------------------
    tcara = np.nonzero(es_cara)[0]
    uvt = uv_l[tl[tcara]].reshape(-1, 2)                        # (3T, 2)
    # la UV es casi frontal: donde los triangulos se solapan (parpados que giran, pliegues) gana el mas cercano a la camara (el de menor Y)
    prio = -V[tv[tcara]][:, :, 1].mean(axis=1)
    tri_idx, bar = I.rasterizar_triangulos(uvt, np.arange(len(tcara) * 3).reshape(-1, 3), TEX, prioridad=prio)
    cubierto = tri_idx >= 0
    tt = np.where(cubierto, tri_idx, 0)
    vv = tv[tcara][tt]                                          # (n, n, 3) indices de vertice
    b = bar[..., None]
    P = (V[vv] * b).sum(axis=2)
    Nn = (NV[vv] * b).sum(axis=2)
    Nn /= np.maximum(np.linalg.norm(Nn, axis=-1, keepdims=True), 1e-9)
    D = (dv[vv] * bar).sum(axis=2)
    P[~cubierto] = 0
    D[~cubierto] = 1e3
    print("texeles cubiertos: %d de %d (%.1f %%)" % (cubierto.sum(), TEX * TEX, 100.0 * cubierto.mean()))

    # ---- UV de la cabeza en el punto mas cercano (solo en la franja cerca del borde) -----------------------------------------------
    th = np.nonzero(~es_cara)[0]
    bvh = BVHTree.FromPolygons([tuple(p) for p in V.tolist()], [tuple(t) for t in tv[th].tolist()])
    sel = np.argwhere(cubierto & (D < FRANJA))
    uvh = np.zeros((TEX, TEX, 2), np.float32)
    dh = np.full((TEX, TEX), 1e3, np.float32)
    locs = np.zeros((len(sel), 3))
    tris_h = np.zeros(len(sel), np.int64)
    for k, (j, i) in enumerate(sel):
        loc, nor, ix, d = bvh.find_nearest(Vector(P[j, i].tolist()))
        locs[k] = loc
        tris_h[k] = ix
        dh[j, i] = d
    A_, B_, C_ = (V[tv[th][tris_h][:, c]] for c in range(3))
    v0, v1, v2 = B_ - A_, C_ - A_, locs - A_
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-18)
    w1 = (d11 * d20 - d01 * d21) / den
    w2 = (d00 * d21 - d01 * d20) / den
    w0 = 1 - w1 - w2
    uvs = uv_l[tl[th][tris_h]]                                  # (k, 3, 2)
    uvh[sel[:, 0], sel[:, 1]] = (uvs * np.stack([w0, w1, w2], axis=1)[..., None]).sum(axis=1)
    print("texeles con color de la cabeza: %d (franja %.0f mm)" % (len(sel), FRANJA * 1000))

    # ---- textura de la cabeza ------------------------------------------------------------------------------------------------------
    mat_h = me.materials[0]
    nodo = [n for n in mat_h.node_tree.nodes if n.type == "TEX_IMAGE" and any(l.to_socket.name == "Base Color" for l in n.outputs[0].links)][0]
    img = nodo.image
    img.filepath_raw = os.path.join(W, "cabeza_tex.png")
    img.file_format = "PNG"
    img.save()
    np.savez_compressed(os.path.join(W, "texeles.npz"), cubierto=cubierto, P=P.astype(np.float32), N=Nn.astype(np.float32),
                        D=D.astype(np.float32), uvh=uvh, dh=dh)

    # ---- landmarks 3D sobre la cara unida ------------------------------------------------------------------------------------------
    bvc = BVHTree.FromPolygons([tuple(p) for p in V.tolist()], [tuple(t) for t in tv[tcara].tolist()])
    puntos = {}
    for k, p in lm["base_ajustada_m3d"].items():
        loc, nor, ix, d = bvc.find_nearest(Vector(p))
        puntos[k] = dict(p=list(loc), movido_mm=d * 1000.0, normal=list(nor))
    ojos = {}
    for k, c in zip(("izq", "dch"), centros):
        ids = {i for i in borde_ojo[k]}
        orden = ordenar_lazo_frontal(bm, ids)
        P3 = np.array([bm.verts[i].co[:] for i in orden])
        i0, i1 = int(np.argmin(P3[:, 0])), int(np.argmax(P3[:, 0]))
        n_ = len(P3)
        arco_a = [P3[(i0 + s) % n_] for s in range(((i1 - i0) % n_) + 1)]
        arco_b = [P3[(i0 - s) % n_] for s in range(((i0 - i1) % n_) + 1)]
        sup, inf = (arco_a, arco_b) if np.mean([p[2] for p in arco_a]) > np.mean([p[2] for p in arco_b]) else (arco_b, arco_a)
        ojos[k] = dict(sup=[list(p) for p in sup], inf=[list(p) for p in inf], centro=list(c))
        print("parpados %s: %d vertices en el borde delantero (%d arriba, %d abajo)" % (k, n_, len(sup), len(inf)))
    json.dump(dict(puntos=puntos, ojos=ojos, ancho=lm["ajuste"]["ancho_cara_generada_m"], tex=TEX), open(os.path.join(W, "puntos3d.json"), "w"))
    print("EXTRAIDO", W)


def aplicar():
    png = arg("--tex-png")
    if not png:
        sys.exit("falta --tex-png")
    bpy.ops.wm.open_mainfile(filepath=os.path.join(DIR, "cabeza_unida.blend"))
    mat = bpy.data.materials["cara_tex"]
    tx = [n for n in mat.node_tree.nodes if n.type == "TEX_IMAGE"][0]
    viejo = tx.image
    img = bpy.data.images.load(os.path.abspath(png))
    img.name = "cara_tex_final"
    img.colorspace_settings.name = "sRGB"
    img.pack()
    tx.image = img
    if viejo:
        bpy.data.images.remove(viejo)
    pr = mat.node_tree.nodes["Principled BSDF"]
    pr.inputs["Roughness"].default_value = 0.55
    # ojos: brillo humedo (cornea = capa transparente brillante sobre el iris, sencillo con la capa "coat" de Principled)
    for o in ("ojo_izq", "ojo_dch"):
        m = bpy.data.objects[o].data.materials[0]
        p = m.node_tree.nodes["Principled BSDF"]
        p.inputs["Roughness"].default_value = 0.05
        p.inputs["Specular IOR Level"].default_value = 0.8
        p.inputs["Coat Weight"].default_value = 1.0
        p.inputs["Coat Roughness"].default_value = 0.02
    bpy.ops.object.select_all(action="DESELECT")
    for o in ("cabeza", "ojo_izq", "ojo_dch"):
        bpy.data.objects[o].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects["cabeza"]
    bpy.ops.export_scene.gltf(filepath=os.path.join(DIR, "cabeza_final.glb"), export_format="GLB", use_selection=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(DIR, "cabeza_final.blend"), compress=True)
    print("APLICADO", DIR)


def hoja():
    """Hoja con textura y luz suave de DIR/cabeza_final.blend frente al glb `--glb-antes` y al retrato `--retrato`."""
    lm = json.load(open(os.path.join(DIR, "landmarks.json")))
    bpy.ops.wm.open_mainfile(filepath=os.path.join(DIR, "cabeza_final.blend"))
    m3d = {int(k): np.array(v) for k, v in lm["generada"]["m3d"].items()}
    ctr = np.mean([m3d[i] for i in (234, 454, 152, 10) if i in m3d], axis=0)
    import cara_hoja
    objs = [bpy.data.objects[n] for n in ("cabeza", "ojo_izq", "ojo_dch")]
    cara_hoja.hoja_final(bpy.context.scene, objs, arg("--glb-antes"), arg("--retrato"), ctr, lm["ajuste"]["ancho_cara_generada_m"],
                         arg("--salida", os.path.join(DIR, "hoja_final.jpg")), [lm["ajuste"]["ojos"][k]["centro"] for k in ("izq", "dch")])


if FASE == "extraer":
    extraer()
elif FASE == "aplicar":
    aplicar()
else:
    hoja()
