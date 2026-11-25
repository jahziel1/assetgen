"""orientacion.py -- normales hacia fuera por componente cerrada (volumen firmado > 0). Modulo para `retopologia.py` y `hornear.py` (bpy).

`bmesh.ops.recalc_face_normals` decide el sentido con una heuristica local y en mallas con zonas concavas (daga, mangas) puede dejar
una pieza entera hacia dentro; el volumen firmado de cada componente cerrada no se equivoca."""
import os
import sys

import bmesh
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cli"))
from assetgen_cli import retopo_geom as RG  # noqa: E402


def _arrays(me):
    co = np.empty(len(me.vertices) * 3, np.float64); me.vertices.foreach_get("co", co)
    lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get("vertex_index", lv)
    ini = np.empty(len(me.polygons), np.int32); me.polygons.foreach_get("loop_start", ini)
    fin = np.append(ini[1:], len(lv))
    return co.reshape(-1, 3).tolist(), [tuple(lv[i:j].tolist()) for i, j in zip(ini.tolist(), fin.tolist())]


def volumen_firmado(me):
    """Volumen con signo de toda la malla `me` (positivo = normales hacia fuera)."""
    co, caras = _arrays(me)
    return RG.volumen_firmado(co, caras)


def orientar_hacia_fuera(me):
    """Voltea las componentes CERRADAS con volumen firmado negativo (conserva UV; las abiertas no se tocan). Devuelve un resumen para retopo.json."""
    co, caras = _arrays(me)
    comp = RG.orientacion_por_componente(co, caras)
    antes = sum(v for v, _ in comp)
    malas = RG.caras_a_voltear(co, caras)
    if malas:
        bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
        bmesh.ops.reverse_faces(bm, faces=[bm.faces[i] for i in malas])
        bm.to_mesh(me); bm.free(); me.update()
    co, caras = _arrays(me)
    comp2 = RG.orientacion_por_componente(co, caras)
    return {"componentes": len(comp), "caras_volteadas": len(malas), "volumen_antes_m3": round(antes, 6),
            "volumen_despues_m3": round(sum(v for v, _ in comp2), 6), "hacia_fuera": not RG.caras_a_voltear(co, caras)}


def quitar_piel_interior(me, grosor=0.015, dirs=24, umbral=0.25, sesgo=0.004):
    """La malla de capas finas (tela) remallada por voxeles sale con dos pieles: exterior e interior (interior de mangas, perneras, cuello).
    La interior no se ve, duplica los triangulos y, con pesos distintos a los de la exterior, la atraviesa al posar. Quita las caras que
    cumplen `retopo_geom.es_piel_interior`: otra cara de frente a menos de `grosor` m por detras y, desde `sesgo` m delante, solo choca con
    la propia malla en menos de `umbral` de `dirs` direcciones. Devuelve el resumen (caras antes/despues, aristas abiertas nuevas). EXPERIMENTAL: deja la malla abierta (telas de una piel) y no
    arreglo las manchas de la oclusion/color; ver el informe de la etapa 5d.7."""
    from mathutils.bvhtree import BVHTree
    from mathutils import Vector
    bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
    bvh = BVHTree.FromBMesh(bm)
    dirs_ = [Vector(d) for d in RG.direcciones_esfera(dirs)]
    borrar = []
    for f in bm.faces:
        c, n = f.calc_center_median(), f.normal
        atras = bvh.ray_cast(c - n * 1e-4, -n, grosor)
        fina = atras[0] is not None and atras[1].dot(n) < -0.3
        if not fina:
            continue
        o = c + n * sesgo
        esc = sum(1 for d in dirs_ if d.dot(n) > 0 and bvh.ray_cast(o, d)[0] is None) / max(1, sum(1 for d in dirs_ if d.dot(n) > 0))
        if RG.es_piel_interior(esc, fina, umbral):
            borrar.append(f)
    n0 = len(bm.faces)
    abiertas0 = sum(1 for e in bm.edges if e.is_boundary)
    bmesh.ops.delete(bm, geom=borrar, context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    abiertas1 = sum(1 for e in bm.edges if e.is_boundary)
    bm.to_mesh(me); bm.free(); me.update()
    return {"caras_antes": n0, "caras_quitadas": len(borrar), "caras_despues": n0 - len(borrar), "aristas_abiertas_antes": abiertas0,
            "aristas_abiertas_despues": abiertas1}
