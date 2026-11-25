"""exportar_visor.py -- de un .glb con grupos de vertices `hNN` + poses de hueso (bind -> pantalla) a la carpeta de datos del visor Godot.

    blender -b -P exportar_visor.py -- --glb modelo_bind.glb --pose-a A.json --pose-b B.json --salida DIR
        [--enlace enlace.json] [--caras auto|mantener|invertir] [--altura 1.75] [--capsulas-min 60]
        [--validar modelo_poseA.glb [--grupos grupos.json] [--validar-pose A|B]]

Entrada
  * glb (o .blend): una o varias mallas; cada vertice tiene pesos en grupos llamados `h00`, `h01`... (hueso global NN). Se usan los 4 mayores.
  * --pose-a/--pose-b: JSON = lista (un elemento por hueso global) de 12 floats: base 3x3 por COLUMNAS + traslacion, que llevan el espacio
    bind ORIGINAL del hueso a la pantalla del juego (x derecha, y abajo, z hacia dentro).
  * --enlace: igual formato; lleva el bind ORIGINAL del hueso al bind en que estan los vertices del glb. Hace falta cuando hay piezas con
    espacio bind propio (cabeza, arma, faldon) llevadas al del cuerpo: la matriz final es pantalla . pose . enlace^-1. Sin el, enlace = identidad.
  * Texturas: si el glb tiene material con imagenes (color base, metal-rugosidad R=oclusion G=rugosidad B=metal, normal) se guardan como
    color.jpg / mat.jpg / normal.jpg; si no, `color` plano en el JSON (el visor genera texturas de 1 pixel).
Salida (DIR): pos.f32 nrm.f32 uv.f32 hueso.i32 peso.f32 idx.i32 modelo.json (poses A/B en el orden Transform3D de Godot, ya en
  espacio Godot, pies en y=0, centrado y de `--altura` m en la pose A; capsulas por hueso en espacio bind; escala).
  idx.i32 va con caras antihorarias hacia fuera (convencion glTF; el visor las gira para Godot). Si las caras salen hacia dentro (volumen con signo negativo, tipico de mallas con orden de vertices D3D) se invierten caras y normales.
--validar: skinning de las matrices de la pose (--validar-pose A|B, por defecto A) sobre los vertices del glb comparado con otro glb
  que ya trae esos vertices en pantalla (convencion Blender = (x, z, -y) de la pantalla, ver `pantalla_a_blender`; se quitan min z y media x/y como
  hacia el script que lo creo). Imprime el error maximo en mm, total y por grupo (--grupos: {"nombre": [huesos]}). Imprime 'VISOR {resumen}'."""
import json
import math
import os
import re
import sys

import bpy
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "..", "cli"))
from assetgen_cli import visor_geom as VG  # noqa: E402

a = sys.argv[sys.argv.index("--") + 1:]


def arg(n, d=None):
    return a[a.index(n) + 1] if n in a else d


def pantalla_a_blender(p):
    """Pantalla del juego (x der, y abajo, z dentro) -> Blender (Z arriba, el frente mira a -Y)."""
    return [p[0], p[2], -p[1]]


def limpiar():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def importar(ruta):
    if ruta.lower().endswith(".blend"):         # p. ej. el proxy con pesos de `transferir_pesos.py`
        bpy.ops.wm.open_mainfile(filepath=ruta)
    else:
        limpiar()
        bpy.ops.import_scene.gltf(filepath=ruta)
    # el importador crea una «Icosphere» como forma de hueso: se descartan las mallas sin grupos de vertices (no son piel)
    mallas = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    return [o for o in mallas if o.vertex_groups] or mallas


def leer_malla(objs):
    """Lee las mallas sin expandir: posiciones mundo, grupos hNN por vertice y, por separado, las esquinas (vi, uv, normal) y triangulos."""
    P, G, esquinas = [], [], []
    off = 0
    for ob in objs:
        me = ob.data
        mw = np.array(ob.matrix_world)
        n = len(me.vertices)
        co = np.empty(n * 3); me.vertices.foreach_get("co", co); co = co.reshape(-1, 3)
        P.append(co @ mw[:3, :3].T + mw[:3, 3])
        nombres = {i: vg.name for i, vg in enumerate(ob.vertex_groups)}
        for v in me.vertices:
            g = {}
            for e in v.groups:
                m = re.fullmatch(r"h(\d+)", nombres.get(e.group, ""))
                if m and e.weight > 0:
                    g[int(m.group(1))] = g.get(int(m.group(1)), 0.0) + e.weight
            G.append(g)
        me.calc_loop_triangles()
        nl = len(me.loops)
        vi = np.empty(nl, np.int64); me.loops.foreach_get("vertex_index", vi)
        cn = np.empty(nl * 3); me.corner_normals.foreach_get("vector", cn); cn = cn.reshape(-1, 3)
        cn = cn @ mw[:3, :3].T
        cn /= np.maximum(np.linalg.norm(cn, axis=1, keepdims=True), 1e-12)
        if me.uv_layers.active is not None:
            uv = np.empty(nl * 2); me.uv_layers.active.data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
        else:
            uv = np.zeros((nl, 2))
        tl = np.empty(len(me.loop_triangles) * 3, np.int64); me.loop_triangles.foreach_get("loops", tl)
        esquinas.append((vi + off, uv, cn, tl.reshape(-1, 3), ob))
        off += n
    return np.vstack(P), G, esquinas


def volumen_con_signo(P, T):
    A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    return float(np.einsum("ij,ij->i", A, np.cross(B, C)).sum() / 6.0)


def guardar_imagen(img, ruta):
    # save_render respeta el formato pedido tambien en imagenes generadas/empaquetadas (img.save() las dejaba en PNG con extension .jpg)
    ajustes = bpy.context.scene.render.image_settings
    ajustes.file_format, ajustes.color_mode, ajustes.quality = "JPEG", "RGB", 92
    img.save_render(ruta)


def texturas(objs, salida):
    """Busca en el primer material con nodos las imagenes de color, metal-rugosidad y normal. Devuelve ({nombre: bool}, color base o None)."""
    hechas, color = {"color": False, "mat": False, "normal": False}, None
    for ob in objs:
        for slot in ob.material_slots:
            mat = slot.material
            if mat is None or not mat.use_nodes:
                continue
            bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if bsdf is None:
                continue

            def imagen(entrada, bajar=0):
                nodo = entrada.links[0].from_node if entrada.is_linked else None
                for _ in range(bajar + 3):
                    if nodo is None or nodo.type == "TEX_IMAGE":
                        break
                    ent = next((i for i in nodo.inputs if i.is_linked), None)
                    nodo = ent.links[0].from_node if ent else None
                return nodo.image if nodo is not None and nodo.type == "TEX_IMAGE" else None
            for clave, entrada in (("color", bsdf.inputs["Base Color"]), ("normal", bsdf.inputs["Normal"]),
                                   ("mat", bsdf.inputs["Roughness"])):
                if hechas[clave]:
                    continue
                img = imagen(entrada)
                if img is not None:
                    guardar_imagen(img, os.path.join(salida, clave + ".jpg"))
                    hechas[clave] = True
            if not hechas["color"]:
                c = bsdf.inputs["Base Color"].default_value
                color = [round(float(c[0]), 4), round(float(c[1]), 4), round(float(c[2]), 4)]
            return hechas, color
    return hechas, color


def main():
    glb, salida = arg("--glb"), arg("--salida")
    if not glb or not salida:
        raise SystemExit("faltan --glb y --salida")
    pose = {k: json.load(open(arg("--pose-" + k.lower()))) for k in "AB"}
    enlace = json.load(open(arg("--enlace"))) if arg("--enlace") else None
    nb = len(pose["A"])
    os.makedirs(salida, exist_ok=True)
    objs = importar(glb)
    if not objs:
        raise SystemExit("el glb no tiene mallas")
    P, G, esq = leer_malla(objs)
    hmax = max((h for g in G for h in g), default=-1)
    if hmax >= nb or len(pose["B"]) != nb:
        raise SystemExit("hueso h%02d fuera de las poses (%d huesos en A, %d en B)" % (hmax, nb, len(pose["B"])))
    sinpeso = [i for i, g in enumerate(G) if not g]
    if sinpeso:
        print("AVISO %d vertices sin pesos hNN (se dejan en el hueso 0 con peso 1)" % len(sinpeso))
    HP = [VG.mejores_pesos(g if g else {0: 1.0}) for g in G]

    # --- esquinas -> vertices expandidos (uv y normal por esquina), caras
    V, U, N, T, o = [], [], [], [], 0
    for vi, uv, cn, tl, ob in esq:
        clave = np.concatenate([vi[:, None].astype(np.float64), np.round(uv * 1e5), np.round(cn * 1e3)], 1)
        _, primero, inverso = np.unique(clave, axis=0, return_index=True, return_inverse=True)
        inverso = inverso.ravel()
        V.append(vi[primero]); U.append(uv[primero]); N.append(cn[primero]); T.append(inverso[tl] + o)
        o += len(primero)
    V, U, N, T = np.concatenate(V), np.vstack(U), np.vstack(N), np.vstack(T)
    modo = arg("--caras", "auto")
    vol = volumen_con_signo(P[V], T)
    invertir = modo == "invertir" or (modo == "auto" and vol < 0)
    if invertir:
        T = T[:, [0, 2, 1]]
        N = -N
    print("CARAS volumen con signo %.4f m3 -> %s" % (vol, "invertidas" if invertir else "se mantienen"))

    # --- poses: bind del modelo -> Godot, normalizadas (pies en 0, centrado, altura fija en A, misma escala en B)
    Pl = P.tolist()
    H = [h for h, _ in HP]; W = [w for _, w in HP]
    mats = {k: VG.matrices_hueso(pose[k], enlace, VG.PANTALLA_A_GODOT) for k in "AB"}
    sk = {k: VG.skin_lineal(Pl, H, W, mats[k]) for k in "AB"}
    TA, escala = VG.normalizar_a_suelo(sk["A"], float(arg("--altura", 1.75)))
    TB, _ = VG.normalizar_a_suelo(sk["B"], escala=escala)
    salida_poses = {}
    for k, Tk in (("A", TA), ("B", TB)):
        salida_poses[k] = [VG.a_transform3d(VG.mul4(Tk, m)) for m in mats[k]]
        c, eje, _ = VG.eje_principal(sk[k][::max(1, len(Pl) // 4000)])
        print("EJE pose %s: inclinacion del eje principal sobre la vertical %.1f grados" % (k, math.degrees(math.acos(min(1.0, abs(eje[1]))))))
    dom = [h[max(range(4), key=lambda j: w[j])] for h, w in zip(H, W)]
    caps = VG.capsulas(Pl, dom, int(arg("--capsulas-min", 60)))

    # --- binarios
    nv = len(V)
    Pv = P[V].astype("<f4"); Pv.tofile(os.path.join(salida, "pos.f32"))
    N.astype("<f4").tofile(os.path.join(salida, "nrm.f32"))
    U.astype("<f4").tofile(os.path.join(salida, "uv.f32"))
    np.array([H[i] for i in V], "<i4").tofile(os.path.join(salida, "hueso.i32"))
    np.array([W[i] for i in V], "<f4").tofile(os.path.join(salida, "peso.f32"))
    T.astype("<i4").ravel().tofile(os.path.join(salida, "idx.i32"))
    hechas, color = texturas(objs, salida)
    json.dump({"vertices": nv, "indices": int(T.size), "huesos": nb, "escala": escala, "poses": salida_poses, "capsulas": caps,
               "texturas": hechas, "color": color, "caras_invertidas": bool(invertir)}, open(os.path.join(salida, "modelo.json"), "w"))
    resumen = {"vertices": nv, "triangulos": int(len(T)), "huesos": nb, "capsulas": len(caps), "escala": round(escala, 5),
               "texturas": [k for k, v in hechas.items() if v], "caras_invertidas": bool(invertir)}

    # --- validacion opcional contra un glb ya en pantalla
    if arg("--validar"):
        clave = arg("--validar-pose", "A").upper()
        M = VG.matrices_hueso(pose[clave], enlace, VG.IDENT)
        S = [pantalla_a_blender(p) for p in VG.skin_lineal(Pl, H, W, M)]
        S = np.array(S)
        S[:, :2] -= S[:, :2].mean(0); S[:, 2] -= S[:, 2].min()
        ref, _, _ = leer_malla(importar(arg("--validar")))
        if len(ref) != len(S):
            raise SystemExit("el glb de validacion tiene %d vertices y el modelo %d" % (len(ref), len(S)))
        d = np.linalg.norm(S - ref, axis=1) * 1000.0
        val = {"max_mm": round(float(d.max()), 5), "media_mm": round(float(d.mean()), 5), "n": len(d)}
        if arg("--grupos"):
            grupos = json.load(open(arg("--grupos")))
            val["grupos"] = {}
            for nombre, hs in grupos.items():
                sel = np.array([h in hs for h in dom])
                val["grupos"][nombre] = {"n": int(sel.sum()), "max_mm": round(float(d[sel].max()), 5) if sel.any() else None}
        resumen["validacion"] = val
    print("VISOR " + json.dumps(resumen))


main()
