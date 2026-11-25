"""estiramiento.py -- estiramiento de aristas de una malla con pesos `hNN` entre el reposo y las poses A y B del juego.

    blender -b -P estiramiento.py -- --malla proxy_pesos.blend --pose-a pose_A.json --pose-b pose_B.json --enlace pose_bind.json
        [--largo-min 0.001] [--visor DIR_DATOS --articulaciones art.json] [--salida est.json]

Razon = longitud de la arista en la pose / longitud en reposo (skinning lineal, igual que el visor; matriz = pose . enlace^-1).
Da percentiles, cuantas aristas pasan de 2x o bajan de 0.5x y donde (segmento del vertice, altura y lado). Con --visor y --articulaciones
escribe ademas la posicion en el espacio del visor (Godot) de hombro, codo, rodilla y cuello en la pose B (para los primeros planos).
Imprime 'ESTIRA {resumen}'."""
import json
import os
import sys

import bpy
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "..", "cli"))
from assetgen_cli import pesos_geom as PG  # noqa: E402

a = sys.argv[sys.argv.index("--") + 1:]


def arg(n, d=None):
    return a[a.index(n) + 1] if n in a else d


def m4(m):
    return np.array([[m[0], m[3], m[6], m[9]], [m[1], m[4], m[7], m[10]], [m[2], m[5], m[8], m[11]], [0, 0, 0, 1.0]])


def main():
    bpy.ops.wm.open_mainfile(filepath=arg("--malla"))
    ob = next(o for o in bpy.data.objects if o.type == "MESH"); me = ob.data
    n = len(me.vertices)
    co = np.empty(n * 3); me.vertices.foreach_get("co", co); P = co.reshape(-1, 3)
    nombres = {i: int(g.name[1:]) for i, g in enumerate(ob.vertex_groups)}
    W = np.zeros((n, 3)); H = np.zeros((n, 3), int)
    for v in me.vertices:
        for k, e in enumerate(sorted(v.groups, key=lambda e: -e.weight)[:3]):
            H[v.index, k] = nombres[e.group]; W[v.index, k] = e.weight
    W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
    me.calc_loop_triangles(); T = np.empty(len(me.loop_triangles) * 3, np.int64); me.loop_triangles.foreach_get("vertices", T); T = T.reshape(-1, 3)
    seg = np.empty(n, np.int32); me.attributes["segmento"].data.foreach_get("value", seg)
    enl = json.load(open(arg("--enlace")))
    E = np.unique(np.sort(np.vstack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1), axis=0)
    L0 = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)
    ok = L0 > float(arg("--largo-min", 0.001))      # aristas de <1 mm (vertices duplicados en costuras) no cuentan: su razon no significa nada
    res = {"aristas": int(ok.sum())}
    sk = {}
    for k in "AB":
        pose = json.load(open(arg("--pose-" + k.lower())))
        M = np.array([m4(p) @ np.linalg.inv(m4(e)) for p, e in zip(pose, enl)])
        S = np.zeros_like(P)
        for j in range(3):
            Mj = M[H[:, j]]
            S += W[:, j:j + 1] * (np.einsum("nij,nj->ni", Mj[:, :3, :3], P) + Mj[:, :3, 3])
        sk[k] = S
        r = (np.linalg.norm(S[E[:, 0]] - S[E[:, 1]], axis=1) / np.where(ok, L0, 1.0))[ok]
        info = PG.resumen_estiramiento(r.tolist())
        mal = np.where(ok)[0][(r > 2.0) | (r < 0.5)]
        zonas = {}
        for e in mal:
            i, j = E[e]; c = (P[i] + P[j]) / 2
            clave = "%s z=%.1f %s" % (PG.SEGMENTOS[seg[i]], round(c[2], 1), "I" if c[0] > 0.05 else ("D" if c[0] < -0.05 else "centro"))
            zonas[clave] = zonas.get(clave, 0) + 1
        info["donde"] = dict(sorted(zonas.items(), key=lambda kv: -kv[1])[:12])
        res[k] = info
    if arg("--visor") and arg("--articulaciones"):
        art = json.load(open(arg("--articulaciones")))["articulaciones"]
        mv = json.load(open(os.path.join(arg("--visor"), "modelo.json")))["poses"]["B"]
        alvo = {}
        for nombre, hueso in (("hombro_I", 20), ("codo_I", 21), ("rodilla_I", 25), ("cuello", 19)):
            g = np.array(mv[hueso]); p = np.array(art[nombre])
            alvo[nombre] = (np.array(g[:9]).reshape(3, 3).T @ p + g[9:12]).tolist()   # base por columnas
        res["objetivos_B_godot"] = alvo
    if arg("--salida"):
        json.dump(res, open(arg("--salida"), "w"), indent=1)
    print("ESTIRA " + json.dumps(res))


main()
