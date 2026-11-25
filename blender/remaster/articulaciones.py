"""articulaciones.py -- estima hombros, codos, munecas, caderas, rodillas, tobillos y cuello de un personaje (z arriba, pies en z = 0,
mirando a -Y, brazos a +X/-X) y los guarda en un JSON. Sirve para el original (con grupos de vertices de hueso) y para un modelo sin
esqueleto (analisis de la malla). Sin rutas ni datos de ningun juego.

    blender -b -P articulaciones.py -- --modo original --glb original.glb --salida articulaciones.json
    blender -b -P articulaciones.py -- --modo malla    --glb modelo.glb   --salida articulaciones.json

Modo original: cada articulacion es el cruce 0.5 de la fraccion de peso distal entre dos grupos de huesos vecinos (pose_geom.cruce_pesos).
Modo malla (modelo en pose A o T, cualquier brazo): por brazo, eje hombro-mano (hombro provisional bajo el acromion, mano = centroide
de los vertices con x >= 0.88 de la x maxima) y secciones transversales a lo largo del eje; muneca = estrechamiento minimo de la seccion antes de la mano (centro
de la seccion); hombro = eje del brazo superior (centros de seccion) a `acromion - 0.04 m` de altura; codo = punto medio hombro-muneca
(antropometrico; el puno abullonado acaba hacia 0.7 de ese tramo y se informa como `puno`); tobillo = primera z (desde el suelo) donde la
profundidad (y) de la seccion de la pierna cae a la del tramo de la caña; cadera = 0.50 de la altura; rodilla = mitad entre tobillo y
cadera (la ropa holgada tapa las articulaciones: antropometrico); cuello = seccion de menor anchura entre 0.80 y 0.90 de la altura.
Cada articulacion lleva `metodo`. Imprime 'ARTICULACIONES {resumen}'."""
import json
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "..", "cli"))
from assetgen_cli import pose_geom as G  # noqa: E402

CLAVES = ["cuello", "hombro_I", "codo_I", "muneca_I", "cadera_I", "rodilla_I", "tobillo_I",
          "hombro_D", "codo_D", "muneca_D", "cadera_D", "rodilla_D", "tobillo_D"]


def midrango(a, p=2.0):
    return (np.percentile(a, p) + np.percentile(a, 100 - p)) / 2


def perfil_brazo(V, S0, H0, paso=0.03, tmin=-0.05, perp_max=0.2, x_min=0.12):
    """Secciones transversales del brazo (lado +X) a lo largo del eje S0->H0. Devuelve (u, w, L, lista de dict t, n, ancho, centro)."""
    S0, H0 = np.asarray(S0, float), np.asarray(H0, float)
    u = H0 - S0
    L = float(np.linalg.norm(u))
    u = u / L
    w = np.cross(u, [0, 1, 0])
    w /= np.linalg.norm(w)
    r = V - S0
    t = r @ u
    rr = r - np.outer(t, u)
    perp = np.linalg.norm(rr, axis=1)
    base = (V[:, 0] > x_min) & (perp < perp_max)
    out = []
    for t0 in np.arange(tmin, L + 0.1, paso):
        k = base & (t >= t0) & (t < t0 + paso)
        if k.sum() < 30:
            continue
        a = rr[k] @ w
        b = rr[k][:, 1]
        ca, cb = midrango(a, 3), midrango(b, 3)
        c = S0 + u * (t0 + paso / 2) + w * ca + np.array([0, cb, 0])
        out.append({"t": float(t0 + paso / 2), "n": int(k.sum()), "ancho": float(np.percentile(a, 97) - np.percentile(a, 3)),
                    "prof": float(np.percentile(b, 97) - np.percentile(b, 3)), "c": c})
    return u, w, L, out


def analizar_brazo(V, lado, altura):
    """Hombro, codo, muneca, centro de mano del brazo `lado` (+1 / -1) de una malla. Devuelve dict de puntos y datos auxiliares."""
    X = V * np.array([lado, 1.0, 1.0])
    banda = (X[:, 0] > 0.19) & (X[:, 0] < 0.32) & (X[:, 2] > 0.7 * altura)
    acromion_z = float(X[banda][:, 2].max())
    lat = X[:, 0] >= 0.88 * X[:, 0].max()
    H0 = X[lat].mean(0)
    S0 = np.array([0.20, 0.0, acromion_z - 0.05])
    u, w, L, prof = perfil_brazo(X, S0, H0)
    # muneca: ancho minimo entre 0.55 L y 0.85 L
    cand = [p for p in prof if 0.55 * L <= p["t"] <= 0.85 * L]
    # suavizado de 3 secciones
    anch = [np.mean([q["ancho"] for q in prof[max(0, prof.index(p) - 1):prof.index(p) + 2]]) for p in cand]
    pm = cand[int(np.argmin(anch))]
    muneca = pm["c"]
    # brazo superior: centros de seccion en [0.30 L, 0.55 L] de la muneca hacia el hombro
    sup = [p for p in prof if 0.30 * L <= p["t"] <= 0.55 * L]
    P = np.array([p["c"] for p in sup])
    P0 = P.mean(0)
    v = np.linalg.svd(P - P0)[2][0]
    if v[2] > 0:
        v = -v
    zs = acromion_z - 0.04
    hombro = P0 + v * ((zs - P0[2]) / v[2])
    hombro[1] = P0[1]
    codo = (hombro + muneca) / 2
    # puno: primera seccion tras el maximo de anchura donde el ancho cae a la mitad entre maximo y muneca
    amax = max(prof, key=lambda p: p["ancho"] if p["t"] < 0.55 * L else 0)
    mitad = (amax["ancho"] + pm["ancho"]) / 2
    pun = next((p for p in prof if p["t"] > amax["t"] and p["ancho"] <= mitad), pm)
    # centro de la mano: vertices mas alla de la muneca (plano perpendicular a muneca->H0) y a menos de 0.12 de esa recta
    d = H0 - muneca
    d /= np.linalg.norm(d)
    r = X - muneca
    tt = r @ d
    pp = np.linalg.norm(r - np.outer(tt, d), axis=1)
    mh = (tt > 0.01) & (pp < 0.12) & (X[:, 0] > muneca[0] - 0.02)
    mano = X[mh].mean(0)
    punta = X[mh][np.argmax(tt[mh])]
    f = np.array([lado, 1.0, 1.0])
    return {"hombro": hombro * f, "codo": codo * f, "muneca": muneca * f, "mano_centro": mano * f, "mano_punta": punta * f,
            "puno": pun["c"] * f, "acromion_z": acromion_z, "ancho_muneca": pm["ancho"], "ancho_max_manga": amax["ancho"],
            "n_mano": int(mh.sum()), "perfil": prof, "L": L}


def angulos_brazo(V, lado, altura):
    """Direccion del eje del brazo superior y del antebrazo (ajuste lineal de los centros de seccion de la manga en 0.30-0.55 L y 0.60-0.80 L
    del eje hombro-mano) como grados sobre la horizontal y grados con la vertical hacia abajo. Valido en pose A y en pose T."""
    X = V * np.array([lado, 1.0, 1.0])
    banda = (X[:, 0] > 0.19) & (X[:, 0] < 0.32) & (X[:, 2] > 0.7 * altura)
    ac = float(X[banda][:, 2].max())
    H0 = X[X[:, 0] >= 0.88 * X[:, 0].max()].mean(0)
    u, w, L, prof = perfil_brazo(X, [0.20, 0.0, ac - 0.05], H0)
    out = {}
    for nm, (f0, f1) in (("brazo", (0.30, 0.55)), ("antebrazo", (0.60, 0.80))):
        P = np.array([p["c"] for p in prof if f0 * L <= p["t"] <= f1 * L])
        v = np.linalg.svd(P - P.mean(0))[2][0]
        if v @ u < 0:
            v = -v
        v = v * np.array([lado, 1.0, 1.0])
        out[nm] = {"sobre_horizontal_deg": round(G.angulo_sobre_horizontal(list(v)), 2),
                   "con_vertical_abajo_deg": round(G.angulo_con_vertical_abajo(list(v)), 2)}
    return out


def analizar_piernas(V, lado, altura):
    X = V * np.array([lado, 1.0, 1.0])
    P = X[(X[:, 0] > 0.0) & (X[:, 0] < 0.3)]

    def seccion(z0, z1):
        k = (P[:, 2] >= z0) & (P[:, 2] < z1)
        q = P[k]
        return np.array([midrango(q[:, 0], 5), midrango(q[:, 1], 5)]), float(np.percentile(q[:, 1], 98) - np.percentile(q[:, 1], 2))
    _, prof_ref = seccion(0.15, 0.27)
    tob_z = None
    for z in np.arange(0.01, 0.2, 0.01):
        if seccion(z, z + 0.02)[1] < 1.6 * prof_ref:
            tob_z = float(z + 0.01)
            break
    c_t, _ = seccion(tob_z, tob_z + 0.10)
    tobillo = np.array([c_t[0], c_t[1], tob_z])
    cad_z = 0.50 * altura
    c_c, _ = seccion(cad_z - 0.10, cad_z - 0.02)
    cadera = np.array([c_c[0], c_c[1], cad_z])
    rod_z = (tob_z + cad_z) / 2
    c_r, _ = seccion(rod_z - 0.02, rod_z + 0.02)
    rodilla = np.array([c_r[0], c_r[1], rod_z])
    f = np.array([lado, 1.0, 1.0])
    return {"cadera": cadera * f, "rodilla": rodilla * f, "tobillo": tobillo * f}


def analizar_cuello(V, altura):
    zs = np.arange(0.80 * altura, 0.90 * altura, 0.01)
    best = None
    for z in zs:
        k = (V[:, 2] >= z) & (V[:, 2] < z + 0.02) & (np.abs(V[:, 0]) < 0.2)
        if k.sum() < 200:
            continue
        a = V[k]
        ancho = np.percentile(a[:, 0], 98) - np.percentile(a[:, 0], 2)
        if best is None or ancho < best[0]:
            best = (ancho, z + 0.01, midrango(a[:, 0], 2), midrango(a[:, 1], 2))
    return np.array([best[2], best[3], best[1]]), float(best[0])


def pt(p, nd=4):
    return [round(float(x), nd) for x in p]


def modo_malla(V):
    altura = float(V[:, 2].max() - V[:, 2].min())
    res = {}
    aux = {}
    for lado, s in ((1, "I"), (-1, "D")):
        b = analizar_brazo(V, lado, altura)
        l = analizar_piernas(V, lado, altura)
        res["hombro_" + s], res["codo_" + s], res["muneca_" + s] = b["hombro"], b["codo"], b["muneca"]
        res["cadera_" + s], res["rodilla_" + s], res["tobillo_" + s] = l["cadera"], l["rodilla"], l["tobillo"]
        aux["mano_centro_" + s], aux["mano_punta_" + s], aux["puno_" + s] = b["mano_centro"], b["mano_punta"], b["puno"]
        aux["info_brazo_" + s] = {"acromion_z": round(b["acromion_z"], 4), "ancho_muneca": round(b["ancho_muneca"], 4),
                                  "ancho_max_manga": round(b["ancho_max_manga"], 4), "vertices_mano": b["n_mano"]}
    res["cuello"], aux["ancho_cuello"] = analizar_cuello(V, altura)
    return res, aux, altura


def modo_original(V, pesos):
    """V: lista [x,y,z]; pesos: lista de dict hueso->peso (indices numericos de los grupos hNN)."""
    res = {}
    tren = [16, 17, 18]
    cuello_d = [19] + list(range(4, 15)) + [44, 45, 46, 47]
    cfg = {"I": dict(sg=1, br=([20, 21], [48, 56, 57, 49, 50, 51, 52, 53, 54, 55], [48, 56, 57], [49, 50, 51, 52, 53, 54, 55]),
                     pi=([24], [25, 26], [25], [26])),
           "D": dict(sg=-1, br=([22, 23], [58, 66, 67, 59, 60, 61, 62, 63, 64, 65], [58, 66, 67], [59, 60, 61, 62, 63, 64, 65]),
                     pi=([27], [28, 29], [28], [29]))}
    n = {}
    for s, c in cfg.items():
        sg = c["sg"]
        hs, n["hombro_" + s] = G.cruce_pesos(V, pesos, tren, c["br"][0], 0, sg, (0.0, 0.5))
        ce, n["codo_" + s] = G.cruce_pesos(V, pesos, c["br"][0], c["br"][1], 0, sg, (0.2, 0.9))
        mu, n["muneca_" + s] = G.cruce_pesos(V, pesos, c["br"][2], c["br"][3], 0, sg, (0.5, 1.0))
        ca, n["cadera_" + s] = G.cruce_pesos(V, pesos, [15], c["pi"][0], 2, -1, (-1.2, -0.7))
        ro, n["rodilla_" + s] = G.cruce_pesos(V, pesos, c["pi"][0], c["pi"][1], 2, -1, (-0.9, -0.1))
        to, n["tobillo_" + s] = G.cruce_pesos(V, pesos, c["pi"][2], c["pi"][3], 2, -1, (-0.4, 0.0))
        res.update({"hombro_" + s: hs, "codo_" + s: ce, "muneca_" + s: mu, "cadera_" + s: ca, "rodilla_" + s: ro, "tobillo_" + s: to})
    res["cuello"], n["cuello"] = G.cruce_pesos(V, pesos, tren, cuello_d, 2, 1, (1.4, 1.8))
    aux = {}
    for s, hand in (("I", range(49, 56)), ("D", range(59, 66))):
        hand = set(hand)
        num, den = [0.0] * 3, 0.0
        for p, w in zip(V, pesos):
            m = sum(v for k, v in w.items() if k in hand)
            if m > 0:
                num = [a + m * b for a, b in zip(num, p)]
                den += m
        aux["mano_centro_" + s] = [a / den for a in num]
    return res, aux, n


def main():
    a = sys.argv[sys.argv.index("--") + 1:]

    def arg(n, d=None):
        return a[a.index(n) + 1] if n in a else d
    modo, glb, salida = arg("--modo"), arg("--glb"), arg("--salida")
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb)
    if modo == "original":
        o = next(o for o in bpy.data.objects if o.type == "MESH" and any(g.name.startswith("h") for g in o.vertex_groups))
        M = np.array(o.matrix_world)
        names = [g.name for g in o.vertex_groups]
        pos = [list(map(float, M[:3, :3] @ np.array(v.co) + M[:3, 3])) for v in o.data.vertices]
        pes = [{int(names[g.group][1:]): g.weight for g in v.groups if names[g.group][1:].isdigit()} for v in o.data.vertices]
        res, aux, n = modo_original(pos, pes)
        altura = max(p[2] for p in pos) - min(p[2] for p in pos)
        meta = {"metodo": "cruce 0.5 de la fraccion de peso distal entre grupos de hueso vecinos (pose_geom.cruce_pesos)",
                "vertices_en_losa": n, "grupos": {"tronco": [16, 17, 18], "brazo_I": [20, 21], "antebrazo_I": [48, 56, 57], "mano_I": list(range(49, 56)),
                                                  "brazo_D": [22, 23], "antebrazo_D": [58, 66, 67], "mano_D": list(range(59, 66)),
                                                  "pelvis": [15], "muslo_I": [24], "pierna_I": [25], "pie_I": [26], "muslo_D": [27],
                                                  "pierna_D": [28], "pie_D": [29], "cabeza": [19] + list(range(4, 15)) + [44, 45, 46, 47]}}
    else:
        o = max((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: len(o.data.vertices))
        M = np.array(o.matrix_world)
        n = len(o.data.vertices)
        co = np.empty(n * 3, np.float32)
        o.data.vertices.foreach_get("co", co)
        V = co.reshape(-1, 3).astype(np.float64) @ M[:3, :3].T + M[:3, 3]
        res, aux, altura = modo_malla(V)
        meta = {"metodo": __doc__.split("Modo malla")[1].split("Cada articulacion")[0].strip()}
    out = {"unidades": "m", "ejes": "z arriba, frente -Y, I = +X, D = -X", "altura_m": round(float(altura), 4),
           "articulaciones": {k: pt(res[k]) for k in CLAVES}, "extras": {k: (pt(v) if hasattr(v, "__len__") and len(v) == 3 and not isinstance(v, dict) else v) for k, v in aux.items()},
           **meta}
    json.dump(out, open(salida, "w"), indent=1, ensure_ascii=False)
    print("ARTICULACIONES", json.dumps(out["articulaciones"]))


if __name__ == "__main__":
    main()
