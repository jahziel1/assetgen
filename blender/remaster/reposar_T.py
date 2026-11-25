"""reposar_T.py -- lleva un personaje de pose A a pose T (brazos horizontales) sin esqueleto previo y deja la T como nueva forma de reposo.

    blender -b -P reposar_T.py -- --glb modelo_A.glb --articulaciones art_A.json --salida DIR [--nombre modelo_T]
        [--angulo -1.3] [--clavicula 0.25] [--semilla-brazo 0.18] [--suavizado 6] [--piel dqs|lbs] [--no-exportar]

Metodo (la malla puede tener >1M de triangulos; nada de bpy.ops sobre ella; el calor de Blender falla en estas mallas: "Bone Heat
Weighting: failed to find solution", por eso los pesos se calculan aqui):
  1. Se sueldan los vertices duplicados por UV/material (indice canonico) y se arma el grafo de aristas.
  2. Pesos: campo armonico u en ese grafo por gradiente conjugado (u = 1 en vertices claramente de manga/brazo: a >= `--semilla-brazo` m del
     hombro a lo largo de hombro-codo-muneca-mano, a < 0.125 m de esa poligonal y x > 0.215; u = 0 en el tronco central, piernas y el otro
     lado). u cae de forma natural en la sisa. u se reparte en tronco / clavicula / brazo con smoothstep, el brazo en brazo / antebrazo /
     mano por los planos bisectores de codo (+-`--ancho-codo`) y muneca (+-`--ancho-muneca`), y se suaviza `--suavizado` iteraciones.
  3. Pose: el brazo se lleva al plano horizontal a `--angulo` grados sobre la horizontal (negativo = mano mas baja). El giro se reparte:
     la clavicula toma la fraccion `--clavicula` del giro del brazo superior (sube el hombro) alrededor del esternon; despues giran
     brazo (alrededor del hombro), antebrazo (codo) y mano (muneca) hasta quedar los tres tramos en la direccion objetivo. Piel por
     cuaterniones duales (`dqs`, por defecto) o lineal (`lbs`). Los dedos no se tocan. Piernas y tronco no se mueven.
  4. Se escribe <nombre>.glb y <nombre>.blend (texturas embebidas), <nombre>_reposo.json (giros, articulaciones en T, estiramiento
     de aristas, angulos medidos sobre la malla antes/despues).
Imprime 'REPOSO {resumen}'."""
import json
import math
import os
import sys
import time

import bpy
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "..", "cli"))
from assetgen_cli import pose_geom as G  # noqa: E402
import articulaciones as ART  # noqa: E402

a = sys.argv[sys.argv.index("--") + 1:]


def arg(n, d=None):
    return a[a.index(n) + 1] if n in a else d


T0 = time.time()


def log(*m):
    print("[%5.1fs]" % (time.time() - T0), *m, flush=True)


def np_mat(m):
    return np.array(m, float)


def afin(R, pivote_antes, pivote_despues=None):
    """Mapa afin p -> R (p - c) + c2. Devuelve (R 3x3 np, t np)."""
    c = np.asarray(pivote_antes, float)
    c2 = c if pivote_despues is None else np.asarray(pivote_despues, float)
    R = np_mat(R)
    return R, c2 - R @ c


def comp(f, g):
    """f o g (primero g)."""
    return f[0] @ g[0], f[0] @ g[1] + f[1]


def aplica(f, p):
    return np.asarray(p, float) @ f[0].T + f[1]


def cuaternion(R):
    """Matriz 3x3 -> (w,x,y,z)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    return np.array(q)


def qmul(a_, b_):
    w1, x1, y1, z1 = a_
    w2, x2, y2, z2 = b_
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def dq_de(f):
    """(R,t) -> dual quaternion (real, dual)."""
    qr = cuaternion(f[0])
    qd = 0.5 * qmul(np.array([0, *f[1]]), qr)
    return qr, qd


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def suave_np(x, a0, a1):
    t = np.clip((x - a0) / (a1 - a0), 0, 1)
    return t * t * (3 - 2 * t)


def distancia_poligonal(P, pts):
    """Distancia minima de cada punto de P a la poligonal pts y longitud de arco (desde pts[0]) del punto mas cercano."""
    mejor = np.full(len(P), np.inf)
    lon = np.zeros(len(P))
    acum = 0.0
    for a_, b_ in zip(pts[:-1], pts[1:]):
        d = b_ - a_
        L = float(np.linalg.norm(d))
        t = np.clip(((P - a_) @ d) / (L * L), 0, 1)
        dd = np.linalg.norm(P - (a_ + t[:, None] * d), axis=1)
        m = dd < mejor
        mejor[m] = dd[m]
        lon[m] = acum + t[m] * L
        acum += L
    return mejor, lon


def armonica(nv, ea, eb, s0, s1, iters=2500, tol=1e-5):
    """Campo armonico en el grafo (u = 0 en s0, 1 en s1) por gradiente conjugado; Laplaciano combinatorio."""
    deg = np.bincount(ea, minlength=nv).astype(np.float64)
    libre = ~(s0 | s1)
    x = np.zeros(nv)
    x[s1] = 1.0

    def Lm(v):
        return deg * v - np.bincount(ea, weights=v[eb], minlength=nv)
    r = -Lm(x)
    r[~libre] = 0
    p = r.copy()
    rs = float(r @ r)
    objetivo = tol * math.sqrt(max(1, int(libre.sum())))
    for it in range(iters):
        Ap = Lm(p)
        Ap[~libre] = 0
        den = float(p @ Ap)
        if den <= 0:
            break
        al = rs / den
        x += al * p
        r -= al * Ap
        rn = float(r @ r)
        if math.sqrt(rn) < objetivo:
            break
        p = r + (rn / rs) * p
        rs = rn
    return np.clip(x, 0, 1)


def piel(V, W, mapas, modo):
    """W: n x nb (filas suman 1); mapas: lista de (R,t) por hueso. Piel lineal o por cuaterniones duales; solo se calculan los vertices
    que tienen peso en algun hueso que se mueve."""
    mueve = [i for i, f in enumerate(mapas) if not (np.allclose(f[0], np.eye(3), atol=1e-12) and np.allclose(f[1], 0, atol=1e-12))]
    sel = np.where(W[:, mueve].sum(1) > 1e-6)[0]
    if len(sel) == 0:
        raise RuntimeError('ningun vertice pesa en los huesos que se mueven: fallaron los pesos automaticos')
    out = V.copy()
    Vs, Ws = V[sel].astype(np.float64), W[sel].astype(np.float64)
    if modo == "lbs":
        acc = np.zeros_like(Vs)
        for i, f in enumerate(mapas):
            if Ws[:, i].max() > 0:
                acc += Ws[:, i:i + 1] * (Vs @ f[0].T + f[1])
        out[sel] = acc
        return out
    qr = np.zeros((len(sel), 4))
    qd = np.zeros((len(sel), 4))
    ref = None
    for i, f in enumerate(mapas):
        if Ws[:, i].max() <= 0:
            continue
        r, d = dq_de(f)
        if ref is None:
            ref = r
        sg = -1.0 if np.dot(r, ref) < 0 else 1.0
        qr += Ws[:, i:i + 1] * (sg * r)
        qd += Ws[:, i:i + 1] * (sg * d)
    n = np.linalg.norm(qr, axis=1, keepdims=True)
    qr /= n
    qd /= n
    w, x, y, z = qr.T
    # rotacion desde el cuaternion normalizado
    Rm = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                   2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                   2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)
    # traslacion = 2 * qd * conj(qr)
    cw, cx, cy, cz = w, -x, -y, -z
    dw, dx, dy, dz = qd.T
    tx = 2 * (dw * cx + dx * cw + dy * cz - dz * cy)
    ty = 2 * (dw * cy - dx * cz + dy * cw + dz * cx)
    tz = 2 * (dw * cz + dx * cy - dy * cx + dz * cw)
    out[sel] = np.einsum("nij,nj->ni", Rm, Vs) + np.stack([tx, ty, tz], 1)
    return out


def main():
    glb, artj, salida = arg("--glb"), arg("--articulaciones"), arg("--salida")
    nombre = arg("--nombre", "modelo_T")
    ang_obj = float(arg("--angulo", "-1.3"))
    f_cla = float(arg("--clavicula", "0.25"))
    iters = int(arg("--suavizado", "6"))
    semilla_brazo = float(arg("--semilla-brazo", "0.18"))
    ancho_codo = float(arg("--ancho-codo", "0.07"))
    ancho_muneca = float(arg("--ancho-muneca", "0.03"))
    modo_piel = arg("--piel", "dqs")
    os.makedirs(salida, exist_ok=True)
    J = json.load(open(artj))
    jn = {k: np.array(v, float) for k, v in J["articulaciones"].items()}
    ex = {k: np.array(v, float) for k, v in J["extras"].items() if isinstance(v, list) and len(v) == 3}
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb)
    obj = max((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: len(o.data.vertices))
    me = obj.data
    n = len(me.vertices)
    co = np.empty(n * 3, np.float32)
    me.vertices.foreach_get("co", co)
    V0 = co.reshape(-1, 3).astype(np.float64)
    me.calc_loop_triangles()
    tri = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", tri)
    tri = tri.reshape(-1, 3)
    log("malla", n, "vertices", len(tri), "triangulos")
    altura = float(V0[:, 2].max() - V0[:, 2].min())
    # --- 1. malla soldada (los vertices duplicados por UV/material comparten indice canonico) y su grafo
    q = np.round(V0 / 1e-5).astype(np.int64)
    _, canon, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    nc = len(canon)
    Vc = V0[canon]
    e = np.concatenate([inv[tri[:, [0, 1]]], inv[tri[:, [1, 2]]], inv[tri[:, [2, 0]]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    e = e[e[:, 0] != e[:, 1]]
    ea, eb = np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]])
    log("malla soldada", nc, "vertices", len(e), "aristas")
    nombres = ["tronco"] + [p + s for s in ("_I", "_D") for p in ("clav", "brazo", "antebrazo", "mano")]
    col = {nm: i for i, nm in enumerate(nombres)}
    cy = float(jn["cuello"][1])
    huesos = {"clav_" + s: (np.array([0.0, jn["hombro_" + s][1], jn["hombro_" + s][2] - 0.02]),) for s in ("I", "D")}
    # --- 2. pesos: campo armonico en el grafo (brazo = 1, tronco = 0) y reparto por tramos del brazo
    Wc = np.zeros((nc, len(nombres)), np.float32)
    Wc[:, col["tronco"]] = 1.0
    campos = {}
    for s, sg in (("I", 1), ("D", -1)):
        S, E, Wr, Hp = jn["hombro_" + s], jn["codo_" + s], jn["muneca_" + s], ex["mano_punta_" + s]
        X = Vc * np.array([sg, 1.0, 1.0])
        Sx, Ex, Wx, Hx = (p * np.array([sg, 1.0, 1.0]) for p in (S, E, Wr, Hp))
        dist, lon = distancia_poligonal(X, [Sx, Ex, Wx, Hx])
        s1 = (lon >= semilla_brazo) & (dist <= 0.125) & (X[:, 0] > 0.215)
        s0 = ((np.abs(X[:, 0]) < 0.13) | (X[:, 2] < 0.7) | (X[:, 0] < -0.13)) & ~s1
        u = armonica(nc, ea, eb, s0, s1)
        campos[s] = u
        log("campo armonico", s, "semillas", int(s1.sum()), int(s0.sum()), "u medio brazo %.2f" % float(u[s1].mean()))
        # reparto de u entre torso / clavicula / brazo y de brazo entre brazo, antebrazo y mano
        mitad = np.clip(2 * u, 0, 1)
        t_cl = mitad * mitad * (3 - 2 * mitad)
        arm_t = np.clip(2 * u - 1, 0, 1)
        arm_t = arm_t * arm_t * (3 - 2 * arm_t)
        cl = np.where(u < 0.5, t_cl, 1 - arm_t)
        torso = np.where(u < 0.5, 1 - t_cl, 0.0)
        cl = np.where(u < 0.5, t_cl, 1 - arm_t)
        n_el = unit(E - S) + unit(Wr - E)
        n_mu = unit(Wr - E) + unit(Hp - Wr)
        f_el = suave_np((Vc - E) @ n_el, -ancho_codo, ancho_codo)
        f_mu = suave_np((Vc - Wr) @ n_mu, -ancho_muneca, ancho_muneca)
        # (en el espejo D los normales ya estan en coordenadas reales: se usan puntos reales, no espejados)
        Wc[:, col["tronco"]] = np.minimum(Wc[:, col["tronco"]], torso)
        Wc[:, col["clav_" + s]] = cl
        Wc[:, col["brazo_" + s]] = arm_t * (1 - f_el)
        Wc[:, col["antebrazo_" + s]] = arm_t * f_el * (1 - f_mu)
        Wc[:, col["mano_" + s]] = arm_t * f_el * f_mu
    suma = Wc.sum(1, keepdims=True)
    Wc = (Wc / np.maximum(suma, 1e-9)).astype(np.float32)
    grado = np.bincount(ea, minlength=nc).astype(np.float32)
    grado[grado == 0] = 1
    for _ in range(iters):
        for c in range(1, len(nombres)):
            sv = np.bincount(ea, weights=Wc[eb, c], minlength=nc).astype(np.float32)
            Wc[:, c] = 0.5 * Wc[:, c] + 0.5 * sv / grado
        Wc[:, 0] = np.maximum(0, 1 - Wc[:, 1:].sum(1))
    Wc = Wc / np.maximum(Wc.sum(1, keepdims=True), 1e-9)
    log("pesos repartidos y suavizados", iters, "iteraciones")
    W = Wc[inv]
    sin_peso = 0
    npv = nc
    # --- 3. pose
    mapas = [(np.eye(3), np.zeros(3)) for _ in nombres]
    T = {}
    medido = {}
    giros = {}
    for s, sg in (("I", 1), ("D", -1)):
        S, E, Wr, Hp = jn["hombro_" + s], jn["codo_" + s], jn["muneca_" + s], ex["mano_punta_" + s]
        objetivo = np.array(G.direccion_horizontal(sg, ang_obj))
        d1 = E - S
        ang_antes = G.angulo_con_vertical_abajo(list(d1))
        # clavicula: giro parcial del brazo superior alrededor del esternon (eje de la rotacion total)
        R_tot = np_mat(G.rot_entre(list(d1), list(objetivo)))
        eje = np.cross(d1, objetivo)
        ang_tot = math.atan2(np.linalg.norm(eje), float(d1 @ objetivo))
        C0 = huesos["clav_" + s][0]
        Fc = afin(G.rot_eje_angulo(list(eje), f_cla * ang_tot), C0)
        S1, E1, W1, H1 = (aplica(Fc, x) for x in (S, E, Wr, Hp))
        Ra = np_mat(G.rot_entre(list(E1 - S1), list(objetivo)))
        Fa = comp(afin(Ra, S1), Fc)
        E2 = aplica(Fa, E)
        W2 = aplica(Fa, Wr)
        Rf = np_mat(G.rot_entre(list(W2 - E2), list(objetivo)))
        Ff = comp(afin(Rf, E2), Fa)
        W3 = aplica(Ff, Wr)
        H3 = aplica(Ff, Hp)
        Rm = np_mat(G.rot_entre(list(H3 - W3), list(objetivo)))
        Fm = comp(afin(Rm, W3), Ff)
        mapas[col["clav_" + s]], mapas[col["brazo_" + s]] = Fc, Fa
        mapas[col["antebrazo_" + s]], mapas[col["mano_" + s]] = Ff, Fm
        T["hombro_" + s], T["codo_" + s], T["muneca_" + s] = aplica(Fa, S), aplica(Ff, E), aplica(Fm, Wr)
        medido["mano_centro_" + s] = aplica(Fm, ex["mano_centro_" + s])
        medido["mano_punta_" + s] = aplica(Fm, Hp)
        giros[s] = {"brazo_antes_deg_vertical": round(ang_antes, 2), "giro_total_deg": round(math.degrees(ang_tot), 2),
                    "clavicula_deg": round(math.degrees(f_cla * ang_tot), 2),
                    "brazo_deg": round(math.degrees(math.acos(max(-1, min(1, (Ra[0, 0] + Ra[1, 1] + Ra[2, 2] - 1) / 2)))), 2),
                    "antebrazo_deg": round(math.degrees(math.acos(max(-1, min(1, (Rf[0, 0] + Rf[1, 1] + Rf[2, 2] - 1) / 2)))), 2),
                    "mano_deg": round(math.degrees(math.acos(max(-1, min(1, (Rm[0, 0] + Rm[1, 1] + Rm[2, 2] - 1) / 2)))), 2)}
    V1 = piel(V0, W, mapas, modo_piel)
    log("piel aplicada (%s); max desplazamiento %.3f m" % (modo_piel, np.abs(V1 - V0).max()))
    # --- verificacion
    el = np.linalg.norm(V0[e[:, 0] * 0 + canon[e[:, 0]]] - V0[canon[e[:, 1]]], axis=1)
    el1 = np.linalg.norm(V1[canon[e[:, 0]]] - V1[canon[e[:, 1]]], axis=1)
    r = el1 / np.maximum(el, 1e-9)
    ok = el > 2e-4
    estir = {"aristas": int(ok.sum()), "ratio_p01": round(float(np.percentile(r[ok], 0.1)), 3), "ratio_p999": round(float(np.percentile(r[ok], 99.9)), 3),
             "ratio_min": round(float(r[ok].min()), 3), "ratio_max": round(float(r[ok].max()), 3),
             "aristas_estiradas_mas_de_1.5": int((r[ok] > 1.5).sum()), "aristas_comprimidas_menos_de_0.6": int((r[ok] < 0.6).sum())}
    # normales y escritura de la malla
    me.vertices.foreach_set("co", V1.astype(np.float32).reshape(-1))
    me.update()
    mala = ok & ((r > 1.6) | (r < 0.55))
    mid_ = (V0[canon[e[:, 0]]] + V0[canon[e[:, 1]]]) / 2
    celdas, cuenta = np.unique(np.round(mid_[mala] / 0.05).astype(int), axis=0, return_counts=True)
    o_ = np.argsort(-cuenta)[:8]
    estir["focos_pose_A_celdas_5cm"] = [{"centro": [round(float(x) * 0.05, 3) for x in celdas[i]], "aristas": int(cuenta[i])} for i in o_]
    # normales: las importadas quedaron viejas; se recalculan suaves sobre la malla soldada (los duplicados por UV comparten normal)
    fn = np.cross(V1[tri[:, 1]] - V1[tri[:, 0]], V1[tri[:, 2]] - V1[tri[:, 0]])
    Nc = np.zeros((nc, 3))
    for k in range(3):
        np.add.at(Nc, inv[tri[:, k]], fn)
    Nc /= np.maximum(np.linalg.norm(Nc, axis=1, keepdims=True), 1e-12)
    me.normals_split_custom_set_from_vertices(Nc[inv].tolist())
    # mediciones sobre la malla posada con el mismo analisis que el original
    med = {"antes": {}, "despues": {}}
    for lado, s in ((1, "I"), (-1, "D")):
        med["antes"][s] = ART.angulos_brazo(V0, lado, altura)
        med["despues"][s] = ART.angulos_brazo(V1, lado, altura)
    res = {k: ART.pt(v) for k, v in jn.items()}
    res.update({k: ART.pt(v) for k, v in T.items()})
    out = {"unidades": "m", "ejes": J.get("ejes"), "altura_m": round(float((V1[:, 2].max() - V1[:, 2].min())), 4),
           "metodo": "pose A -> T con pesos por campo armonico en el grafo de la malla, clavicula %.2f del giro, piel %s; las "
                     "articulaciones en T son las de pose A llevadas por la cadena clavicula-brazo-antebrazo-mano" % (f_cla, modo_piel),
           "angulo_objetivo_sobre_horizontal_deg": ang_obj,
           "articulaciones": {k: res[k] for k in ART.CLAVES},
           "extras": {**{k: ART.pt(v) for k, v in medido.items()}, **{k: J["extras"][k] for k in J["extras"] if k.startswith("ancho_cuello")}},
           "giros": giros, "estiramiento_aristas": estir, "angulos_brazo_medidos_en_malla": med}
    json.dump(out, open(os.path.join(salida, "articulaciones_nuevo_T.json"), "w"), indent=1, ensure_ascii=False)
    if "--no-exportar" not in a:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(salida, nombre + ".blend"), compress=True)
        for o in bpy.context.view_layer.objects:
            o.select_set(False)
        obj.select_set(True)
        bpy.ops.export_scene.gltf(filepath=os.path.join(salida, nombre + ".glb"), export_format="GLB", use_selection=True)
        log("exportado")
    print("REPOSO", json.dumps({"giros": giros, "estiramiento": estir, "angulos": med}))


if __name__ == "__main__":
    main()
