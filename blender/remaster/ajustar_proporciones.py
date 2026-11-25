"""ajustar_proporciones.py -- deforma un personaje en pose T para que sus referencias de proporcion caigan sobre las de un original
(escala uniforme global + ajuste por segmentos con mezcla suave) y lleva a la malla ajustada los pivotes del juego.

    blender -b -P ajustar_proporciones.py -- --nuevo modelo_T.glb --original original_bind.glb --articulaciones-original art_orig.json
        --alineacion alineacion.json --salida DIR [--nombre modelo_T_ajustado] [--escala S] [--fade-pierna 0.30] [--suavizado 6]
        [--max-cambio 15] [--solo-medir]

Referencias: `referencias.py` (secciones transversales, el mismo algoritmo en las dos mallas). Deformacion (todo en coordenadas ya
escaladas por S, la escala uniforme global; por defecto la de alineacion.json):
  * cuerpo (tronco, cabeza, piernas): mapa vertical lineal por tramos z -> z' con nudos suelo, tobillo, bombacho, sisa y hombro alto
    (pendiente 1 fuera: la cabeza solo se traslada); desplazamiento en y del tronco y de la cabeza; escala lateral (x) del pecho
    entre cintura y hombro; cada pierna se desplaza en x/y para que tobillo y borde del bombacho caigan en su sitio y el
    desplazamiento se apaga `--fade-pierna` m por encima del bombacho (mezcla en x entre las dos perneras sobre el plano medio).
  * brazos: huesos brazo (brazo_ini -> fin_manga), antebrazo (fin_manga -> muneca) y mano (muneca -> punta; solo giro y traslacion, sin
    estirar). Cada hueso lleva su tramo sobre el del original (giro de arco minimo + estiramiento axial). Pesos por campo armonico en
    el grafo de la malla (mismos que `reposar_T.py`, reutilizando sus funciones) con la clavicula a medias entre tronco y brazo.
Si algun tramo cambia de largo mas de `--max-cambio` % sobre la escala global, se imprime 'PARADA ...' y no se aplica nada.
Escribe <nombre>.glb/.blend, referencias.json, articulaciones_juego_en_nuevo.json. Imprime 'AJUSTE {resumen}'."""
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
from assetgen_cli import ajuste_geom as AG  # noqa: E402
import referencias as REF  # noqa: E402
import reposar_T as RT  # noqa: E402  (solo sus funciones: armonica, distancia_poligonal, suave_np)

a = sys.argv[sys.argv.index("--") + 1:]
T0 = time.time()


def arg(n, d=None):
    return a[a.index(n) + 1] if n in a else d


def log(*m):
    print("[%5.1fs]" % (time.time() - T0), *m, flush=True)


NOMBRES = ["tronco"] + [p + s for s in ("_I", "_D") for p in ("clav", "brazo", "antebrazo", "mano")]
PARES_JUEGO = {"cuello": "hombro_alto", "hombro_I": "brazo_ini_I", "codo_I": "fin_manga_I", "muneca_I": "muneca_I",
               "cadera_I": "bombacho_I", "rodilla_I": "bombacho_I", "tobillo_I": "tobillo_I",
               "hombro_D": "brazo_ini_D", "codo_D": "fin_manga_D", "muneca_D": "muneca_D",
               "cadera_D": "bombacho_D", "rodilla_D": "bombacho_D", "tobillo_D": "tobillo_D"}
# referencias informativas (no se ajustan): coronilla y punta de la mano (cabeza y mano conservan su forma), cintura (valle plano, +-3 cm de
# incertidumbre al medir) y los centros en y que solo sirven de apoyo a la deformacion
INFORMATIVAS = ("coronilla_z", "punta_mano_I", "punta_mano_D", "cabeza_y", "torso_y", "cintura", "cintura_z")


def cargar(ruta):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=ruta)
    obj = max((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: len(o.data.vertices))
    me = obj.data
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    return obj, co.reshape(-1, 3).astype(np.float64), REF.triangulos(me)


def con_derivadas(refs):
    """Referencias aplanadas + cintura y hombro_alto como puntos 3D (para emparejar con las articulaciones del juego)."""
    f = REF.aplanar(refs)
    f["cintura"] = [0.0, f["torso_y"], f["cintura_z"]]
    f["hombro_alto"] = [0.0, f["cabeza_y"], (f["hombro_alto_z_I"] + f["hombro_alto_z_D"]) / 2]
    return f


def z_central(f, k):
    """Valor z de una referencia central o promedio de los dos lados."""
    if k in f:
        return f[k]
    if k + "_I" in f:
        return (f[k + "_I"] + f[k + "_D"]) / 2
    return (f[k.replace("_z", "") + "_I"][2] + f[k.replace("_z", "") + "_D"][2]) / 2


def pesos_segmentos(Vc, ea, eb, fn, iters):
    """Pesos (nc x 9) tronco / clavicula, brazo, antebrazo, mano por lado, con las referencias del brazo como articulaciones."""
    nc = len(Vc)
    Wc = np.zeros((nc, len(NOMBRES)), np.float32)
    Wc[:, 0] = 1.0
    col = {nm: i for i, nm in enumerate(NOMBRES)}
    for s, sg in (("I", 1.0), ("D", -1.0)):
        S, E, Wr, Hp = (np.array(fn[k + "_" + s]) for k in ("brazo_ini", "fin_manga", "muneca", "punta_mano"))
        M = np.array([sg, 1.0, 1.0])
        dist, lon = RT.distancia_poligonal(Vc * M, [S * M, E * M, Wr * M, Hp * M])
        x_sem = fn["semiancho_pecho_" + s] + 0.13
        s1 = (dist <= 0.125) & ((Vc * M)[:, 0] > x_sem)
        s0 = ((np.abs(Vc[:, 0]) < 0.13) | (Vc[:, 2] < 0.7) | ((Vc * M)[:, 0] < -0.13)) & ~s1
        u = RT.armonica(nc, ea, eb, s0, s1)
        log("campo armonico", s, "semillas", int(s1.sum()), int(s0.sum()))
        mitad = np.clip(2 * u, 0, 1)
        t_cl = mitad * mitad * (3 - 2 * mitad)
        arm_t = np.clip(2 * u - 1, 0, 1)
        arm_t = arm_t * arm_t * (3 - 2 * arm_t)
        cl = np.where(u < 0.5, t_cl, 1 - arm_t)
        torso = np.where(u < 0.5, 1 - t_cl, 0.0)
        n_el = RT.unit(E - S) + RT.unit(Wr - E)
        n_mu = RT.unit(Wr - E) + RT.unit(Hp - Wr)
        f_el = RT.suave_np((Vc - E) @ n_el, -0.05, 0.05)
        f_mu = RT.suave_np((Vc - Wr) @ n_mu, -0.03, 0.03)
        Wc[:, 0] = np.minimum(Wc[:, 0], torso)
        Wc[:, col["clav_" + s]] = cl
        Wc[:, col["brazo_" + s]] = arm_t * (1 - f_el)
        Wc[:, col["antebrazo_" + s]] = arm_t * f_el * (1 - f_mu)
        Wc[:, col["mano_" + s]] = arm_t * f_el * f_mu
    Wc = (Wc / np.maximum(Wc.sum(1, keepdims=True), 1e-9)).astype(np.float32)
    grado = np.bincount(ea, minlength=nc).astype(np.float32)
    grado[grado == 0] = 1
    for _ in range(iters):
        for c in range(1, len(NOMBRES)):
            sv = np.bincount(ea, weights=Wc[eb, c], minlength=nc).astype(np.float32)
            Wc[:, c] = 0.5 * Wc[:, c] + 0.5 * sv / grado
        Wc[:, 0] = np.maximum(0, 1 - Wc[:, 1:].sum(1))
    return Wc / np.maximum(Wc.sum(1, keepdims=True), 1e-9)


def mapa_z(z, xs, ys):
    """Lineal por tramos con pendiente 1 fuera de los nudos."""
    r = np.interp(z, xs, ys)
    r = np.where(z > xs[-1], ys[-1] + (z - xs[-1]), r)
    return np.where(z < xs[0], ys[0] + (z - xs[0]), r)


def plan(fn, fo, s, fade):
    """Parametros de la deformacion (todo lo que depende solo de las referencias)."""
    P = {"s": s}
    zn = [0.0, z_central(fn, "tobillo_z"), z_central(fn, "bombacho_z"), z_central(fn, "sisa_z"), z_central(fn, "hombro_alto_z")]
    zo = [0.0, z_central(fo, "tobillo_z"), z_central(fo, "bombacho_z"), z_central(fo, "sisa_z"), z_central(fo, "hombro_alto_z")]
    zn_s = [s * z for z in zn]
    if any(zn_s[i] >= zn_s[i + 1] or zo[i] >= zo[i + 1] for i in range(4)):
        raise RuntimeError("los nudos verticales no son crecientes: " + str((zn_s, zo)))
    P["z_de"], P["z_a"] = zn_s, zo
    cint_s = s * fn["cintura_z"]
    cab_s = s * (fn["coronilla_z"] - 0.10)
    P["dy_t"] = fo["torso_y"] - s * fn["torso_y"]
    P["dy_h"] = fo["cabeza_y"] - s * fn["cabeza_y"]
    P["dy_z"] = [cint_s, cab_s]
    r = np.mean([fo["semiancho_pecho_" + k] / (s * fn["semiancho_pecho_" + k]) for k in "ID"])
    P["r_pecho"] = float(r)
    sisa_s, ha_s = zn_s[3], zn_s[4]
    P["k_z"] = [cint_s, sisa_s - 0.10, sisa_s, ha_s, ha_s + 0.06]
    P["k_v"] = [1.0, r, r, 1.0 + (r - 1.0) / 2, 1.0]
    P["pierna"] = {}
    for k in "ID":
        res = []
        for nm in ("tobillo", "bombacho"):
            xn, yn, zz = fn[nm + "_" + k]
            xo, yo, _ = fo[nm + "_" + k]
            res.append((xo - s * xn, yo - (s * yn + P["dy_t"]), s * zz))
        P["pierna"][k] = {"z": [res[0][2], res[1][2], res[1][2] + fade], "dx": [res[0][0], res[1][0], 0.0], "dy": [res[0][1], res[1][1], 0.0]}
    P["huesos"] = {}
    for k in "ID":
        pn = {nm: [s * c for c in fn[nm + "_" + k]] for nm in ("brazo_ini", "fin_manga", "muneca", "punta_mano")}
        po = {nm: fo[nm + "_" + k] for nm in pn}
        P["huesos"][k] = {
            "brazo": AG.matriz_tramo(pn["brazo_ini"], pn["fin_manga"], po["brazo_ini"], po["fin_manga"], True),
            "antebrazo": AG.matriz_tramo(pn["fin_manga"], pn["muneca"], po["fin_manga"], po["muneca"], True),
            "mano": AG.matriz_tramo(pn["muneca"], pn["punta_mano"], po["muneca"], po["punta_mano"], False)}
    return P


def deformar(V, W, P):
    """V (n x 3, sin escalar), W (n x 9) pesos -> vertices ajustados."""
    s = P["s"]
    X = V * s
    z = X[:, 2]
    B = np.empty_like(X)
    B[:, 2] = mapa_z(z, P["z_de"], P["z_a"])
    B[:, 1] = X[:, 1] + np.interp(z, P["dy_z"], [P["dy_t"], P["dy_h"]])
    B[:, 0] = X[:, 0] * np.interp(z, P["k_z"], P["k_v"])
    aI = np.clip((X[:, 0] + 0.04) / 0.08, 0, 1)
    aI = aI * aI * (3 - 2 * aI)
    for k, al in (("I", aI), ("D", 1 - aI)):
        pl = P["pierna"][k]
        B[:, 0] += al * np.interp(z, pl["z"], pl["dx"])
        B[:, 1] += al * np.interp(z, pl["z"], pl["dy"])
    col = {nm: i for i, nm in enumerate(NOMBRES)}
    out = W[:, 0:1] * B
    for k in "ID":
        mb = {h: X @ np.array(m[0]).T + np.array(m[1]) for h, m in P["huesos"][k].items()}
        wc = W[:, col["clav_" + k]][:, None]
        out += wc * 0.5 * (B + mb["brazo"])
        for h in ("brazo", "antebrazo", "mano"):
            out += W[:, col[h + "_" + k]][:, None] * mb[h]
    return out


def largos(f):
    """Largos de los tramos (m) a partir de referencias aplanadas; centrales y de lado promediados donde conviene."""
    z = lambda k: z_central(f, k)
    L = {"pie": z("tobillo_z"), "pierna": z("bombacho_z") - z("tobillo_z"), "muslo_tronco": z("sisa_z") - z("bombacho_z"),
         "hombro_a_hombro_alto": z("hombro_alto_z") - z("sisa_z"), "cabeza": f["coronilla_z"] - z("hombro_alto_z")}
    for k in "ID":
        L["brazo_" + k] = AG.largo(f["brazo_ini_" + k], f["fin_manga_" + k])
        L["antebrazo_" + k] = AG.largo(f["fin_manga_" + k], f["muneca_" + k])
        L["mano_" + k] = AG.largo(f["muneca_" + k], f["punta_mano_" + k])
        L["semiancho_pecho_" + k] = f["semiancho_pecho_" + k]
    return L


def main():
    ruta_n, ruta_o = arg("--nuevo"), arg("--original")
    salida = arg("--salida")
    nombre = arg("--nombre", "modelo_T_ajustado")
    iters = int(arg("--suavizado", "6"))
    fade = float(arg("--fade-pierna", "0.30"))
    tope = float(arg("--max-cambio", "15"))
    s = float(arg("--escala")) if arg("--escala") else float(json.load(open(arg("--alineacion")))["escala"])
    os.makedirs(salida, exist_ok=True)
    # --- original: referencias
    _, Vo, To = cargar(ruta_o)
    fo = con_derivadas(REF.medir(Vo, To))
    log("referencias del original medidas")
    # --- nuevo
    obj, V0, T = cargar(ruta_n)
    me = obj.data
    fn_raw = REF.medir(V0, T)
    fn = con_derivadas(fn_raw)
    log("referencias del nuevo medidas;", len(V0), "vertices")
    P = plan(fn, fo, s, fade)
    # cambio de largo por tramo: total y extra sobre la escala global
    Ln, Lo = largos(fn), largos(fo)
    cambio = {k: {"antes_m": round(Ln[k], 4), "original_m": round(Lo[k], 4), "cambio_total_pct": round(AG.cambio_pct(Ln[k], Lo[k]), 1),
                  "cambio_sobre_escala_global_pct": round(AG.cambio_pct(s * Ln[k], Lo[k]), 1)} for k in Ln if k != "cabeza" and not k.startswith("mano")}
    exceso = {k: v["cambio_sobre_escala_global_pct"] for k, v in cambio.items() if abs(v["cambio_sobre_escala_global_pct"]) > tope}
    if exceso:
        print("PARADA", json.dumps(exceso))
        json.dump({"parada": exceso, "cambio_largo": cambio}, open(os.path.join(salida, "referencias_parada.json"), "w"), indent=1)
        return
    if "--solo-medir" in a:
        print("AJUSTE", json.dumps({"escala": s, "cambio": cambio, "r_pecho": P["r_pecho"]}))
        return
    # --- grafo soldado y pesos
    q = np.round(V0 / 1e-5).astype(np.int64)
    _, canon, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    nc = len(canon)
    Vc = V0[canon]
    e = np.concatenate([inv[T[:, [0, 1]]], inv[T[:, [1, 2]]], inv[T[:, [2, 0]]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    e = e[e[:, 0] != e[:, 1]]
    ea, eb = np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]])
    log("malla soldada", nc, "vertices", len(e), "aristas")
    Wc = pesos_segmentos(Vc, ea, eb, fn, iters)
    log("pesos listos")
    W = Wc[inv]
    V1 = deformar(V0, W, P)
    log("deformada; desplazamiento max %.3f m" % np.abs(V1 - V0 * s).max())
    # estiramiento de aristas (sobre la escala global)
    el = np.linalg.norm(V0[canon[e[:, 0]]] - V0[canon[e[:, 1]]], axis=1) * s
    el1 = np.linalg.norm(V1[canon[e[:, 0]]] - V1[canon[e[:, 1]]], axis=1)
    ok = el > 2e-4
    r = el1[ok] / el[ok]
    estir = {"aristas": int(ok.sum()), "ratio_p01": round(float(np.percentile(r, 0.1)), 3), "ratio_p999": round(float(np.percentile(r, 99.9)), 3),
             "ratio_min": round(float(r.min()), 3), "ratio_max": round(float(r.max()), 3),
             "aristas_estiradas_mas_de_1.5": int((r > 1.5).sum()), "aristas_comprimidas_menos_de_0.6": int((r < 0.6).sum())}
    # --- escribir malla y normales
    me.vertices.foreach_set("co", V1.astype(np.float32).reshape(-1))
    me.update()
    fnrm = np.cross(V1[T[:, 1]] - V1[T[:, 0]], V1[T[:, 2]] - V1[T[:, 0]])
    Nc = np.zeros((nc, 3))
    for k in range(3):
        np.add.at(Nc, inv[T[:, k]], fnrm)
    Nc /= np.maximum(np.linalg.norm(Nc, axis=1, keepdims=True), 1e-12)
    me.normals_split_custom_set_from_vertices(Nc[inv].tolist())
    # --- referencias despues
    fa = con_derivadas(REF.medir(V1, T))
    log("referencias del ajustado medidas")
    La = largos(fa)
    claves = [k for k in fo if k not in INFORMATIVAS and k in fa]
    err = AG.errores_cm(fa, fo, claves)
    err_inf = AG.errores_cm(fa, fo, [k for k in INFORMATIVAS if k in fo])
    for k in cambio:
        cambio[k]["despues_m"] = round(La[k], 4)
        cambio[k]["error_vs_original_cm"] = round((La[k] - Lo[k]) * 100, 2)
    rd = {"escala_global": s, "unidades": "m", "ejes": "z arriba, frente -Y, I = +X, D = -X",
          "metodo": "secciones transversales (referencias.py), mismo algoritmo en las dos mallas; ver cabecera de ajustar_proporciones.py",
          "referencias": {k: {"nuevo_antes": fn[k], "nuevo_antes_escalado": ([s * c for c in fn[k]] if isinstance(fn[k], list) else s * fn[k]),
                              "original": fo[k], "nuevo_despues": fa[k], "error_despues_cm": (err.get(k) if k in err else err_inf.get(k)),
                              "informativa": k in INFORMATIVAS} for k in sorted(fo) if k in fa},
          "error_max_cm": max(err.values()), "error_rms_cm": round(math.sqrt(sum(v * v for v in err.values()) / len(err)), 2),
          "fuera_de_1.5cm": sorted(k for k, v in err.items() if v > 1.5),
          "cambio_largo_por_tramo": cambio, "escala_lateral_pecho": round(P["r_pecho"], 4), "estiramiento_aristas": estir}
    json.dump(rd, open(os.path.join(salida, "referencias.json"), "w"), indent=1, ensure_ascii=False)
    # --- pivotes del juego en el nuevo ajustado
    jo = json.load(open(arg("--articulaciones-original")))["articulaciones"]
    des = AG.desfases(jo, fo, PARES_JUEGO)
    jn = AG.trasladar_juego(fa, des, PARES_JUEGO)
    jd = {"unidades": "m", "ejes": "z arriba, frente -Y, I = +X, D = -X",
          "metodo": "articulacion del juego en el nuevo = referencia medida en el nuevo ajustado + (articulacion por pesos del original - misma referencia en el original)",
          "pares_articulacion_referencia": PARES_JUEGO,
          "articulaciones": {k: [round(x, 4) for x in v] for k, v in jn.items()},
          "desfase_original_articulacion_menos_referencia_m": {k: [round(x, 4) for x in v] for k, v in des.items()},
          "articulaciones_original": {k: jo[k] for k in jn},
          "diferencia_con_original_cm": {k: round(AG.largo(jn[k], jo[k]) * 100, 2) for k in jn}}
    json.dump(jd, open(os.path.join(salida, "articulaciones_juego_en_nuevo.json"), "w"), indent=1, ensure_ascii=False)
    if "--no-exportar" not in a:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(salida, nombre + ".blend"), compress=True)
        for o in bpy.context.view_layer.objects:
            o.select_set(False)
        obj.select_set(True)
        bpy.ops.export_scene.gltf(filepath=os.path.join(salida, nombre + ".glb"), export_format="GLB", use_selection=True)
        log("exportado")
    print("AJUSTE", json.dumps({"escala": s, "error_max_cm": rd["error_max_cm"], "error_rms_cm": rd["error_rms_cm"], "fuera": rd["fuera_de_1.5cm"],
                                "estiramiento": estir, "informativas": err_inf}))


if __name__ == "__main__":
    main()
