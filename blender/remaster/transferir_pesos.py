"""transferir_pesos.py -- pesos de piel PROVISIONALES: del modelo original (grupos `hNN`) al proxy ligero de un modelo nuevo.

    blender -b -P transferir_pesos.py -- --proxy proxy.blend --original original_bind.glb --articulaciones art_juego_en_nuevo.json
        --salida proxy_pesos.blend [--sueltas "daga=0-3,cabeza=4-14,faldon=44-47"] [--cadera 15] [--ancho-manga 0.16]
        [--pose-a A.json --pose-b B.json --enlace E.json [--tol-grados 3 --tol-m 0.02]] [--masa-min 10] [--cerca-daga 0.15] [--umbral-daga 0.03] [--min-daga 60] [--radio-daga 0.035] [--iteraciones 8] [--lambda 0.5]
        [--z-pierna Z] [--sin-piernas] [--banda-entrepierna 0.08] [--caida-entrepierna 0.08] [--peso-cadera 0.5] [--masa-min-pierna 35] [--iteraciones-cuerpo 30] [--iteraciones-hombro 12] [--transicion-manga 0.08]

Metodo
  1. Segmentos del ORIGINAL: mapa hueso -> segmento (`pesos_geom.mapa_hueso_segmento`) derivado de las piezas sueltas (`--sueltas`) y de
     la posicion de los vertices de cada hueso; cada vertice/triangulo toma el segmento de su hueso dominante.
  2. Segmentos del NUEVO (proxy, un vertice = un segmento): material de la cabeza (atributo `parte` = 1) -> cabeza/capucha/mechones;
     manga, antebrazo y mano por la distancia a la poligonal hombro-codo-muneca-mano y |x| mayor que el hombro; mano = mas alla de
     la muneca; daga = protuberancia del cuerpo (lejos de toda la superficie original) al nivel de la cintura; el resto, cuerpo.
  3. Pesos: punto mas cercano SOLO sobre los triangulos del original de los segmentos compatibles (cabeza+capucha | cuerpo | brazo del
     mismo lado), interpolacion baricentrica y descarte de huesos incompatibles (la manga nunca toma el torso; los mechones y la capucha
     nunca el cuello del cuerpo). Daga: huesos de la daga del original si queda a menos de `--cerca-daga` m, si no 100 % cadera.
     Mano (dedos incluidos): 100 % el hueso de la palma (el de la mano con mas peso total mas alla de la muneca).
  3b. Piernas: los huesos de cuerpo por debajo de `--z-pierna` (por defecto, la cadera + 6 cm) son de pierna_I (x > 0) o pierna_D; cada
     vertice del cuerpo se transfiere tres veces (solo cadera/columna; cadera + pierna I; cadera + pierna D) y se combina con
     `pesos_geom.pesos_entrepierna`: bajo la entrepierna cada pierna es un segmento puro separado por el plano x = 0 y, en la banda
     |x| < `--banda-entrepierna` por encima de ella, se mezclan la cadera (hasta `--peso-cadera`) y las dos piernas, de forma continua.
     Raiz de la manga: en la franja de `--transicion-manga` m a cada lado del limite torso/manga se mezclan los pesos de brazo (hombro +
     brazo) y de torso con un escalon suave, en vez del salto de segmento.
  4. Suavizado en el grafo del proxy entre vertices del mismo segmento (los de la mano mas alla de la muneca quedan fijos), <=3 pesos.
Escribe los grupos hNN en el proxy y un JSON `<salida>.json` con el mapa hueso->segmento y las cuentas. Imprime 'PESOS {resumen}'."""
import json
import os
import re
import sys
import time

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "..", "cli"))
from assetgen_cli import pesos_geom as PG  # noqa: E402

a = sys.argv[sys.argv.index("--") + 1:]


def arg(n, d=None):
    return a[a.index(n) + 1] if n in a else d


T0 = time.time()


def log(*m):
    print("[%5.1fs]" % (time.time() - T0), *m, flush=True)


def rango(s):
    out = []
    for t in s.split("+"):
        lo, _, hi = t.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def leer_original(ruta):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=ruta)
    ob = next(o for o in bpy.context.scene.objects if o.type == "MESH" and o.vertex_groups)
    me = ob.data
    mw = np.array(ob.matrix_world)
    n = len(me.vertices)
    co = np.empty(n * 3); me.vertices.foreach_get("co", co)
    P = co.reshape(-1, 3) @ mw[:3, :3].T + mw[:3, 3]
    nombres = {i: int(re.fullmatch(r"h(\d+)", vg.name).group(1)) for i, vg in enumerate(ob.vertex_groups) if re.fullmatch(r"h(\d+)", vg.name)}
    W = []
    for v in me.vertices:
        d = {}
        for e in v.groups:
            if e.group in nombres and e.weight > 0:
                d[nombres[e.group]] = d.get(nombres[e.group], 0.0) + e.weight
        W.append(PG.normalizar(d))
    me.calc_loop_triangles()
    t = np.empty(len(me.loop_triangles) * 3, np.int64); me.loop_triangles.foreach_get("vertices", t)
    return P, W, t.reshape(-1, 3)


def dist_poligonal(P, pts):
    """Para cada punto: distancia a la poligonal pts, indice del tramo y parametro t (0..1) del pie en ese tramo."""
    mejor = np.full(len(P), np.inf); tramo = np.zeros(len(P), int); par = np.zeros(len(P))
    for k in range(len(pts) - 1):
        A, B = np.array(pts[k], float), np.array(pts[k + 1], float)
        AB = B - A
        t = np.clip(((P - A) @ AB) / max(AB @ AB, 1e-12), 0.0, 1.0)
        d = np.linalg.norm(P - (A + t[:, None] * AB), axis=1)
        m = d < mejor
        mejor[m], tramo[m], par[m] = d[m], k, t[m]
    return mejor, tramo, par


def main():
    ruta_proxy, ruta_orig, ruta_art, salida = arg("--proxy"), arg("--original"), arg("--articulaciones"), arg("--salida")
    if not (ruta_proxy and ruta_orig and ruta_art and salida):
        raise SystemExit("faltan --proxy --original --articulaciones --salida")
    sueltas = {}
    for t in arg("--sueltas", "daga=0-3;cabeza=4-14;faldon=44-47").replace(",", ";").split(";"):
        k, v = t.split("=")
        sueltas[k] = rango(v)
    cadera = int(arg("--cadera", 15)); ancho = float(arg("--ancho-manga", 0.16)); cerca_daga = float(arg("--cerca-daga", 0.15))
    umbral_daga = float(arg("--umbral-daga", 0.03)); min_daga = int(arg("--min-daga", 60))
    radio_daga = float(arg("--radio-daga", 0.035))
    iters = int(arg("--iteraciones", 8)); lam = float(arg("--lambda", 0.5)); iters_cuerpo = int(arg("--iteraciones-cuerpo", 30))
    iters_hombro = int(arg("--iteraciones-hombro", 12))
    art = json.load(open(ruta_art))["articulaciones"]
    z_cad = (art["cadera_I"][2] + art["cadera_D"][2]) / 2
    con_piernas = "--sin-piernas" not in a
    z_pierna = float(arg("--z-pierna", z_cad + 0.06)) if con_piernas else None
    banda_ent, caida_ent = float(arg("--banda-entrepierna", 0.08)), float(arg("--caida-entrepierna", 0.08))
    peso_cad, trans_manga = float(arg("--peso-cadera", 0.5)), float(arg("--transicion-manga", 0.08))

    # ---------------- 1. original
    Po, Wo, To = leer_original(ruta_orig)
    nb = 1 + max(h for w in Wo for h in w)
    masa = np.zeros(nb); cen = np.zeros((nb, 3))
    for i, w in enumerate(Wo):
        for h, x in w.items():
            masa[h] += x; cen[h] += x * Po[i]
    centroides = {h: tuple((cen[h] / masa[h]).tolist()) for h in range(nb) if masa[h] > 1e-6}
    x_h = abs(art["hombro_I"][0])
    mapa = PG.mapa_hueso_segmento(sueltas, centroides, x_h, art["hombro_I"][2], z_pierna=z_pierna)
    log("mapa hueso->segmento: " + json.dumps({s: PG.huesos_de(mapa, s) for s in PG.SEGMENTOS}))
    dom = np.array([max(w, key=w.get) if w else 0 for w in Wo])
    seg_v = np.array([mapa.get(int(h), "cuerpo") for h in dom])
    # triangulos del original por segmento (mayoria de sus vertices)
    seg_t = []
    for t in To:
        s = list(seg_v[t])
        seg_t.append(max(set(s), key=s.count))
    seg_t = np.array(seg_t)

    def bvh_de(segs):
        sel = np.where(np.isin(seg_t, segs))[0]
        if len(sel) == 0:
            sel = np.arange(len(To))
        return BVHTree.FromPolygons([tuple(p) for p in Po.tolist()], [tuple(t) for t in To[sel].tolist()]), sel
    familias = {"cabeza": ("cabeza", "capucha"), "cuerpo": ("cuerpo",), "brazo_I": ("brazo_I",), "brazo_D": ("brazo_D",), "daga": ("daga",)}
    if con_piernas:     # cadera sola; cadera + una pierna (el vertice se transfiere con cada familia y despues se combina)
        familias.update({"pierna_I": ("cuerpo", "pierna_I"), "pierna_D": ("cuerpo", "pierna_D")})
    bvh = {k: bvh_de(v) for k, v in familias.items()}
    bvh_todo = BVHTree.FromPolygons([tuple(p) for p in Po.tolist()], [tuple(t) for t in To.tolist()])
    compat = {k: set(PG.huesos_de(mapa, *v)) for k, v in familias.items()}
    # huesos secundarios de la cabeza (ojos, orejas, mechones sueltos, faldon que oscila): se mueven mucho respecto a la cabeza y el
    # nuevo no tiene esas partes en el mismo sitio; con ellos la cara se rasga. Solo se admiten los que siguen a la cabeza como un solido.
    if arg("--pose-a") and arg("--pose-b") and arg("--enlace"):
        pm = {}
        enl = json.load(open(arg("--enlace")))
        for k in "AB":
            pose = json.load(open(arg("--pose-" + k.lower())))
            pm[k] = [PG.mul4_12(p, e) for p, e in zip(pose, enl)]
        cab = compat["cabeza"]
        ref = max(cab, key=lambda h: masa[h])
        rigidos, fuera = set(), {}
        tol_g, tol_m = float(arg("--tol-grados", 3.0)), float(arg("--tol-m", 0.02))
        for h in cab:
            mv = [PG.movimiento_relativo(pm[k][ref], pm[k][h]) for k in "AB"]
            ang, tr = max(m[0] for m in mv), max(m[1] for m in mv)
            if ang <= tol_g and tr <= tol_m:
                rigidos.add(h)
            else:
                fuera[h] = (round(ang, 1), round(tr, 3))
        log("cabeza: hueso de referencia %d; rigidos con ella %s; secundarios descartados %s" % (ref, sorted(rigidos), fuera))
        compat["cabeza"] = rigidos
        cabeza_descartados = {str(h): v for h, v in fuera.items()}
    else:
        cabeza_descartados = {}
    # huesos de cuerpo con muy poca masa de vertices (bandas sueltas, faldillas, correas, dedos de los pies...): en el original corrigen
    # detalles locales y al pasarlos a un traje distinto (bombachos anchos) dejan parches rasgados; se queda la cadena principal.
    masa_min = float(arg("--masa-min", 10.0))
    todo_cuerpo = set(PG.huesos_de(mapa, "cuerpo", "pierna_I", "pierna_D"))
    masa_min_pierna = float(arg("--masa-min-pierna", 35.0))     # huesos de falda/bombacho que bailan respecto a la pierna (43, 37, 40): la cadena principal pesa >= 40
    sec_cuerpo = {h for h in todo_cuerpo if masa[h] < (masa_min_pierna if mapa[h] in ("pierna_I", "pierna_D") else masa_min)}
    if len(compat["cuerpo"] - sec_cuerpo) >= 4:
        for k in ("cuerpo", "pierna_I", "pierna_D"):
            if k in compat:
                compat[k] -= sec_cuerpo
        log("cuerpo: huesos secundarios (masa < %.0f, piernas < %.0f) descartados: %s" % (masa_min, masa_min_pierna, sorted(sec_cuerpo)))
    else:
        sec_cuerpo = set()
    # palma: hueso del brazo con mas peso total entre los que quedan mas alla de la muneca
    palma = {}
    for lado, sg in (("I", "brazo_I"), ("D", "brazo_D")):
        xm = abs(art["muneca_" + lado][0])
        cand = [h for h in PG.huesos_de(mapa, sg) if abs(centroides[h][0]) > xm]
        palma[lado] = max(cand, key=lambda h: masa[h])
    log("palma: %s" % palma)

    # ---------------- 2. proxy
    bpy.ops.wm.open_mainfile(filepath=ruta_proxy)
    ob = bpy.data.objects[next(o.name for o in bpy.data.objects if o.type == "MESH")]
    me = ob.data
    n = len(me.vertices)
    co = np.empty(n * 3); me.vertices.foreach_get("co", co); P = co.reshape(-1, 3)
    parte = np.empty(n, np.int32); me.attributes["parte"].data.foreach_get("value", parte)
    me.calc_loop_triangles()
    Tn = np.empty(len(me.loop_triangles) * 3, np.int64); me.loop_triangles.foreach_get("vertices", Tn); Tn = Tn.reshape(-1, 3)

    seg = np.full(n, "cuerpo", object)
    seg[parte == 1] = "cabeza"
    cuerpo_m = parte == 0
    mano = np.zeros(n, bool); mano_fija = np.zeros(n, bool)
    f_man = np.zeros(n); lado_man = np.zeros(n, int)       # escalon torso -> manga (0 = torso puro, 1 = brazo puro) y lado (+1 I / -1 D)
    for lado, sg in (("I", "brazo_I"), ("D", "brazo_D")):
        h, c, m = np.array(art["hombro_" + lado]), np.array(art["codo_" + lado]), np.array(art["muneca_" + lado])
        punta = m + (m - c) * 0.35
        d, _, _ = dist_poligonal(P, [h, c, m, punta])
        signo = 1 if lado == "I" else -1
        es = cuerpo_m & (d < ancho) & (signo * P[:, 0] >= 0.88 * abs(h[0]))
        seg[es] = sg
        xb = 0.88 * abs(h[0])       # limite torso/manga
        cerca = cuerpo_m & (d < ancho) & (signo * P[:, 0] > 0)
        for i in np.where(cerca)[0]:
            f_man[i] = PG.suave(signo * P[i, 0], xb - trans_manga, xb + trans_manga) if trans_manga > 0 else float(es[i])
            lado_man[i] = signo
        u = (m - c) / np.linalg.norm(m - c)
        proj = (P - m) @ u
        mano[es & (proj >= 0.0)] = True
        mano_fija[es & (proj >= 0.03)] = True
    # daga: protuberancia a la altura de la cintura (lejos de toda la superficie del original), en componentes conexas
    cand = np.where(cuerpo_m & (seg == "cuerpo") & (P[:, 2] > 0.70) & (P[:, 2] < 1.30) & (np.abs(P[:, 0]) < 0.40))[0]
    dd = np.array([bvh_todo.find_nearest(tuple(P[i]))[3] for i in cand])
    log("daga: %d candidatos a menos de 0.40 m del eje, %d a mas de %.3f m de la superficie original" % (len(cand), (dd > umbral_daga).sum(), umbral_daga))
    dd_all = dict(zip(cand.tolist(), dd.tolist()))
    cand = cand[dd > umbral_daga]
    es_c = np.zeros(n, bool); es_c[cand] = True
    lab = np.arange(n)
    E = np.vstack([Tn[:, [0, 1]], Tn[:, [1, 2]], Tn[:, [2, 0]]]); E = E[es_c[E[:, 0]] & es_c[E[:, 1]]]
    for _ in range(200):
        mm = np.minimum(lab[E[:, 0]], lab[E[:, 1]]); nuevo = lab.copy()
        np.minimum.at(nuevo, E[:, 0], mm); np.minimum.at(nuevo, E[:, 1], mm); nuevo = nuevo[nuevo]
        if (nuevo == lab).all():
            break
        lab = nuevo
    u_, c_ = np.unique(lab[cand], return_counts=True)
    log("daga: componentes de protuberancias (vertices): %s" % sorted(c_.tolist(), reverse=True)[:8])
    daga_v = np.zeros(n, bool)
    for k, cnt in zip(u_, c_):
        if cnt >= min_daga and P[es_c & (lab == k)][:, 2].mean() > 0.9:      # (lo que cuelga por debajo es faldon de la banda, no daga)
            daga_v |= es_c & (lab == k)
    # histeresis: la vaina baja pega a la superficie del original; se anade lo que esta a <= `--radio-daga` de las semillas y sobresale > 1 cm
    if daga_v.any():
        semillas = P[daga_v]
        zona = np.array([i for i, d_ in dd_all.items() if d_ > 0.015 and seg[i] == "cuerpo"], int)
        for ini in range(0, len(zona), 2000):
            blq = zona[ini:ini + 2000]
            dm = np.sqrt(((P[blq][:, None, :] - semillas[None, ::3, :]) ** 2).sum(-1)).min(1)
            daga_v[blq[dm <= radio_daga]] = True
    seg[daga_v] = "daga"
    log("segmentos nuevo: " + json.dumps({s: int((seg == s).sum()) for s in set(seg)}) + ", mano %d (fijos %d)" % (mano.sum(), mano_fija.sum()))

    # ---------------- 3. pesos
    dist_daga = None
    c_dn = P[daga_v].mean(0) if daga_v.any() else None
    wd = [h for h in PG.huesos_de(mapa, "daga")]
    c_do = (sum(cen[h] for h in wd) / max(sum(masa[h] for h in wd), 1e-9)) if wd else None
    if c_dn is not None and c_do is not None:
        dist_daga = float(np.linalg.norm(c_dn - c_do))
    daga_cerca = dist_daga is not None and dist_daga < cerca_daga
    log("daga: distancia al original %s -> %s" % (dist_daga, "huesos de la daga" if daga_cerca else "cadera %d" % cadera))
    def transferir(i, fam):
        """Pesos del vertice i del proxy tomados del original solo de la familia `fam` (triangulos y huesos compatibles)."""
        tree, sel = bvh[fam]
        loc, nor, idx, dist = tree.find_nearest(tuple(P[i]))
        if idx is None:
            return {cadera: 1.0}
        t = To[sel[idx]]
        bary = PG.coords_baricentricas(list(loc), *[Po[k].tolist() for k in t])
        w = PG.interpolar_pesos([Wo[k] for k in t], bary, compat[fam])
        if not w:       # los 3 vertices solo traen huesos de otros segmentos: el compatible de mayor peso mas cercano
            w = PG.limitar_pesos({h: 1.0 / (1e-6 + np.linalg.norm(centroides[h] - P[i])) for h in compat[fam]}, 1)
        return w

    # altura de la entrepierna: lo mas bajo del plano medio entre las piernas (|x| pequeno) del cuerpo, sin daga ni piezas colgantes de la banda
    z_ent = None
    if con_piernas:
        zs = P[cuerpo_m & (seg == "cuerpo") & (np.abs(P[:, 0]) < 0.012) & (P[:, 2] > 0.45) & (P[:, 2] < 1.0), 2]
        z_ent = float(np.percentile(zs, 2)) if len(zs) else z_cad
        log("entrepierna: z=%.3f (%d vertices en el plano medio), z-pierna %.3f, banda %.3f" % (z_ent, len(zs), z_pierna, banda_ent))
    pesos = [None] * n
    n_ent = 0
    for i in range(n):
        s = seg[i]
        if mano[i]:
            pesos[i] = {palma["I" if s == "brazo_I" else "D"]: 1.0}
            continue
        if s == "daga" and not daga_cerca:
            pesos[i] = {cadera: 1.0}
            continue
        if s == "cuerpo" and con_piernas:
            w_c = transferir(i, "cuerpo")
            w = PG.pesos_entrepierna(w_c, transferir(i, "pierna_I"), transferir(i, "pierna_D"), float(P[i, 0]), float(P[i, 2]), z_ent,
                                     banda_ent, caida_ent, peso_cad)
            n_ent += int(abs(P[i, 0]) < banda_ent and P[i, 2] > z_ent - caida_ent)
        else:
            w = transferir(i, s)
        if f_man[i] > 0 and s in ("cuerpo", "brazo_I", "brazo_D"):      # raiz de la manga: hombro + brazo mezclados con el torso
            w_b = w if s != "cuerpo" else transferir(i, "brazo_I" if lado_man[i] > 0 else "brazo_D")
            w_t = w if s == "cuerpo" else transferir(i, "cuerpo")
            w = PG.mezclar_pesos([(f_man[i], w_b), (1.0 - f_man[i], w_t)])
        pesos[i] = w
    log("vertices en la banda de la entrepierna: %d; vertices con escalon de manga: %d" % (n_ent, int((f_man > 0).sum())))
    log("pesos transferidos")

    # ---------------- 4. suavizado en el grafo del proxy
    vec = [set() for _ in range(n)]
    for t in Tn:
        for p, q in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            vec[p].add(int(q)); vec[q].add(int(p))
    grupo = [str(s) for s in seg]
    W = PG.suavizar_pesos(pesos, [sorted(v) for v in vec], grupo, fijos=np.where(mano_fija | (seg == "daga"))[0].tolist(), iteraciones=iters, lam=lam)
    if iters_cuerpo > iters:     # el cuerpo (torso, cadera, piernas) necesita un degradado mas largo: la entrepierna cuelga a la altura de la rodilla
        fijos_c = np.where(mano_fija | (seg != "cuerpo"))[0].tolist()
        W = PG.suavizar_pesos(W, [sorted(v) for v in vec], grupo, fijos=fijos_c, iteraciones=iters_cuerpo - iters, lam=lam)
    # alisados que cruzan segmentos, solo en zonas de union (el resto queda fijo): raiz de la manga (franja del escalon torso -> brazo) y
    # base de la daga (rigida a la cadera 15, pero el cuerpo que la rodea lleva cadera + muslo: sin degradado estira la cadera)
    def ampliar(zona, anillos):
        for _ in range(anillos):
            ext = zona.copy()
            for t in Tn:
                if zona[t].any():
                    ext[t] = True
            zona = ext
        return zona

    zona = np.zeros(n, bool)
    if iters_hombro > 0:
        if trans_manga > 0:
            zona |= ampliar((f_man > 0) & (f_man < 1), 3) & ~mano_fija
        es_daga = seg == "daga"
        if es_daga.any():
            borde_daga = es_daga & np.array([any(not es_daga[j] for j in vec[i]) for i in range(n)])
            zona |= ampliar(borde_daga, 3)
        zona &= ~mano_fija
        W = PG.suavizar_pesos(W, [sorted(v) for v in vec], ["h" if z else str(i) for i, z in enumerate(zona)], fijos=np.where(~zona)[0].tolist(),
                              iteraciones=iters_hombro, lam=lam)
        log("zonas de union (hombro, base de la daga): %d vertices alisados sin separar segmentos" % int(zona.sum()))
    log("suavizado")

    # ---------------- comprobacion de la regla y escritura
    malos = podados = 0
    masa_podada = 0.0
    for i in range(n):
        fam = seg[i]
        if zona[i]:      # zonas de union: se mezclan los segmentos que se tocan (nunca la cabeza)
            permitido = {cadera} | compat["daga"] | compat["cuerpo"] | compat.get("pierna_I", set()) | compat.get("pierna_D", set()) | compat["brazo_I"] | compat["brazo_D"]
        elif seg[i] == "daga":
            permitido = {cadera} | compat["daga"]
        elif fam == "cuerpo":
            permitido = compat["cuerpo"] | compat.get("pierna_I", set()) | compat.get("pierna_D", set())
            if f_man[i] > 0:
                permitido = permitido | compat["brazo_I" if lado_man[i] > 0 else "brazo_D"]
        elif f_man[i] > 0 and fam in ("brazo_I", "brazo_D"):
            permitido = compat[fam] | compat["cuerpo"]       # raiz de la manga: se mezcla con el torso
        else:
            permitido = compat[fam]      # los de la muneca se mezclan con el antebrazo al suavizar
        fuera_ = {h: x for h, x in W[i].items() if h not in permitido}
        if fuera_:      # el alisado largo del cuerpo difunde unos pocos % de un hueso vecino: se quita y se renormaliza
            podados += 1; masa_podada = max(masa_podada, sum(fuera_.values()))
            W[i] = PG.limitar_pesos(W[i], 3, permitido) or {cadera: 1.0}
        if any(h not in permitido for h in W[i]):
            malos += 1
    for g in list(ob.vertex_groups):
        ob.vertex_groups.remove(g)
    por_hueso = {}
    for i, w in enumerate(W):
        for h, x in w.items():
            por_hueso.setdefault(h, []).append((i, x))
    for h, lst in sorted(por_hueso.items()):
        vg = ob.vertex_groups.new(name="h%02d" % h)
        for i, x in lst:
            vg.add([i], x, "REPLACE")
    # segmento por vertice como atributo (para revisarlo y para el informe de estiramiento)
    cod = {s: k for k, s in enumerate(PG.SEGMENTOS)}
    att = me.attributes.new("segmento", "INT", "POINT") if "segmento" not in me.attributes else me.attributes["segmento"]
    seg_att = seg.copy()
    if con_piernas:     # etiqueta informativa: el cuerpo por debajo de la cadera es pierna_I / pierna_D (los pesos se decidieron arriba)
        baja = (seg == "cuerpo") & (P[:, 2] < z_cad)
        seg_att[baja & (P[:, 0] > 0)] = "pierna_I"; seg_att[baja & (P[:, 0] <= 0)] = "pierna_D"
    att.data.foreach_set("value", [cod[s] for s in seg_att])
    bpy.ops.wm.save_as_mainfile(filepath=salida)
    res = {"vertices": n, "segmentos": {s: int((seg_att == s).sum()) for s in PG.SEGMENTOS}, "z_entrepierna_m": z_ent, "z_pierna_m": z_pierna,
           "vertices_banda_entrepierna": n_ent, "vertices_escalon_manga": int((f_man > 0).sum()), "mano": int(mano.sum()), "palma": palma,
           "mapa_hueso_segmento": {str(h): s for h, s in sorted(mapa.items())}, "daga_distancia_al_original_m": dist_daga,
           "daga_regla": "huesos de la daga" if daga_cerca else "cadera %d" % cadera, "vertices_con_hueso_incompatible": malos, "vertices_podados_por_difusion": podados, "masa_podada_max": round(masa_podada, 3),
           "max_huesos_por_vertice": max(len(w) for w in W), "cabeza_huesos_secundarios_descartados": cabeza_descartados, "cuerpo_huesos_secundarios_descartados": sorted(sec_cuerpo), "segundos": round(time.time() - T0, 1)}
    json.dump(res, open(os.path.splitext(salida)[0] + "_pesos.json", "w"), indent=1)
    print("PESOS " + json.dumps(res))


main()
