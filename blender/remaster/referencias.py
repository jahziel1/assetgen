"""referencias.py -- mide referencias de proporcion con secciones transversales de la malla (numpy; el mismo algoritmo para cualquier
personaje en pose T: z arriba, pies en z = 0, frente -Y, lado I = +X, lado D = -X).

    blender -b -P referencias.py -- --glb modelo.glb --salida referencias_modelo.json

Referencias (todas por cortes plano-malla, sin depender de pesos ni de esqueleto):
  coronilla_z      z maxima (informativa)
  cintura_z        z de la seccion central de menor ancho en x entre 0.45 y 0.70 de la altura (la cintura / faja)
  torso_y          centro en y de esa seccion (percentiles 2-98)
  cabeza_y         centro en y de la seccion de la cabeza 0.10 m bajo la coronilla
  por lado I/D:
  hombro_alto_z    z maxima del corte a 0.165 m del eje (arriba de la pendiente del hombro; con capucha, la del cuello/capucha)
  tobillo          (x, y, z) centro de la seccion de la pierna (z) de menor ancho+fondo entre z = 0.06 y 0.25
  bombacho         (x, y, z) centro de la seccion donde el ancho+fondo de la pierna supera 1.4 veces el de la caña (borde inferior
                   del pantalon holgado; hace de "rodilla" visible porque la rodilla real queda tapada)
  sisa_z           primera z (sube desde 0.9) donde la seccion llega mas lejos que 0.35 m del eje: bajo la manga en la axila
  semiancho_pecho  alcance lateral del torso 0.10 m bajo la sisa
  brazo_ini        (x, y, z) centro del aro del brazo (corte en x) a semiancho_pecho + 0.07 del eje
  fin_manga        idem donde la seccion sqrt(alto*fondo) cae por debajo de 0.6 de su maximo (fin de la manga holgada / inicio del puno)
  muneca           idem en el centro del valle (a menos de 4 mm del minimo) del alto del aro entre fin_manga + 0.04 y fin_manga + 0.6 del tramo hasta la punta
  punta_mano       (x, y, z) extremo lateral (centro de los vertices a menos de 0.05 m del extremo)
El cuello no se mide (capucha/pañuelo lo tapan: su seccion minima cae en la cabeza). La entrepierna no se usa: con pantalon holgado las dos perneras se juntan a alturas que no son comparables entre mallas.
Imprime 'REFERENCIAS ruta'."""
import json
import os
import sys

import numpy as np

LADOS = (("I", 1.0), ("D", -1.0))


def triangulos(me):
    me.calc_loop_triangles()
    t = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", t)
    return t.reshape(-1, 3)


def corte(V, T, origen, normal):
    """Puntos donde el plano (origen, normal) corta las aristas de los triangulos."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    d = (V - np.asarray(origen, float)) @ n
    dt = d[T]
    pos = dt > 0
    m = pos.any(1) & ~pos.all(1)
    Tm, dm = T[m], dt[m]
    pts = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        k = pos[m][:, a] != pos[m][:, b]
        t = dm[k, a] / (dm[k, a] - dm[k, b])
        pts.append(V[Tm[k, a]] + t[:, None] * (V[Tm[k, b]] - V[Tm[k, a]]))
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def suave3(a):
    a = np.asarray(a, float)
    b = a.copy()
    b[1:-1] = (a[:-2] + a[1:-1] + a[2:]) / 3
    return b


def _cruce(xs, ys, umbral):
    """Primer x donde ys pasa de >= umbral a < umbral (interpolado)."""
    for i in range(1, len(xs)):
        if ys[i - 1] >= umbral > ys[i]:
            return xs[i - 1] + (xs[i] - xs[i - 1]) * (ys[i - 1] - umbral) / (ys[i - 1] - ys[i])
    return None


def _ancho_central(V, T, z, corte_x=0.17):
    P = corte(V, T, [0, 0, z], [0, 0, 1])
    P = P[np.abs(P[:, 0]) < corte_x]
    if len(P) < 3:
        return None, None
    return float(np.ptp(P[:, 0])), P


def medir(V, T):
    H = float(V[:, 2].max())
    out = {"coronilla_z": H}
    # --- cuello y cintura: minimo del ancho central
    def minimo(f0, f1, paso=0.01):
        zs = np.arange(f0 * H, f1 * H, paso)
        w = []
        for z in zs:
            a, _ = _ancho_central(V, T, z)
            w.append(a if a is not None else np.inf)
        w = suave3(w)
        i = int(np.argmin(w))
        return float(zs[i]), zs, w
    out["cintura_z"] = minimo(0.45, 0.70)[0]
    _, P = _ancho_central(V, T, out["cintura_z"])
    out["torso_y"] = float((np.percentile(P[:, 1], 2) + np.percentile(P[:, 1], 98)) / 2)
    Pc = corte(V, T, [0, 0, H - 0.10], [0, 0, 1])
    out["cabeza_y"] = float((Pc[:, 1].min() + Pc[:, 1].max()) / 2)
    for lado, sg in LADOS:
        r = {}
        P = corte(V, T, [sg * 0.165, 0, 0], [1, 0, 0])
        r["hombro_alto_z"] = float(P[:, 2].max())
        # --- pierna: perfil por z
        zs = np.arange(0.04, 0.46, 0.01)
        cx, cy, m = [], [], []
        for z in zs:
            P = corte(V, T, [0, 0, z], [0, 0, 1])
            P = P[sg * P[:, 0] > 0.0]
            cx.append((P[:, 0].min() + P[:, 0].max()) / 2)
            cy.append((P[:, 1].min() + P[:, 1].max()) / 2)
            m.append(np.ptp(P[:, 0]) + np.ptp(P[:, 1]))
        cx, cy, m = np.array(cx), np.array(cy), np.array(m)
        ms = suave3(m)
        sel = (zs >= 0.06) & (zs <= 0.25)
        it = int(np.argmin(np.where(sel, ms, np.inf)))
        zt = float(zs[it])
        r["tobillo"] = [float(cx[it]), float(cy[it]), zt]
        caña = (zs >= zt + 0.04) & (zs <= zt + 0.15)
        m0 = float(np.median(m[caña]))
        zb = None
        for i in range(it + 1, len(zs)):
            if zs[i] > zt + 0.15 and m[i] > 1.4 * m0 > m[i - 1]:
                f = (1.4 * m0 - m[i - 1]) / (m[i] - m[i - 1])
                zb = float(zs[i - 1] + f * (zs[i] - zs[i - 1]))
                ib = i
                break
        if zb is None:
            raise RuntimeError("no se encontro el borde del pantalon holgado (lado %s)" % lado)
        r["bombacho"] = [float(np.interp(zb, zs, cx)), float(np.interp(zb, zs, cy)), zb]
        # --- sisa: primera z con alcance lateral > 0.35 m
        zsisa = None
        zz = np.arange(0.9, 0.95 * H, 0.005)
        for z in zz:
            P = corte(V, T, [0, 0, z], [0, 0, 1])
            if len(P) and (sg * P[:, 0]).max() > 0.35:
                zsisa = float(z)
                break
        r["sisa_z"] = zsisa
        P = corte(V, T, [0, 0, zsisa - 0.10], [0, 0, 1])
        r["semiancho_pecho"] = float((sg * P[:, 0]).max()) if False else float(np.percentile(sg * P[(np.abs(P[:, 0]) < 0.35), 0], 100))
        # --- brazo: perfil por x
        xmax = float((sg * V[:, 0]).max())
        xs = np.arange(0.20, xmax - 0.005, 0.01)
        zlo, zhi, ylo, yhi = [], [], [], []
        for x in xs:
            P = corte(V, T, [sg * x, 0, 0], [1, 0, 0])
            P = P[(P[:, 2] > zsisa - 0.1) & (P[:, 2] < zsisa + 0.35)]
            if len(P) < 3:
                zlo.append(np.nan); zhi.append(np.nan); ylo.append(np.nan); yhi.append(np.nan)
                continue
            zlo.append(P[:, 2].min()); zhi.append(P[:, 2].max()); ylo.append(P[:, 1].min()); yhi.append(P[:, 1].max())
        zlo, zhi, ylo, yhi = (np.array(q) for q in (zlo, zhi, ylo, yhi))
        h, d = zhi - zlo, yhi - ylo
        s = suave3(np.sqrt(h * d))
        hs = suave3(h)

        ok = ~np.isnan(zlo)

        def anillo(x):
            """Centro del aro del brazo a la distancia x del eje (interpolado entre los cortes de 1 cm)."""
            return [float(sg * x), float(np.interp(x, xs[ok], ((ylo + yhi) / 2)[ok])), float(np.interp(x, xs[ok], ((zlo + zhi) / 2)[ok]))]
        x_ini = r["semiancho_pecho"] + 0.07
        r["brazo_ini"] = anillo(x_ini)
        imax = int(np.nanargmax(np.where(xs <= 0.75 * xmax, s, -np.inf)))
        xfin = _cruce(xs[imax:], s[imax:], 0.6 * s[imax])
        if xfin is None:
            raise RuntimeError("no se encontro el fin de la manga (lado %s)" % lado)
        r["fin_manga"] = anillo(xfin)
        lo, hi = xfin + 0.04, xfin + 0.6 * (xmax - xfin)
        k = (xs >= lo) & (xs <= hi)
        iw = int(np.argmin(np.where(k, hs, np.inf)))
        # el valle es plano: se toma el centro del tramo contiguo a menos de 4 mm del minimo (mas estable que el argmin de 1 cm)
        i0 = i1 = iw
        while i0 > 0 and k[i0 - 1] and hs[i0 - 1] <= hs[iw] + 0.004:
            i0 -= 1
        while i1 < len(xs) - 1 and k[i1 + 1] and hs[i1 + 1] <= hs[iw] + 0.004:
            i1 += 1
        r["muneca"] = anillo(float((xs[i0] + xs[i1]) / 2))
        mano = V[sg * V[:, 0] > xmax - 0.05]
        r["punta_mano"] = [float(sg * xmax), float(mano[:, 1].mean()), float(mano[:, 2].mean())]
        r["_perfil_brazo"] = {"x": xs.tolist(), "alto": h.tolist(), "fondo": d.tolist()}
        out[lado] = r
    return out


def aplanar(ref):
    """{'coronilla_z':..., 'cuello':[..], 'I': {...}} -> {'cuello': [x,y,z], 'tobillo_I': [...], ...} (sin perfiles)."""
    out = {}
    for k, v in ref.items():
        if k in ("I", "D"):
            for k2, v2 in v.items():
                if not k2.startswith("_"):
                    out[k2 + "_" + k] = v2
        elif not k.startswith("_"):
            out[k] = v
    return out


def main():
    import bpy
    a = sys.argv[sys.argv.index("--") + 1:]

    def arg(n, d=None):
        return a[a.index(n) + 1] if n in a else d
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=arg("--glb"))
    obj = max((o for o in bpy.data.objects if o.type == "MESH"), key=lambda o: len(o.data.vertices))
    me = obj.data
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    V = co.reshape(-1, 3).astype(np.float64)
    ref = medir(V, triangulos(me))
    json.dump(aplanar(ref), open(arg("--salida"), "w"), indent=1)
    print("REFERENCIAS", arg("--salida"))


if __name__ == "__main__":
    main()
