"""Geometria pura (sin numpy ni bpy) para envolver una malla de juego sobre un modelo nuevo conservando topologia, pesos y piezas.

Piezas: subdivision conforme por longitud de arista (sin suavizado), pesos interpolados (<= n por vertice), vertices en la misma
posicion agrupados (costuras entre piezas), suavizado de desplazamientos y utilidades de metricas. Usado por
`assetgen/blender/remaster/envolver_original.py`.
"""
import math

from . import pesos_geom as PG


def _arista(a, b):
    return (a, b) if a < b else (b, a)


def _dist(p, q):
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2)


def subdividir_por_longitud(P, caras, pesos, pieza, lmax, rondas=8, n_pesos=3, factor=None):
    """Parte las aristas mas largas que `lmax` por su punto medio (red-green conforme: las caras vecinas se parten para no abrir grietas)
    y repite hasta que no quede ninguna. Sin suavizado. Devuelve (P, caras, pesos, pieza, padres) donde `padres[i]` = (a, b) vertices de
    la arista de la que sale el vertice nuevo i (o None si es original). `pieza[k]` = pieza de la cara k (se hereda). Los pesos del
    punto medio son la media de los de la arista (<= n_pesos). `factor[v]` (opcional, por vertice) multiplica `lmax` en las aristas
    que tocan a v (el minimo de los dos extremos): zonas que piden mas densidad (cara, manos); el punto medio hereda el minimo."""
    P = [tuple(p) for p in P]
    caras = [tuple(c) for c in caras]
    pesos = [dict(w) for w in pesos]
    pieza = list(pieza)
    padres = [None] * len(P)
    factor = [1.0] * len(P) if factor is None else list(factor)
    for _ in range(rondas):
        largas = set()
        for c in caras:
            for k in range(3):
                a, b = c[k], c[(k + 1) % 3]
                if _dist(P[a], P[b]) > lmax * min(factor[a], factor[b]):
                    largas.add(_arista(a, b))
        if not largas:
            break
        medio = {}
        for a, b in sorted(largas):
            medio[(a, b)] = len(P)
            P.append(tuple((P[a][i] + P[b][i]) * 0.5 for i in range(3)))
            m = {}
            for h, x in pesos[a].items():
                m[h] = m.get(h, 0.0) + 0.5 * x
            for h, x in pesos[b].items():
                m[h] = m.get(h, 0.0) + 0.5 * x
            pesos.append(PG.limitar_pesos(m, n_pesos) or dict(pesos[a]))
            padres.append((a, b))
            factor.append(min(factor[a], factor[b]))
        nuevas, npieza = [], []
        for c, pz in zip(caras, pieza):
            ms = [medio.get(_arista(c[k], c[(k + 1) % 3])) for k in range(3)]   # punto medio de la arista k (c[k], c[k+1])
            n = sum(m is not None for m in ms)
            if n == 0:
                trozos = [c]
            elif n == 3:
                a, b, d = c
                ab, bd, da = ms
                trozos = [(a, ab, da), (ab, b, bd), (da, bd, d), (ab, bd, da)]
            elif n == 1:
                k = next(i for i in range(3) if ms[i] is not None)
                a, b, d = c[k], c[(k + 1) % 3], c[(k + 2) % 3]
                m = ms[k]
                trozos = [(a, m, d), (m, b, d)]
            else:
                k = next(i for i in range(3) if ms[i] is None)     # arista sin partir: (c[k], c[k+1])
                a, b, d = c[k], c[(k + 1) % 3], c[(k + 2) % 3]
                mbd, mda = ms[(k + 1) % 3], ms[(k + 2) % 3]
                # cuadrilatero a, b, mbd, mda + triangulo (mda, mbd, d); diagonal mas corta del cuadrilatero
                if _dist(P[a], P[mbd]) <= _dist(P[b], P[mda]):
                    trozos = [(a, b, mbd), (a, mbd, mda), (mda, mbd, d)]
                else:
                    trozos = [(a, b, mda), (b, mbd, mda), (mda, mbd, d)]
            nuevas.extend(trozos)
            npieza.extend([pz] * len(trozos))
        caras, pieza = nuevas, npieza
    return P, caras, pesos, pieza, padres


def subdividir_hasta(P, caras, pesos, pieza, objetivo, n_pesos=3, tolerancia=0.1, iteraciones=14, factor=None):
    """Busca por biseccion la `lmax` con la que el numero de triangulos queda cerca de `objetivo` (+-`tolerancia`).
    Devuelve (resultado_de_subdividir_por_longitud, lmax)."""
    bajo, alto = 1e-4, max(max(_dist(P[c[k]], P[c[(k + 1) % 3]]) for k in range(3)) for c in caras) * 1.01
    mejor = None
    for _ in range(iteraciones):
        lm = math.sqrt(bajo * alto)
        r = subdividir_por_longitud(P, caras, pesos, pieza, lm, n_pesos=n_pesos, factor=factor)
        n = len(r[1])
        if mejor is None or abs(n - objetivo) < abs(len(mejor[0][1]) - objetivo):
            mejor = (r, lm)
        if abs(n - objetivo) <= tolerancia * objetivo:
            return r, lm
        if n > objetivo:
            bajo = lm
        else:
            alto = lm
    return mejor


def factor_por_grupo(grupo, caras, pieza, factor_pieza):
    """Factor por vertice = el menor factor de las piezas que tocan su grupo de posicion (las duplicadas de una costura parten igual).
    `factor_pieza[k]` = factor de la pieza k (1.0 si falta)."""
    mejor = {}
    for c, z in zip(caras, pieza):
        f = factor_pieza.get(z, 1.0)
        for v in c:
            g = grupo[v]
            if g not in mejor or f < mejor[g]:
                mejor[g] = f
    return [mejor.get(g, 1.0) for g in grupo]


def agrupar_por_posicion(P, tol=1e-4):
    """Indice de grupo por vertice: los vertices a menos de ~`tol` en cada eje quedan en el mismo grupo (cuadricula; costuras entre piezas)."""
    celdas, grupo = {}, []
    for p in P:
        k = tuple(int(round(c / tol)) for c in p)
        grupo.append(celdas.setdefault(k, len(celdas)))
    return grupo


def vecinos_de_grupos(caras, grupo, n_grupos):
    """Lista de conjuntos: grupos vecinos por aristas de las caras."""
    vec = [set() for _ in range(n_grupos)]
    for c in caras:
        for k in range(3):
            a, b = grupo[c[k]], grupo[c[(k + 1) % 3]]
            if a != b:
                vec[a].add(b)
                vec[b].add(a)
    return vec


def suavizar_desplazamientos(D, vecinos, iteraciones=6, lam=0.5, fijos=None, confianza=None):
    """Suaviza un campo de desplazamientos (lista de (x,y,z)) por el grafo. `fijos`: indices que no cambian. `confianza[i]` en [0,1]:
    los vecinos con poca confianza pesan menos al promediar (un desplazamiento sin impacto no arrastra a los demas)."""
    D = [tuple(d) for d in D]
    fijos = fijos or ()
    for _ in range(iteraciones):
        nuevo = []
        for i, d in enumerate(D):
            vs = vecinos[i]
            if not vs or i in fijos:
                nuevo.append(d)
                continue
            sw = 0.0
            sx = sy = sz = 0.0
            for j in vs:
                w = 1.0 if confianza is None else max(confianza[j], 1e-3)
                sw += w
                sx += w * D[j][0]
                sy += w * D[j][1]
                sz += w * D[j][2]
            m = (sx / sw, sy / sw, sz / sw)
            nuevo.append(tuple((1 - lam) * d[k] + lam * m[k] for k in range(3)))
        D = nuevo
    return D


def limitar_longitud(d, maximo):
    """Recorta el vector `d` a longitud `maximo`."""
    n = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
    if n <= maximo or n < 1e-15:
        return tuple(d)
    s = maximo / n
    return (d[0] * s, d[1] * s, d[2] * s)


def resumen_distancias(valores, qs=(50, 90, 95, 99)):
    """Resumen {n, p50, p90, p95, p99, max, media} (en las unidades de entrada) para un informe por pieza."""
    if not valores:
        return {"n": 0}
    r = {"n": len(valores), "media": round(sum(valores) / len(valores), 5), "max": round(max(valores), 5)}
    for q in qs:
        r["p%d" % q] = round(PG.percentil(valores, q), 5)
    return r


def caras_invertidas(P0, P1, caras, umbral=0.0):
    """Indices de caras cuya normal, tras mover los vertices de P0 a P1, apunta contra la original (producto escalar <= umbral)."""
    malas = []
    for k, (a, b, c) in enumerate(caras):
        n0 = _normal_cara(P0[a], P0[b], P0[c])
        n1 = _normal_cara(P1[a], P1[b], P1[c])
        if n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2] <= umbral:
            malas.append(k)
    return malas


def _normal_cara(p, q, r):
    u = (q[0] - p[0], q[1] - p[1], q[2] - p[2])
    v = (r[0] - p[0], r[1] - p[1], r[2] - p[2])
    n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    m = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
    return (n[0] / m, n[1] / m, n[2] / m) if m > 1e-18 else (0.0, 0.0, 0.0)


def dmax_por_pieza(nombres, base, especiales=None, defecto_mult=1.0):
    """Distancia maxima de proyeccion por pieza: `base` m para todas y `especiales` {nombre_parcial: m} para las que difieren mucho del nuevo
    (se compara por 'contiene'). Devuelve {nombre: m}."""
    out = {}
    for nm in nombres:
        v = base * defecto_mult
        for clave, d in (especiales or {}).items():
            if clave in nm:
                v = d
        out[nm] = v
    return out
