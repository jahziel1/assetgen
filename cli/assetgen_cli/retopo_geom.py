"""Geometria sin dependencias para la retopologia del modelo del juego (`blender/remaster/retopologia.py`, `hornear.py`):
topologia de una malla (aristas abiertas, no-manifold, componentes, caracteristica de Euler), reparto de un presupuesto de caras
entre partes, escalado de islas UV por densidad, densidad de texel y composicion de dos horneados en un mismo atlas por mascaras.
Las caras son tuplas de indices de vertice (triangulos o quads); las mallas en Blender se pasan como listas de tuplas."""
import math
from collections import defaultdict


def aristas(caras):
    """{(a, b) con a<b: numero de caras que la usan}."""
    cuenta = defaultdict(int)
    for f in caras:
        n = len(f)
        for i in range(n):
            a, b = f[i], f[(i + 1) % n]
            if a == b:
                continue
            cuenta[(a, b) if a < b else (b, a)] += 1
    return cuenta


def componentes(caras):
    """Numero de componentes conexas (por vertice compartido) y tamano de cada una en caras, de mayor a menor."""
    padre = {}

    def raiz(x):
        while padre.setdefault(x, x) != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for f in caras:
        r0 = raiz(f[0])
        for v in f[1:]:
            r = raiz(v)
            if r != r0:
                padre[r] = r0
    tam = defaultdict(int)
    for f in caras:
        tam[raiz(f[0])] += 1
    return sorted(tam.values(), reverse=True)


def topologia(caras):
    """Resumen: vertices usados, aristas, abiertas (1 cara), no-manifold (>2 caras), componentes, Euler y si esta cerrada y es manifold."""
    ar = aristas(caras)
    usados = {v for f in caras for v in f}
    abiertas = sum(1 for n in ar.values() if n == 1)
    nomani = sum(1 for n in ar.values() if n > 2)
    comp = componentes(caras)
    euler = len(usados) - len(ar) + len(caras)
    return {"vertices": len(usados), "caras": len(caras), "aristas": len(ar), "aristas_abiertas": abiertas,
            "aristas_no_manifold": nomani, "componentes": len(comp), "tam_componentes": comp[:6], "euler": euler,
            "cerrada_manifold": bool(caras) and abiertas == 0 and nomani == 0}


def reparto(pesos, total, minimo=0):
    """Reparte `total` enteros proporcionalmente a `pesos` (metodo del resto mayor), con un minimo por parte."""
    if not pesos or total < 0:
        raise ValueError("pesos vacios o total negativo")
    s = float(sum(pesos))
    if s <= 0:
        raise ValueError("la suma de pesos debe ser positiva")
    bruto = [max(minimo, p / s * total) for p in pesos]
    base = [int(math.floor(x)) for x in bruto]
    resto = total - sum(base)
    orden = sorted(range(len(pesos)), key=lambda i: -(bruto[i] - base[i]))
    for i in orden[:max(0, resto)]:
        base[i] += 1
    return base


def triangulos_de_quads(caras_quad):
    return 2 * caras_quad


def densidad_texel(area_3d_m2, area_uv, resolucion):
    """Pixeles por metro de una malla con `area_3d_m2` de superficie que ocupa `area_uv` (fraccion 0..1 del atlas) en un atlas cuadrado."""
    if area_3d_m2 <= 0 or area_uv <= 0:
        raise ValueError("areas positivas")
    return resolucion * math.sqrt(area_uv / area_3d_m2)


def factor_escala_uv(area3d_isla, area_uv_isla, densidad_objetivo_px_m, resolucion):
    """Factor lineal para que una isla alcance `densidad_objetivo_px_m` (px/m) en un atlas de `resolucion` px."""
    actual = densidad_texel(area3d_isla, area_uv_isla, resolucion)
    return densidad_objetivo_px_m / actual


def distorsion_triangulo(p3, uv):
    """Razon de estiramiento de un triangulo 3D -> UV: cociente entre los valores singulares de la jacobiana (1 = isometrico).
    p3: 3 puntos (x,y,z); uv: 3 puntos (u,v). Devuelve inf si el triangulo UV es degenerado."""
    e1 = [p3[1][i] - p3[0][i] for i in range(3)]
    e2 = [p3[2][i] - p3[0][i] for i in range(3)]
    l1 = math.sqrt(sum(x * x for x in e1))
    if l1 == 0:
        return math.inf
    x1 = [x / l1 for x in e1]
    d = sum(e2[i] * x1[i] for i in range(3))
    y2 = [e2[i] - d * x1[i] for i in range(3)]
    h = math.sqrt(sum(x * x for x in y2))
    if h == 0:
        return math.inf
    # base local 2D: A=(l1,0), B=(d,h); jacobiana J = [uvA uvB] * inv([A B])
    du1, dv1 = uv[1][0] - uv[0][0], uv[1][1] - uv[0][1]
    du2, dv2 = uv[2][0] - uv[0][0], uv[2][1] - uv[0][1]
    ju = [du1 / l1, (du2 - d * du1 / l1) / h]
    jv = [dv1 / l1, (dv2 - d * dv1 / l1) / h]
    a, b, c, e = ju[0], ju[1], jv[0], jv[1]
    det = a * e - b * c
    if abs(det) < 1e-18:
        return math.inf
    s = a * a + b * b + c * c + e * e
    disc = math.sqrt(max(0.0, s * s - 4 * det * det))
    s1 = math.sqrt((s + disc) / 2)
    s2 = math.sqrt(max((s - disc) / 2, 0.0))
    return math.inf if s2 == 0 else s1 / s2


def elegir_origen(mascara_a, mascara_b, alfa_a, alfa_b):
    """Composicion de dos horneados en un atlas por pixel: 'a' o 'b' donde su mascara exacta cubre; fuera de ambas, el que tenga margen
    (alfa>0), con preferencia por 'a'; None si ninguno. Todo son listas del mismo largo."""
    sal = []
    for ma, mb, aa, ab in zip(mascara_a, mascara_b, alfa_a, alfa_b):
        if ma and not mb:
            sal.append("a")
        elif mb and not ma:
            sal.append("b")
        elif ma and mb:
            sal.append("a")          # solape de islas: no deberia haberlo; gana la primera y se cuenta aparte
        elif aa > 0:
            sal.append("a")
        elif ab > 0:
            sal.append("b")
        else:
            sal.append(None)
    return sal


def solapes(mascara_a, mascara_b):
    """Pixeles cubiertos por ambas mascaras (islas de partes distintas que se pisan en el atlas)."""
    return sum(1 for x, y in zip(mascara_a, mascara_b) if x and y)


def hueco_minimo_voxel(hueco_m, factor=2.0):
    """Voxel mas grueso que no funde un hueco de `hueco_m` metros: el hueco debe abarcar al menos `factor` voxeles."""
    if hueco_m <= 0:
        raise ValueError("hueco positivo")
    return hueco_m / factor
