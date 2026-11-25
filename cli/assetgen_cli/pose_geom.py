"""Geometria sin dependencias para llevar un personaje a pose T y alinearlo con un original (`blender/remaster/articulaciones.py`,
`reposar_T.py`, `alinear_T.py`): rotaciones, angulos de brazo, pesos de mezcla, articulaciones desde pesos de huesos y similitud
uniforme (escala + traslacion) con error por articulacion. Vectores = tuplas/listas de 3 floats; matrices 3x3 = listas de filas."""
import math


def sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def mul(a, s):
    return [a[0] * s, a[1] * s, a[2] * s]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def norma(a):
    return math.sqrt(dot(a, a))


def unitario(a):
    n = norma(a)
    if n < 1e-12:
        raise ValueError("vector nulo")
    return [a[0] / n, a[1] / n, a[2] / n]


def rot_eje_angulo(eje, ang):
    """Matriz de Rodrigues del giro `ang` (rad) alrededor de `eje` (no hace falta que sea unitario)."""
    x, y, z = unitario(eje)
    c, s = math.cos(ang), math.sin(ang)
    t = 1 - c
    return [[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c]]


def rot_entre(a, b):
    """Giro de arco minimo que lleva la direccion `a` a la direccion `b`."""
    a, b = unitario(a), unitario(b)
    c = dot(a, b)
    eje = cross(a, b)
    if norma(eje) < 1e-12:
        if c > 0:
            return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        # opuestos: giro de pi alrededor de cualquier perpendicular
        p = cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else cross(a, [0, 1, 0])
        return rot_eje_angulo(p, math.pi)
    return rot_eje_angulo(eje, math.atan2(norma(eje), c))


def mat_vec(m, v):
    return [dot(m[0], v), dot(m[1], v), dot(m[2], v)]


def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def giro_en_pivote(m, p, pivote):
    """Aplica la rotacion `m` al punto `p` alrededor de `pivote`."""
    return add(mat_vec(m, sub(p, pivote)), pivote)


def angulo_sobre_horizontal(d):
    """Grados de la direccion `d` sobre el plano horizontal (z arriba): 0 = horizontal, -90 = hacia abajo."""
    d = unitario(d)
    return math.degrees(math.asin(max(-1.0, min(1.0, d[2]))))


def angulo_con_vertical_abajo(d):
    """Grados entre `d` y la vertical hacia abajo (0 = colgando, 90 = horizontal)."""
    d = unitario(d)
    return math.degrees(math.acos(max(-1.0, min(1.0, -d[2]))))


def direccion_horizontal(lado, grados_sobre_horizontal):
    """Direccion del brazo en T: lado = +1 (+X) o -1 (-X), inclinada `grados_sobre_horizontal` (negativo = baja la mano)."""
    a = math.radians(grados_sobre_horizontal)
    return [lado * math.cos(a), 0.0, math.sin(a)]


def suave(x):
    """Smoothstep 0..1."""
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def peso_tramo(t, t0, t1):
    """Peso 0 -> 1 al avanzar a lo largo de un eje entre t0 y t1 (smoothstep). Si t0 == t1 es un escalon."""
    if t1 == t0:
        return 1.0 if t >= t0 else 0.0
    return suave((t - t0) / (t1 - t0))


def cruce_pesos(pos, pesos, prox, dist, eje, signo=1, rango=None, losa=0.05, minimo=0.5):
    """Articulacion entre dos grupos de huesos a partir de los pesos de un original.
    pos: lista de [x,y,z]; pesos: lista de dict hueso->peso; prox/dist: huesos del lado proximal y del distal; eje: 0/1/2 a lo largo
    del cual avanza el miembro; signo: +1/-1 segun el miembro crezca hacia + o - de ese eje; rango: (lo, hi) de la coordenada (en el
    sentido del miembro) donde buscar. Por cada intervalo se promedia la fraccion distal D/(P+D) de los vertices que pesan en alguno
    de los dos grupos; la articulacion esta donde esa fraccion cruza 0.5 (interpolacion). Las otras dos coordenadas son el promedio
    de los vertices de la losa (+-`losa`) alrededor del cruce, ponderado por min(P, D) + 0.05. Devuelve (punto, n_vertices_losa)."""
    prox, dist = set(prox), set(dist)
    pts = []
    for p, w in zip(pos, pesos):
        P = sum(v for k, v in w.items() if k in prox)
        D = sum(v for k, v in w.items() if k in dist)
        if P + D < minimo:
            continue
        s = p[eje] * signo
        if rango and not (rango[0] <= s <= rango[1]):
            continue
        pts.append((s, D / (P + D), p, min(P, D)))
    if len(pts) < 4:
        raise ValueError("pocos vertices para estimar la articulacion")
    lo = min(q[0] for q in pts)
    hi = max(q[0] for q in pts)
    nb = 14
    cs, fs = [], []
    for i in range(nb):
        a, b = lo + (hi - lo) * i / nb, lo + (hi - lo) * (i + 1) / nb
        sel = [q for q in pts if a <= q[0] < b or (i == nb - 1 and q[0] == b)]
        if len(sel) >= 2:
            cs.append(sum(q[0] for q in sel) / len(sel))
            fs.append(sum(q[1] for q in sel) / len(sel))
    x0 = None
    for i in range(len(cs) - 1):
        if (fs[i] - 0.5) * (fs[i + 1] - 0.5) <= 0 and fs[i] != fs[i + 1]:
            x0 = cs[i] + (0.5 - fs[i]) * (cs[i + 1] - cs[i]) / (fs[i + 1] - fs[i])
            break
    if x0 is None:
        raise ValueError("la fraccion distal no cruza 0.5")
    sel = [q for q in pts if abs(q[0] - x0) <= losa]
    if not sel:
        raise ValueError("losa vacia")
    wsum = sum(q[3] + 0.05 for q in sel)
    out = [sum(q[2][k] * (q[3] + 0.05) for q in sel) / wsum for k in range(3)]
    out[eje] = x0 * signo
    return out, len(sel)


def ajustar_escala_traslacion(origen, destino):
    """Escala uniforme s y traslacion t que minimizan sum |s*a_i + t - b_i|^2 (sin giro). Devuelve (s, t)."""
    n = len(origen)
    if n < 2 or n != len(destino):
        raise ValueError("hacen falta >= 2 pares")
    ca = [sum(p[k] for p in origen) / n for k in range(3)]
    cb = [sum(p[k] for p in destino) / n for k in range(3)]
    num = sum(dot(sub(a, ca), sub(b, cb)) for a, b in zip(origen, destino))
    den = sum(dot(sub(a, ca), sub(a, ca)) for a in origen)
    if den < 1e-18:
        raise ValueError("puntos de origen degenerados")
    s = num / den
    return s, sub(cb, mul(ca, s))


def errores_alineacion(origen, destino, s, t):
    """Distancia (misma unidad) entre s*a+t y b para cada par."""
    return [norma(sub(add(mul(a, s), t), b)) for a, b in zip(origen, destino)]
