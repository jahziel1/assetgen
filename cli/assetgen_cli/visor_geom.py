"""Geometria sin dependencias para el visor Godot de poses (`blender/remaster/exportar_visor.py`): lectura de matrices de hueso
(12 floats = base 3x3 por columnas + traslacion, bind -> pantalla), composicion con una transformacion de enlace por hueso (piezas con
espacio bind propio), skinning lineal, camaras de pantalla -> Godot, capsulas por hueso (eje principal de sus vertices) y error entre dos
nubes de puntos. Matrices 4x4 = listas de 4 filas; vectores = listas de 3 floats."""
import math

IDENT = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
# Pantalla del juego (x derecha, y abajo, z hacia dentro) -> Godot (y arriba, -z hacia la camara): giro de 180 grados sobre X.
PANTALLA_A_GODOT = [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def m4_de_12(m):
    """12 floats (columnas de la base 3x3, luego traslacion) -> 4x4."""
    if len(m) != 12:
        raise ValueError("la matriz de hueso debe tener 12 floats, tiene %d" % len(m))
    return [[m[0], m[3], m[6], m[9]], [m[1], m[4], m[7], m[10]], [m[2], m[5], m[8], m[11]], [0.0, 0.0, 0.0, 1.0]]


def m12_de_4(M):
    return [M[r][c] for c in range(3) for r in range(3)] + [M[r][3] for r in range(3)]


def mul4(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def inv4(M):
    """Inversa de una afin (ultima fila 0 0 0 1) con base 3x3 invertible."""
    a, b, c = M[0][:3], M[1][:3], M[2][:3]
    cof = [[b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2], b[0] * c[1] - b[1] * c[0]],
           [c[1] * a[2] - c[2] * a[1], c[2] * a[0] - c[0] * a[2], c[0] * a[1] - c[1] * a[0]],
           [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]]
    det = a[0] * cof[0][0] + a[1] * cof[0][1] + a[2] * cof[0][2]
    if abs(det) < 1e-18:
        raise ValueError("matriz de hueso singular")
    inv = [[cof[j][i] / det for j in range(3)] for i in range(3)]      # adj = cof^T
    t = [M[0][3], M[1][3], M[2][3]]
    return [inv[0] + [-sum(inv[0][k] * t[k] for k in range(3))], inv[1] + [-sum(inv[1][k] * t[k] for k in range(3))],
            inv[2] + [-sum(inv[2][k] * t[k] for k in range(3))], [0.0, 0.0, 0.0, 1.0]]


def aplicar(M, p):
    return [M[0][0] * p[0] + M[0][1] * p[1] + M[0][2] * p[2] + M[0][3],
            M[1][0] * p[0] + M[1][1] * p[1] + M[1][2] * p[2] + M[1][3],
            M[2][0] * p[0] + M[2][1] * p[1] + M[2][2] * p[2] + M[2][3]]


def matrices_hueso(pose12, enlace12=None, pantalla=IDENT):
    """Matriz de cada hueso global: espacio bind del modelo -> `pantalla` (p. ej. Godot).
    `pose12[g]`: bind ORIGINAL del hueso g -> pantalla del juego. `enlace12[g]` (opcional): bind original -> bind del modelo (piezas con
    espacio bind propio llevadas al del cuerpo); se compone con su inversa: M = pantalla . pose . enlace^-1."""
    if enlace12 is not None and len(enlace12) != len(pose12):
        raise ValueError("enlace y pose tienen distinto numero de huesos (%d y %d)" % (len(enlace12), len(pose12)))
    out = []
    for g, m in enumerate(pose12):
        M = m4_de_12(m)
        if enlace12 is not None:
            M = mul4(M, inv4(m4_de_12(enlace12[g])))
        out.append(mul4(pantalla, M))
    return out


def skin_lineal(P, H, W, mats):
    """Skinning lineal. P: [[x,y,z]], H/W: por vertice lista de huesos y de pesos (mismo largo; peso 0 se ignora)."""
    out = []
    for p, hs, ws in zip(P, H, W):
        x = y = z = 0.0
        for h, w in zip(hs, ws):
            if w <= 0.0:
                continue
            q = aplicar(mats[h], p)
            x += w * q[0]; y += w * q[1]; z += w * q[2]
        out.append([x, y, z])
    return out


def mejores_pesos(grupos, k=4):
    """grupos: {hueso: peso} de un vertice -> (huesos, pesos) con los k mayores, normalizados y rellenados con (0, 0.0) hasta k."""
    top = sorted(((w, h) for h, w in grupos.items() if w > 1e-6), reverse=True)[:k]
    s = sum(w for w, _ in top)
    if s <= 0:
        return [0] * k, [0.0] * k
    hs = [h for _, h in top] + [0] * (k - len(top))
    ws = [w / s for w, _ in top] + [0.0] * (k - len(top))
    return hs, ws


def a_transform3d(M):
    """4x4 -> 12 floats en el orden que lee Godot (columnas de la base y origen)."""
    return m12_de_4(M)


def caja(P):
    lo = [min(p[i] for p in P) for i in range(3)]
    hi = [max(p[i] for p in P) for i in range(3)]
    return lo, hi


def normalizar_a_suelo(P, altura=1.75, escala=None):
    """Transformacion uniforme (escala + traslacion) que pone la nube centrada en x/z, pies en y=0 y `altura` de alto (eje y).
    Devuelve (4x4, escala). Si se da `escala`, se usa esa (la misma para varias poses)."""
    lo, hi = caja(P)
    if escala is None:
        escala = altura / (hi[1] - lo[1]) if hi[1] - lo[1] > 1e-6 else 1.0     # nube plana: sin escalar
    cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
    return [[escala, 0.0, 0.0, -escala * cx], [0.0, escala, 0.0, -escala * lo[1]], [0.0, 0.0, escala, -escala * cz],
            [0.0, 0.0, 0.0, 1.0]], escala


def eje_principal(P, iteraciones=60):
    """(centro, eje unitario, valor propio) del mayor autovector de la covarianza (iteracion de potencias)."""
    n = len(P)
    c = [sum(p[i] for p in P) / n for i in range(3)]
    cov = [[0.0] * 3 for _ in range(3)]
    for p in P:
        d = [p[i] - c[i] for i in range(3)]
        for i in range(3):
            for j in range(i, 3):
                cov[i][j] += d[i] * d[j]
    for i in range(3):
        for j in range(i):
            cov[i][j] = cov[j][i]
    v = [0.577, 0.5773, 0.5774]
    lam = 0.0
    for _ in range(iteraciones):
        w = [sum(cov[i][j] * v[j] for j in range(3)) for i in range(3)]
        lam = math.sqrt(sum(x * x for x in w))
        if lam < 1e-30:
            break
        v = [x / lam for x in w]
    return c, v, lam / n


def _percentil(v, q):
    s = sorted(v)
    k = (len(s) - 1) * q
    i = int(math.floor(k))
    j = min(i + 1, len(s) - 1)
    return s[i] + (s[j] - s[i]) * (k - i)


def capsulas(P, hueso, minimo=60):
    """Una capsula por hueso (el dominante de cada vertice): eje principal de sus vertices en espacio bind, extremos entre los percentiles
    4 y 96 % acortados medio radio, radio = 0.75 x mediana de la distancia al eje. `hueso[i]` = hueso dominante del vertice i."""
    por = {}
    for p, h in zip(P, hueso):
        por.setdefault(h, []).append(p)
    out = []
    for h in sorted(por):
        q = por[h]
        if len(q) < minimo:
            continue
        c, e, _ = eje_principal(q)
        t = [sum((p[i] - c[i]) * e[i] for i in range(3)) for p in q]
        d = [math.sqrt(sum((p[i] - c[i] - tt * e[i]) ** 2 for i in range(3))) for p, tt in zip(q, t)]
        r = 0.75 * _percentil(d, 0.5)
        t0, t1 = _percentil(t, 0.04), _percentil(t, 0.96)
        a = [c[i] + e[i] * (t0 + r * 0.5) for i in range(3)]
        b = [c[i] + e[i] * (t1 - r * 0.5) for i in range(3)]
        out.append({"hueso": h, "a": [round(x, 5) for x in a], "b": [round(x, 5) for x in b], "r": round(r, 5), "n": len(q)})
    return out


def error_max(A, B):
    """Maximo de la distancia entre puntos homologos (y su indice)."""
    if len(A) != len(B):
        raise ValueError("nubes de distinto tamano (%d y %d)" % (len(A), len(B)))
    mx, ix = 0.0, -1
    for i, (a, b) in enumerate(zip(A, B)):
        d = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
        if d > mx:
            mx, ix = d, i
    return mx, ix
