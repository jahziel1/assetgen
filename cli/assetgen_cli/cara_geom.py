"""Geometria de Python puro (sin numpy) para el ajuste de una cabeza base a una cabeza generada: similitud de Umeyama/Horn,
deformacion RBF (thin-plate 3D con termino afin), poligono 2D con distancia con signo, color de iris y tinte de iris.
La usan los scripts de Blender de blender/remaster/cara_*.py (que anaden cli/ al sys.path) y las pruebas; no depende de ningun juego."""
import colorsys
import math


# ---------------------------------------------------------------- algebra pequena
def _resolver(a, b):
    """Resuelve a x = b (a: n x n, b: n x m) por eliminacion gaussiana con pivote parcial. Devuelve n x m."""
    n = len(a)
    m = len(b[0])
    mat = [list(map(float, a[i])) + list(map(float, b[i])) for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(mat[r][c]))
        if abs(mat[p][c]) < 1e-12:
            raise ValueError("sistema singular")
        mat[c], mat[p] = mat[p], mat[c]
        piv = mat[c][c]
        for r in range(c + 1, n):
            f = mat[r][c] / piv
            if f:
                fila_c, fila_r = mat[c], mat[r]
                for k in range(c, n + m):
                    fila_r[k] -= f * fila_c[k]
    x = [[0.0] * m for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(m):
            s = mat[i][n + j] - sum(mat[i][k] * x[k][j] for k in range(i + 1, n))
            x[i][j] = s / mat[i][i]
    return x


def _jacobi_simetrica(a, iters=60):
    """Autovalores/autovectores de una matriz simetrica pequena (Jacobi). Devuelve (valores, vectores en columnas)."""
    n = len(a)
    a = [list(map(float, f)) for f in a]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(iters):
        off = sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off < 1e-24:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-30:
                    continue
                th = (a[q][q] - a[p][p]) / (2 * a[p][q])
                t = (1 if th >= 0 else -1) / (abs(th) + math.sqrt(th * th + 1))
                c = 1 / math.sqrt(t * t + 1)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    return [a[i][i] for i in range(n)], v


# ---------------------------------------------------------------- similitud (Umeyama / Horn)
def _quat_a_matriz(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def umeyama(src, dst, escalar=True, pesos=None):
    """Similitud dst ~ s * R @ src + t (R rotacion propia, sin reflexion). Metodo de Horn (cuaternion). -> (s, R 3x3, t)."""
    n = len(src)
    w = pesos or [1.0] * n
    sw = float(sum(w))
    cs = [sum(w[i] * src[i][k] for i in range(n)) / sw for k in range(3)]
    cd = [sum(w[i] * dst[i][k] for i in range(n)) / sw for k in range(3)]
    s_ = [[src[i][k] - cs[k] for k in range(3)] for i in range(n)]
    d_ = [[dst[i][k] - cd[k] for k in range(3)] for i in range(n)]
    S = [[sum(w[i] * s_[i][a] * d_[i][b] for i in range(n)) for b in range(3)] for a in range(3)]
    sxx, sxy, sxz = S[0]
    syx, syy, syz = S[1]
    szx, szy, szz = S[2]
    N = [[sxx + syy + szz, syz - szy, szx - sxz, sxy - syx],
         [syz - szy, sxx - syy - szz, sxy + syx, szx + sxz],
         [szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy],
         [sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz]]
    vals, vecs = _jacobi_simetrica(N)
    i = max(range(4), key=lambda k: vals[k])
    q = [vecs[r][i] for r in range(4)]
    nq = math.sqrt(sum(x * x for x in q))
    q = [x / nq for x in q]
    R = _quat_a_matriz(q)
    if escalar:
        num = sum(w[i] * sum(d_[i][a] * sum(R[a][b] * s_[i][b] for b in range(3)) for a in range(3)) for i in range(n))
        den = sum(w[i] * sum(x * x for x in s_[i]) for i in range(n))
        s = num / den if den > 0 else 1.0
    else:
        s = 1.0
    t = [cd[a] - s * sum(R[a][b] * cs[b] for b in range(3)) for a in range(3)]
    return s, R, t


def aplicar_similitud(sim, pts):
    s, R, t = sim
    return [[s * sum(R[a][b] * p[b] for b in range(3)) + t[a] for a in range(3)] for p in pts]


def umeyama_robusta(src, dst, umbral_mad=3.0, minimo=6, escalar=True):
    """Umeyama con rechazo iterativo de atipicos (residuo > mediana + umbral_mad * MAD). -> (sim, indices_usados)."""
    usados = list(range(len(src)))
    for _ in range(5):
        sim = umeyama([src[i] for i in usados], [dst[i] for i in usados], escalar)
        res = [math.dist(a, b) for a, b in zip(aplicar_similitud(sim, [src[i] for i in usados]), [dst[i] for i in usados])]
        med = sorted(res)[len(res) // 2]
        mad = sorted(abs(r - med) for r in res)[len(res) // 2] or 1e-12
        nuevos = [i for i, r in zip(usados, res) if r <= med + umbral_mad * 1.4826 * mad]
        if len(nuevos) == len(usados) or len(nuevos) < minimo:
            break
        usados = nuevos
    return umeyama([src[i] for i in usados], [dst[i] for i in usados], escalar), usados


# ---------------------------------------------------------------- RBF thin-plate (3D)
def rbf_ajustar(src, dst, lam=0.0):
    """Mapa suave src -> dst: f(p) = A [1, p] + sum_i w_i |p - src_i| (biarmonica 3D = thin-plate). `lam` regulariza (suaviza)."""
    n = len(src)
    mat = [[0.0] * (n + 4) for _ in range(n + 4)]
    for i in range(n):
        for j in range(n):
            mat[i][j] = math.dist(src[i], src[j]) + (lam if i == j else 0.0)
        mat[i][n] = mat[n][i] = 1.0
        for k in range(3):
            mat[i][n + 1 + k] = mat[n + 1 + k][i] = src[i][k]
    rhs = [list(map(float, d)) for d in dst] + [[0.0] * 3 for _ in range(4)]
    sol = _resolver(mat, rhs)
    return dict(src=[list(map(float, p)) for p in src], w=sol[:n], a=sol[n:])


def rbf_evaluar(m, pts):
    src, w, a = m["src"], m["w"], m["a"]
    out = []
    for p in pts:
        r = [a[0][k] + a[1][k] * p[0] + a[2][k] * p[1] + a[3][k] * p[2] for k in range(3)]
        for s, wi in zip(src, w):
            d = math.dist(p, s)
            r[0] += wi[0] * d
            r[1] += wi[1] * d
            r[2] += wi[2] * d
        out.append(r)
    return out


# ---------------------------------------------------------------- poligono 2D y suavizados
def distancia_poligono(poly, p):
    """Distancia con signo de p=(x,y) al poligono cerrado `poly` (positiva dentro, negativa fuera)."""
    dentro = False
    dmin = float("inf")
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > p[1]) != (y2 > p[1]) and p[0] < (x2 - x1) * (p[1] - y1) / (y2 - y1) + x1:
            dentro = not dentro
        dx, dy = x2 - x1, y2 - y1
        l2 = dx * dx + dy * dy
        t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((p[0] - x1) * dx + (p[1] - y1) * dy) / l2))
        dmin = min(dmin, math.hypot(p[0] - (x1 + t * dx), p[1] - (y1 + t * dy)))
    return dmin if dentro else -dmin


def suavizado(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- color de iris
def color_iris(pixeles, descartar_oscuros=0.5, descartar_claros=0.1):
    """Color medio del iris a partir de los pixeles RGB (0..255) de un disco centrado en la pupila: descarta la pupila/pestanas
    (los mas oscuros) y los brillos/esclerotica (los mas claros) y promedia el resto. -> (r, g, b) enteros."""
    if not pixeles:
        raise ValueError("sin pixeles")
    orden = sorted(pixeles, key=lambda c: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
    n = len(orden)
    a, b = int(n * descartar_oscuros), n - int(n * descartar_claros)
    medio = orden[a:b] or orden
    return tuple(int(round(sum(c[k] for c in medio) / len(medio))) for k in range(3))


def estadistica_iris(pixeles_rgb_textura, sat_min=0.30):
    """Referencia (h, s, v medios, 0..1) del iris en una textura de ojo: media de los pixeles saturados (iris, no la esclerotica)."""
    hs = [colorsys.rgb_to_hsv(r / 255, g / 255, b / 255) for r, g, b in pixeles_rgb_textura]
    iris = [x for x in hs if x[1] >= sat_min and x[2] > 0.15] or hs
    n = len(iris)
    # el tono es circular: promedio vectorial
    cx = sum(math.cos(2 * math.pi * x[0]) for x in iris) / n
    sx = sum(math.sin(2 * math.pi * x[0]) for x in iris) / n
    return ((math.atan2(sx, cx) / (2 * math.pi)) % 1.0, sum(x[1] for x in iris) / n, sum(x[2] for x in iris) / n)


def tintar_pixel(rgb, ref_hsv, objetivo_rgb, sat_min=0.30):
    """Cambia el tono/saturacion/valor de un pixel de iris (saturado) de la referencia `ref_hsv` al `objetivo_rgb`, conservando
    el relieve (la variacion respecto a la media). Los pixeles poco saturados (esclerotica, alfa) se devuelven sin cambios."""
    h, s, v = colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)
    if s < sat_min:
        return tuple(rgb)
    ht, st, vt = colorsys.rgb_to_hsv(objetivo_rgb[0] / 255, objetivo_rgb[1] / 255, objetivo_rgb[2] / 255)
    s2 = max(0.0, min(1.0, st * (s / ref_hsv[1] if ref_hsv[1] > 1e-6 else 1.0)))
    v2 = max(0.0, min(1.0, vt * (v / ref_hsv[2] if ref_hsv[2] > 1e-6 else 1.0)))
    r, g, b = colorsys.hsv_to_rgb(ht, s2, v2)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))
