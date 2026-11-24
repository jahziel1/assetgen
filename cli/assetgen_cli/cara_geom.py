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


# ---------------------------------------------------------------- region de caras, lazos cerrados, parpados (etapa "unir")
def apertura_relativa(sup, inf, ext, interior):
    """Apertura de un ojo = distancia entre los puntos de parpado superior e inferior / distancia entre comisuras (2D o 3D)."""
    return math.dist(sup, inf) / math.dist(ext, interior)


def vecinos_de_caras(caras):
    """Lista, por cara, de las caras que comparten una arista con ella (aristas de borde: sin vecino)."""
    por_arista = {}
    for k, f in enumerate(caras):
        for i in range(len(f)):
            e = (min(f[i], f[(i + 1) % len(f)]), max(f[i], f[(i + 1) % len(f)]))
            por_arista.setdefault(e, []).append(k)
    vec = [[] for _ in caras]
    for fs in por_arista.values():
        for a in fs:
            vec[a] += [b for b in fs if b != a]
    return vec


def abrir_region(caras, activas, pasos=1):
    """Apertura morfologica de un conjunto de caras (indices): erosiona `pasos` veces (una cara sobrevive si todas sus caras vecinas
    existentes estan activas; las aristas de borde de la malla no cuentan) y vuelve a dilatar dentro del conjunto original. Quita
    rabitos y dientes de ancho 1 cara sin tocar los bordes propios de la malla. -> set de indices."""
    vec = vecinos_de_caras(caras)
    orig = set(activas)
    cur = set(orig)
    for _ in range(pasos):
        cur = {k for k in cur if all(n in cur for n in vec[k])}
    for _ in range(pasos):
        cur = {n for k in cur for n in [k] + vec[k] if n in orig} | cur
    return cur


def componente_mayor(caras, activas):
    """Componente conexa (por aristas) de mas caras dentro de `activas`. -> set de indices."""
    vec = vecinos_de_caras(caras)
    activas = set(activas)
    visto, mejor = set(), set()
    for s in activas:
        if s in visto:
            continue
        comp, pila = {s}, [s]
        visto.add(s)
        while pila:
            k = pila.pop()
            for n in vec[k]:
                if n in activas and n not in visto:
                    visto.add(n)
                    comp.add(n)
                    pila.append(n)
        if len(comp) > len(mejor):
            mejor = comp
    return mejor


def lazo_ordenado(aristas):
    """Ordena aristas de borde (pares de indices) en lazos cerrados -> lista de listas de indices de vertice."""
    adj = {}
    for a, b in aristas:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    visto, lazos = set(), []
    for s in adj:
        if s in visto:
            continue
        lazo, prev, cur = [s], None, s
        visto.add(s)
        while True:
            sig = [n for n in adj[cur] if n != prev and n not in visto]
            if not sig:
                break
            prev, cur = cur, sig[0]
            visto.add(cur)
            lazo.append(cur)
        lazos.append(lazo)
    return lazos


def lazo_taubin(puntos, iteraciones=15, lam=0.5, mu=-0.53):
    """Suavizado de Taubin de un lazo cerrado de puntos (listas de coordenadas): quita el dentado sin encoger el contorno."""
    p = [list(map(float, q)) for q in puntos]
    n, d = len(p), len(p[0])
    for it in range(iteraciones * 2):
        f = lam if it % 2 == 0 else mu
        nuevo = []
        for i in range(n):
            a, b = p[i - 1], p[(i + 1) % n]
            nuevo.append([p[i][k] + f * ((a[k] + b[k]) / 2 - p[i][k]) for k in range(d)])
        p = nuevo
    return p


def proyectar_a_polilinea(p, poli, cerrada=True):
    """Punto mas cercano de p (n-D) a la polilinea `poli`. -> (indice de arista, t en 0..1, punto, distancia)."""
    n = len(poli)
    mejor = (None, 0.0, None, float("inf"))
    for i in range(n if cerrada else n - 1):
        a, b = poli[i], poli[(i + 1) % n]
        ab = [b[k] - a[k] for k in range(len(a))]
        l2 = sum(x * x for x in ab)
        t = 0.0 if l2 == 0 else max(0.0, min(1.0, sum((p[k] - a[k]) * ab[k] for k in range(len(a))) / l2))
        q = [a[k] + t * ab[k] for k in range(len(a))]
        d = math.dist(p, q)
        if d < mejor[3]:
            mejor = (i, t, q, d)
    return mejor


def emparejar_cadenas(a, b):
    """Emparejado monotono (tipo DTW) de dos secuencias de coordenadas ordenadas: cada elemento de ambas queda emparejado al menos
    una vez. -> lista de (i, j). Sirve para soldar el labio superior con el inferior de una boca abierta."""
    n, m = len(a), len(b)
    inf = float("inf")
    c = [[inf] * m for _ in range(n)]
    c[0][0] = abs(a[0] - b[0])
    for i in range(n):
        for j in range(m):
            if i == j == 0:
                continue
            prev = min(c[i - 1][j] if i else inf, c[i][j - 1] if j else inf, c[i - 1][j - 1] if i and j else inf)
            c[i][j] = prev + abs(a[i] - b[j])
    i, j, camino = n - 1, m - 1, [(n - 1, m - 1)]
    while (i, j) != (0, 0):
        opc = [(c[i - 1][j - 1], i - 1, j - 1) if i and j else None, (c[i - 1][j], i - 1, j) if i else None,
               (c[i][j - 1], i, j - 1) if j else None]
        _, i, j = min(o for o in opc if o)
        camino.append((i, j))
    return camino[::-1]


def separar_parpados(xz):
    """Dado el lazo ordenado de un ojo (lista de (x, z)), separa el borde del parpado superior y del inferior. Las comisuras son los
    puntos de x minima y maxima. -> (superior, inferior), listas de indices del lazo ordenadas por x creciente (incluyen las comisuras)."""
    n = len(xz)
    i0 = min(range(n), key=lambda i: xz[i][0])
    i1 = max(range(n), key=lambda i: xz[i][0])
    def tramo(a, b):
        t = [a]
        while t[-1] != b:
            t.append((t[-1] + 1) % n)
        return t
    t1, t2 = tramo(i0, i1), tramo(i1, i0)[::-1]
    z1 = sum(xz[i][1] for i in t1) / len(t1)
    z2 = sum(xz[i][1] for i in t2) / len(t2)
    sup, inf = (t1, t2) if z1 > z2 else (t2, t1)
    return sorted(sup, key=lambda i: xz[i][0]), sorted(inf, key=lambda i: xz[i][0])


def interpolar_lineal(xs, ys, x):
    """Interpolacion lineal de la curva (xs creciente, ys) en x (se sujeta en los extremos)."""
    if x <= xs[0]:
        return ys[0]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = 0.0 if xs[i] == xs[i - 1] else (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def desplazar_margenes(z_sup, z_inf, factor, reparto_sup=0.8):
    """Desplazamientos verticales (dz_sup, dz_inf) de los margenes de los parpados para que la apertura (z_sup - z_inf) se multiplique
    por `factor`; el parpado superior asume `reparto_sup` del cambio (el inferior se mueve menos)."""
    cambio = (z_sup - z_inf) * (factor - 1.0)
    return reparto_sup * cambio, -(1.0 - reparto_sup) * cambio


# ---------------------------------------------------------------- UV: alineacion y distorsion
def procrustes_2d(src, dst, reflexion=True):
    """Similitud 2D (con reflexion opcional) que lleva `src` a `dst` en minimos cuadrados. -> (matriz 2x2 con la escala incluida, t)."""
    n = len(src)
    cs = [sum(p[k] for p in src) / n for k in range(2)]
    cd = [sum(p[k] for p in dst) / n for k in range(2)]
    mejor = None
    for refl in ((False, True) if reflexion else (False,)):
        s = [[-(p[0] - cs[0]) if refl else p[0] - cs[0], p[1] - cs[1]] for p in src]
        d = [[p[0] - cd[0], p[1] - cd[1]] for p in dst]
        a = sum(s[i][0] * d[i][0] + s[i][1] * d[i][1] for i in range(n))
        b = sum(s[i][0] * d[i][1] - s[i][1] * d[i][0] for i in range(n))
        ss = sum(x * x + y * y for x, y in s) or 1.0
        c, sn = a / math.hypot(a, b) if a or b else 1.0, b / math.hypot(a, b) if a or b else 0.0
        esc = (a * c + b * sn) / ss
        # matriz = esc * R(theta) @ F
        f = -1.0 if refl else 1.0
        m = [[esc * c * f, -esc * sn], [esc * sn * f, esc * c]]
        err = sum((m[0][0] * s0 + m[0][1] * s1 - d0) ** 2 + (m[1][0] * s0 + m[1][1] * s1 - d1) ** 2
                  for (s0, s1), (d0, d1) in zip([[p[0] - cs[0], p[1] - cs[1]] for p in src], d))
        if mejor is None or err < mejor[0]:
            mejor = (err, m)
    m = mejor[1]
    t = [cd[0] - (m[0][0] * cs[0] + m[0][1] * cs[1]), cd[1] - (m[1][0] * cs[0] + m[1][1] * cs[1])]
    return m, t


def distorsion_uv(p3, uv, tris):
    """Calidad de un desenvuelto: por triangulo, el jacobiano 3D->UV tiene valores singulares s1>=s2; anisotropia = s1/s2 (1 = conforme)
    y cambio de area = log(area_uv/area_3d) normalizado por su mediana (0 = area uniforme). Cuenta tambien los triangulos invertidos
    respecto al signo mayoritario (orientacion UV). -> dict(aniso_media, aniso_p90, area_desv, invertidos, n)."""
    aniso, logs, signos = [], [], []
    for t in tris:
        a, b, c = [p3[i] for i in t]
        u, v, w = [uv[i] for i in t]
        e1 = [b[k] - a[k] for k in range(3)]
        e2 = [c[k] - a[k] for k in range(3)]
        l1 = math.sqrt(sum(x * x for x in e1))
        if l1 < 1e-15:
            continue
        x1 = [x / l1 for x in e1]
        d = sum(e2[k] * x1[k] for k in range(3))
        y = [e2[k] - d * x1[k] for k in range(3)]
        h = math.sqrt(sum(q * q for q in y))
        if h < 1e-15:
            continue
        # base local (l1,0), (d,h); jacobiano J = [du1 du2] @ inv([[l1,d],[0,h]])
        du1 = (v[0] - u[0], v[1] - u[1])
        du2 = (w[0] - u[0], w[1] - u[1])
        j00, j10 = du1[0] / l1, du1[1] / l1
        j01, j11 = (du2[0] - d * j00) / h, (du2[1] - d * j10) / h
        det = j00 * j11 - j01 * j10
        a_ = j00 * j00 + j10 * j10
        b_ = j00 * j01 + j10 * j11
        c_ = j01 * j01 + j11 * j11
        tr, disc = a_ + c_, math.sqrt(max(0.0, (a_ - c_) ** 2 + 4 * b_ * b_))
        l_max, l_min = (tr + disc) / 2, (tr - disc) / 2
        if abs(det) < 1e-18 or l_min <= 0:
            aniso.append(1e3)
            signos.append(0)
            continue
        aniso.append(math.sqrt(l_max / l_min))
        logs.append(math.log(abs(det)))
        signos.append(1 if det > 0 else -1)
    if not aniso:
        return dict(aniso_media=0.0, aniso_p90=0.0, area_desv=0.0, invertidos=0, n=0)
    aniso.sort()
    logs.sort()
    med = logs[len(logs) // 2]
    desv = math.sqrt(sum((x - med) ** 2 for x in logs) / len(logs))
    pos, neg = signos.count(1), signos.count(-1)
    return dict(aniso_media=sum(aniso) / len(aniso), aniso_p90=aniso[int(0.9 * (len(aniso) - 1))], area_desv=desv,
                invertidos=min(pos, neg) + signos.count(0), n=len(aniso))


def monotonizar_param(s, eps=1e-4):
    """Hace estrictamente creciente una secuencia de parametros de arco (maximo acumulado + separacion minima eps), cambiando lo minimo:
    los dientes del borde de un hueco (que van y vienen) se reparten en orden en vez de amontonarse."""
    out, ult = [], -float("inf")
    for v in s:
        ult = max(ult + eps, v) if out else v
        out.append(ult)
    return out


def desenrollar_param(s, longitud):
    """Desenrolla parametros de arco ciclicos (mod `longitud`) para que variaciones pequenas no den saltos de `longitud`."""
    out = [s[0]]
    for v in s[1:]:
        out.append(v + longitud * round((out[-1] - v) / longitud))
    return out


def siguiente_factor(historial, objetivo, minimo=0.2, maximo=1.5):
    """Siguiente factor de apertura para acercar la apertura medida a `objetivo`. `historial` = [(factor, medida), ...] (la medida crece con el
    factor). Con un solo punto, proporcional (objetivo/medida); con dos o mas, secante sobre los dos ultimos. Siempre dentro de [minimo, maximo]."""
    f1, m1 = historial[-1]
    if len(historial) == 1 or abs(historial[-1][1] - historial[-2][1]) < 1e-9:
        f = f1 * objetivo / max(m1, 1e-9)
    else:
        f0, m0 = historial[-2]
        f = f1 + (objetivo - m1) * (f1 - f0) / (m1 - m0)
    return min(max(f, minimo), maximo)


def giro_parpado(dy, dz, ddz):
    """Angulo (rad) que hay que girar un punto del parpado alrededor del eje X que pasa por el centro del globo para que su altura cambie en
    `ddz` conservando su distancia al centro (dy, dz = posicion relativa al centro; delante = -dy). Positivo = bajar. Asi el parpado rueda
    sobre el globo en vez de hundirse o separarse de el."""
    rho = math.hypot(dy, dz)
    if rho < 1e-12:
        return 0.0
    phi = math.atan2(dz, -dy)
    return phi - math.asin(max(-1.0, min(1.0, (dz + ddz) / rho)))
