"""Imagen y proyeccion para texturizar una cara 3D con un retrato (numpy; no depende de ningun juego ni de Blender).

La usan blender/remaster/cara_textura.py (dentro de Blender, con su numpy) y cara_textura_img.py (venv con numpy/opencv).
Contenido: color sRGB<->CIELab y transferencia de media/desviacion en Lab, camara afin por minimos cuadrados (3D -> pixeles),
deformacion thin-plate 2D (TPS), muestreo bilineal, desenfoque gaussiano y alta frecuencia, rasterizado de triangulos en el espacio UV
y relleno de huecos por dilatacion.

Convencion de imagenes: arreglos (alto, ancho, canales) con la fila 0 = ARRIBA, salvo `rasterizar_triangulos`, que devuelve una rejilla
UV con la fila 0 = v 0 (abajo, como las imagenes de Blender)."""
import numpy as np

_D65 = np.array([0.95047, 1.0, 1.08883])
_M_RGB_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])


def suavizado(t):
    """smoothstep de t en [0, 1] (se recorta)."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------- color
def srgb_a_lineal(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lineal_a_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def rgb_a_lab(rgb):
    """sRGB 0..1 (..., 3) -> CIELab D65 (..., 3)."""
    xyz = srgb_a_lineal(rgb) @ _M_RGB_XYZ.T / _D65
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def lab_a_rgb(lab):
    """CIELab D65 -> sRGB 0..1 (recortado)."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f ** 3 > 216 / 24389, f ** 3, (116 * f - 16) * 27 / 24389) * _D65
    return lineal_a_srgb(xyz @ np.linalg.inv(_M_RGB_XYZ).T)


def delta_e76(a, b):
    return np.sqrt(((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2).sum(axis=-1))


def es_piel_lab(lab, l_rango=(35.0, 92.0), tono_grados=(15.0, 75.0), croma_rango=(6.0, 55.0)):
    """Mascara (...,) de pixeles Lab con aspecto de piel: luminosidad media-alta, tono entre rojizo y amarillento y croma moderado. Deja fuera
    pelo y barba (oscuros), telas azules (tono negativo), dorados (tono > 75 y croma alto) y los blancos del ojo (croma bajo)."""
    lab = np.asarray(lab, dtype=np.float64)
    croma = np.hypot(lab[..., 1], lab[..., 2])
    tono = np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))
    return ((lab[..., 0] >= l_rango[0]) & (lab[..., 0] <= l_rango[1]) & (tono >= tono_grados[0]) & (tono <= tono_grados[1]) &
            (croma >= croma_rango[0]) & (croma <= croma_rango[1]))


def estadistica_lab(lab, mascara=None, tol_de=None, iteraciones=3):
    """Media y desviacion por canal de los pixeles Lab (n, 3) o (h, w, 3) dentro de `mascara`. Con `tol_de` es robusta: parte de la
    mediana y descarta los pixeles a mas de `tol_de` unidades de Lab de la media (pelo, sombras, barba), `iteraciones` veces."""
    lab = np.asarray(lab, dtype=np.float64)
    px = lab[mascara] if mascara is not None else lab.reshape(-1, 3)
    if len(px) == 0:
        raise ValueError("sin pixeles para la estadistica")
    if tol_de is not None:
        centro = np.median(px, axis=0)
        for _ in range(iteraciones):
            sel = delta_e76(px, centro) <= tol_de
            if sel.sum() < 16:
                break
            centro = px[sel].mean(axis=0)
        sel = delta_e76(px, centro) <= tol_de
        if sel.sum() >= 16:
            px = px[sel]
    return dict(media=px.mean(axis=0).tolist(), desv=px.std(axis=0).tolist(), n=int(len(px)))


def transferir_lab(lab, origen, destino, razon_min=0.5, razon_max=1.5):
    """Lleva la media y la desviacion de `origen` a las de `destino` (dicts de estadistica_lab), por canal: (lab - m_o) * r + m_d, con
    r = desv_d / desv_o recortada a [razon_min, razon_max] (para no exagerar el contraste ni el croma)."""
    mo, mdst = np.array(origen["media"]), np.array(destino["media"])
    r = np.clip(np.array(destino["desv"]) / np.maximum(np.array(origen["desv"]), 1e-6), razon_min, razon_max)
    return (np.asarray(lab, dtype=np.float64) - mo) * r + mdst


# ---------------------------------------------------------------- camara afin y TPS
def ajustar_camara_afin(p3, p2):
    """Camara afin 3D -> 2D por minimos cuadrados: [u, v] = A @ [x, y, z, 1]. p3 (n, 3), p2 (n, 2) -> A (2, 4)."""
    p3 = np.asarray(p3, dtype=np.float64)
    h = np.hstack([p3, np.ones((len(p3), 1))])
    sol, *_ = np.linalg.lstsq(h, np.asarray(p2, dtype=np.float64), rcond=None)
    return sol.T


def proyectar_afin(A, p3):
    p3 = np.asarray(p3, dtype=np.float64)
    return np.hstack([p3, np.ones((len(p3), 1))]) @ np.asarray(A).T


def _u_tps(r2):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r2 > 0, 0.5 * r2 * np.log(r2), 0.0)       # r^2 log r


def tps_ajustar(src, dst, lam=0.0):
    """Thin-plate spline 2D src (n, 2) -> dst (n, 2) con termino afin; `lam` > 0 regulariza (no interpola exacto)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    d2 = ((src[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    K = _u_tps(d2) + lam * np.eye(n)
    P = np.hstack([np.ones((n, 1)), src])
    L = np.zeros((n + 3, n + 3))
    L[:n, :n], L[:n, n:], L[n:, :n] = K, P, P.T
    Y = np.vstack([dst, np.zeros((3, 2))])
    sol = np.linalg.solve(L, Y)
    return dict(src=src, w=sol[:n], a=sol[n:])


def tps_evaluar(m, pts, trozo=20000):
    pts = np.asarray(pts, dtype=np.float64)
    out = np.empty_like(pts)
    for i in range(0, len(pts), trozo):
        q = pts[i:i + trozo]
        U = _u_tps(((q[:, None, :] - m["src"][None, :, :]) ** 2).sum(-1))
        out[i:i + trozo] = np.hstack([np.ones((len(q), 1)), q]) @ m["a"] + U @ m["w"]
    return out


def tps_loo(src, dst, lam=0.0):
    """Error (n,) por validacion dejando uno fuera: distancia entre dst[i] y la TPS ajustada sin el punto i."""
    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    e = np.empty(len(src))
    for i in range(len(src)):
        k = np.arange(len(src)) != i
        e[i] = np.linalg.norm(tps_evaluar(tps_ajustar(src[k], dst[k], lam), src[i:i + 1])[0] - dst[i])
    return e


# ---------------------------------------------------------------- muestreo y filtros
def muestrear_bilineal(img, x, y):
    """img (h, w) o (h, w, c); x, y en pixeles con el centro del pixel (0, 0) en (0, 0); se recortan al borde. -> (len(x),) o (len(x), c)."""
    h, w = img.shape[:2]
    x = np.clip(np.asarray(x, dtype=np.float64), 0, w - 1)
    y = np.clip(np.asarray(y, dtype=np.float64), 0, h - 1)
    x0, y0 = np.minimum(np.floor(x).astype(int), w - 2), np.minimum(np.floor(y).astype(int), h - 2)
    fx, fy = x - x0, y - y0
    if img.ndim == 3:
        fx, fy = fx[..., None], fy[..., None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy) +
            img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def nucleo_gauss(sigma):
    r = max(1, int(np.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return k / k.sum()


def desenfoque_gauss(img, sigma):
    """Gaussiano separable (2D o 3D, bordes reflejados)."""
    img = np.asarray(img, dtype=np.float64)
    k = nucleo_gauss(sigma)
    r = len(k) // 2
    for eje in (0, 1):
        pad = [(0, 0)] * img.ndim
        pad[eje] = (r, r)
        p = np.pad(img, pad, mode="reflect")
        out = np.zeros_like(img)
        for i, c in enumerate(k):
            sl = [slice(None)] * img.ndim
            sl[eje] = slice(i, i + img.shape[eje])
            out += c * p[tuple(sl)]
        img = out
    return img


def alta_frecuencia(lum, sigma):
    """Luminancia menos su desenfoque gaussiano: solo el detalle fino (media ~0)."""
    lum = np.asarray(lum, dtype=np.float64)
    return lum - desenfoque_gauss(lum, sigma)


# ---------------------------------------------------------------- UV
def rasterizar_triangulos(uv, tris, n, tol=1e-4, prioridad=None):
    """Rasteriza triangulos UV (uv (m, 2) en 0..1, tris (t, 3)) en una rejilla n x n (fila j = v, columna i = u, centros de texel).
    -> (tri (n, n) int32 con -1 donde no hay nada, bar (n, n, 3) float32 baricentricas). Donde se solapan gana el de mayor `prioridad`
    (por triangulo; p. ej. el mas cercano a la camara) y, sin ella, el ultimo."""
    uv = np.asarray(uv, dtype=np.float64) * n
    tri_id = np.full((n, n), -1, dtype=np.int32)
    bar = np.zeros((n, n, 3), dtype=np.float32)
    orden = range(len(tris)) if prioridad is None else np.argsort(np.asarray(prioridad), kind="stable")
    for t in orden:
        a, b, c = tris[t]
        p0, p1, p2 = uv[a], uv[b], uv[c]
        det = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
        if abs(det) < 1e-12:
            continue
        i0, i1 = int(max(0, np.floor(min(p0[0], p1[0], p2[0]) - 1))), int(min(n - 1, np.ceil(max(p0[0], p1[0], p2[0]) + 1)))
        j0, j1 = int(max(0, np.floor(min(p0[1], p1[1], p2[1]) - 1))), int(min(n - 1, np.ceil(max(p0[1], p1[1], p2[1]) + 1)))
        if i1 < i0 or j1 < j0:
            continue
        gx, gy = np.meshgrid(np.arange(i0, i1 + 1) + 0.5, np.arange(j0, j1 + 1) + 0.5)
        l1 = ((p1[1] - p2[1]) * (gx - p2[0]) + (p2[0] - p1[0]) * (gy - p2[1])) / det
        l2 = ((p2[1] - p0[1]) * (gx - p2[0]) + (p0[0] - p2[0]) * (gy - p2[1])) / det
        l3 = 1 - l1 - l2
        dentro = (l1 >= -tol) & (l2 >= -tol) & (l3 >= -tol)
        jj, ii = np.nonzero(dentro)
        tri_id[j0 + jj, i0 + ii] = t
        bar[j0 + jj, i0 + ii] = np.stack([l1[jj, ii], l2[jj, ii], l3[jj, ii]], axis=-1)
    return tri_id, bar


def rellenar_vacios(img, valido, iteraciones=32):
    """Rellena los pixeles no validos con el vecino valido mas cercano (dilatacion de 4 vecinos). img (h, w, c) o (h, w)."""
    img = np.array(img, copy=True)
    val = np.array(valido, dtype=bool, copy=True)
    for _ in range(iteraciones):
        if val.all():
            break
        nuevo = np.zeros_like(val)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            vs = np.roll(val, (dy, dx), axis=(0, 1))
            src = np.roll(img, (dy, dx), axis=(0, 1))
            tomar = vs & ~val & ~nuevo
            img[tomar] = src[tomar]
            nuevo |= tomar
        val |= nuevo
    return img
