"""Color de piel sin dependencias: lectura de tonos, sRGB -> CIELAB, diferencia de color (ΔE76 y ΔE2000) y el contraste amortiguado
con que `blender/remaster/tono_piel.py` lleva la piel de una textura a un tono objetivo."""
import math


def parse_rgb(texto):
    """'172,128,100' -> [172.0, 128.0, 100.0] (sRGB 0-255). Acepta tambien '#ac8064'."""
    t = texto.strip()
    if t.startswith("#") and len(t) == 7:
        return [float(int(t[i:i + 2], 16)) for i in (1, 3, 5)]
    v = [float(x) for x in t.split(",")]
    if len(v) != 3 or any(x < 0 or x > 255 for x in v):
        raise ValueError(f"color sRGB 0-255 invalido: {texto!r}")
    return v


def _lineal(c):
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def srgb_a_lab(rgb):
    """sRGB 0-255 -> CIELAB (D65)."""
    r, g, b = (_lineal(c) for c in rgb)
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883
    f = lambda t: t ** (1 / 3) if t > 216 / 24389 else (24389 / 27 * t + 16) / 116
    fx, fy, fz = f(x), f(y), f(z)
    return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)]


def delta_e76(rgb1, rgb2):
    a, b = srgb_a_lab(rgb1), srgb_a_lab(rgb2)
    return math.dist(a, b)


def delta_e2000(rgb1, rgb2):
    """CIEDE2000 entre dos colores sRGB 0-255."""
    l1, a1, b1 = srgb_a_lab(rgb1)
    l2, a2, b2 = srgb_a_lab(rgb2)
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    cm = (c1 + c2) / 2
    g = 0.5 * (1 - math.sqrt(cm ** 7 / (cm ** 7 + 25 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if c1p else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if c2p else 0.0
    dl = l2 - l1
    dc = c2p - c1p
    if c1p * c2p == 0:
        dh = 0.0
    else:
        d = h2p - h1p
        dh = d - 360 if d > 180 else d + 360 if d < -180 else d
    dH = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dh) / 2)
    lpm = (l1 + l2) / 2
    cpm = (c1p + c2p) / 2
    if c1p * c2p == 0:
        hpm = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hpm = (h1p + h2p) / 2
    else:
        hpm = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    t = (1 - 0.17 * math.cos(math.radians(hpm - 30)) + 0.24 * math.cos(math.radians(2 * hpm))
         + 0.32 * math.cos(math.radians(3 * hpm + 6)) - 0.20 * math.cos(math.radians(4 * hpm - 63)))
    sl = 1 + 0.015 * (lpm - 50) ** 2 / math.sqrt(20 + (lpm - 50) ** 2)
    sc = 1 + 0.045 * cpm
    sh = 1 + 0.015 * cpm * t
    rt = -2 * math.sqrt(cpm ** 7 / (cpm ** 7 + 25 ** 7)) * math.sin(math.radians(60 * math.exp(-(((hpm - 275) / 25) ** 2))))
    return math.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + (dH / sh) ** 2 + rt * (dc / sc) * (dH / sh))


def contraste_amortiguado(tono, media):
    """Factor c = sqrt(media(tono / media)) con tono y media en la misma escala (cualquiera). La razon pura amplificaria el ruido de la textura
    2-3 veces; la raiz conserva parte del detalle."""
    razones = [t / max(m, 1e-9) for t, m in zip(tono, media)]
    return math.sqrt(sum(razones) / len(razones))


def recolorear_pixel(rgb, media, tono, peso=1.0):
    """Un pixel (0-1 por canal): obj = tono + (rgb - media) * c, mezclado con `peso` (0 = sin cambio)."""
    c = contraste_amortiguado(tono, media)
    out = []
    for x, m, t in zip(rgb, media, tono):
        obj = t + (x - m) * c
        out.append(min(1.0, max(0.0, x + peso * (obj - x))))
    return out
