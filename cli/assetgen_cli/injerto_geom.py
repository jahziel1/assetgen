"""Geometria sin dependencias de `blender/remaster/injertar_cabeza.py`: modelo del borde de una capucha (altura en funcion del
azimut alrededor del eje de la cabeza, serie de Fourier), ajuste robusto por minimos cuadrados, angulos de Euler y similitudes."""
import math
import random


def rasgos_borde(x, y, y0, orden=4):
    """[1, cos t, sin t, cos 2t, sin 2t, ...] con t = atan2(y - y0, x)."""
    t = math.atan2(y - y0, x)
    f = [1.0]
    for k in range(1, orden + 1):
        f += [math.cos(k * t), math.sin(k * t)]
    return f


def resolver(a, b):
    """Resuelve a x = b (gauss con pivoteo parcial). a: n x n (se copia)."""
    n = len(b)
    m = [list(map(float, a[i])) + [float(b[i])] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-12:
            raise ValueError("sistema singular")
        m[c], m[p] = m[p], m[c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            for k in range(c, n + 1):
                m[r][k] -= f * m[c][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (m[i][n] - sum(m[i][k] * x[k] for k in range(i + 1, n))) / m[i][i]
    return x


def _minimos_cuadrados(filas, z, ridge=1e-9):
    n = len(filas[0])
    ata = [[sum(f[i] * f[j] for f in filas) for j in range(n)] for i in range(n)]
    for i in range(n):
        ata[i][i] += ridge
    atb = [sum(f[i] * zz for f, zz in zip(filas, z)) for i in range(n)]
    return resolver(ata, atb)


def z_borde(coefs, x, y, orden=4):
    """Altura del borde en (x, y). coefs = [y0, c0, a1, b1, ...] (lo que devuelve `ajustar_borde`)."""
    f = rasgos_borde(x, y, coefs[0], orden)
    return sum(c * v for c, v in zip(coefs[1:], f))


def ajustar_borde(puntos, orden=4, iteraciones=8, tol_min=0.008, max_puntos=4000, semilla=1):
    """Ajusta z = c0 + sum_k (a_k cos k t + b_k sin k t) a puntos (x, y, z) del ribete, con recorte robusto de atipicos (se descarta
    el 30 % de mayor residuo en cada pasada salvo las 2 ultimas, que usan `tol_min`). y0 (centro del lazo) se recalcula con los
    puntos buenos. -> (coefs [y0, c0, ...], rms, usados)."""
    pts = list(puntos)
    if len(pts) < 2 * orden + 8:
        raise ValueError("pocos puntos para ajustar el borde")
    if len(pts) > max_puntos:
        pts = random.Random(semilla).sample(pts, max_puntos)
    y0 = sum(p[1] for p in pts) / len(pts)
    bueno = [True] * len(pts)
    coefs = None
    for it in range(iteraciones):
        sel = [p for p, b in zip(pts, bueno) if b]
        coefs = _minimos_cuadrados([rasgos_borde(p[0], p[1], y0, orden) for p in sel], [p[2] for p in sel])
        res = [abs(p[2] - sum(c * v for c, v in zip(coefs, rasgos_borde(p[0], p[1], y0, orden)))) for p in pts]
        usados = sorted(r for r, b in zip(res, bueno) if b)
        thr = usados[int(0.7 * (len(usados) - 1))] if it < iteraciones - 2 else 0.0
        bueno = [r < max(thr, tol_min) for r in res]
        if sum(bueno) < 2 * orden + 8:
            raise ValueError("el recorte de atipicos dejo muy pocos puntos")
        y0 = sum(p[1] for p, b in zip(pts, bueno) if b) / sum(bueno)
    sel = [p for p, b in zip(pts, bueno) if b]
    coefs = _minimos_cuadrados([rasgos_borde(p[0], p[1], y0, orden) for p in sel], [p[2] for p in sel])
    r2 = [(p[2] - sum(c * v for c, v in zip(coefs, rasgos_borde(p[0], p[1], y0, orden)))) ** 2 for p in sel]
    return [y0] + coefs, math.sqrt(sum(r2) / len(r2)), len(sel)


def matriz_euler_xyz(rx, ry, rz):
    """Grados -> matriz 3x3 R = Rz Ry Rx."""
    ax, ay, az = (math.radians(v) for v in (rx, ry, rz))
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    return [[cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx]]


def euler_xyz(r):
    """Matriz 3x3 -> grados (rx, ry, rz) con R = Rz Ry Rx."""
    ry = math.asin(max(-1.0, min(1.0, -r[2][0])))
    return [round(math.degrees(v), 3) for v in (math.atan2(r[2][1], r[2][2]), ry, math.atan2(r[1][0], r[0][0]))]


def aplicar_similitud(s, r, t, p):
    return [s * sum(r[i][j] * p[j] for j in range(3)) + t[i] for i in range(3)]


def invertir_similitud(s, r, t):
    """Inversa de p -> s R p + t:  q -> (1/s) R^T (q - t)  =  s' R' q + t'."""
    rt = [[r[j][i] for j in range(3)] for i in range(3)]
    si = 1.0 / s
    ti = [-si * sum(rt[i][j] * t[j] for j in range(3)) for i in range(3)]
    return si, rt, ti
