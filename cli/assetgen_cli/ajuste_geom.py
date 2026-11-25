"""Geometria sin dependencias para ajustar las proporciones de un personaje a las de un original (`blender/remaster/referencias.py`,
`ajustar_proporciones.py`): interpolacion por tramos de un mapa vertical, matriz de un tramo (giro + estiramiento axial), cambio de largo
de tramos, errores por referencia y traslado de las articulaciones del juego a la malla ajustada. Vectores = listas de 3 floats;
matrices 3x3 = listas de filas."""
import math

from . import pose_geom as G


def interp(x, xs, ys, fuera="plano"):
    """Interpolacion lineal por tramos con xs creciente. `fuera`: 'plano' (constante) o 'pendiente1' (y = y_extremo + (x - x_extremo))."""
    if x <= xs[0]:
        return ys[0] + ((x - xs[0]) if fuera == "pendiente1" else 0.0)
    if x >= xs[-1]:
        return ys[-1] + ((x - xs[-1]) if fuera == "pendiente1" else 0.0)
    for i in range(1, len(xs)):
        if x <= xs[i]:
            f = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + f * (ys[i] - ys[i - 1])
    return ys[-1]


def matriz_tramo(p0, p1, q0, q1, estirar=True):
    """Mapa afin p -> A (p - p0) + q0 que lleva el tramo p0->p1 sobre q0->q1: giro de arco minimo entre las direcciones y, si `estirar`,
    estiramiento a lo largo del eje por |q1-q0| / |p1-p0| (lo perpendicular no cambia). Devuelve (A, t) con p' = A p + t."""
    u = G.unitario(G.sub(p1, p0))
    v = G.sub(q1, q0)
    k = G.norma(v) / G.norma(G.sub(p1, p0)) if estirar else 1.0
    R = G.rot_entre(u, v)
    S = [[(1.0 if i == j else 0.0) + (k - 1.0) * u[i] * u[j] for j in range(3)] for i in range(3)]
    A = G.mat_mul(R, S)
    Ap0 = G.mat_vec(A, p0)
    return A, [q0[i] - Ap0[i] for i in range(3)]


def aplica(m, p):
    A, t = m
    r = G.mat_vec(A, p)
    return [r[i] + t[i] for i in range(3)]


def largo(a, b):
    return G.norma(G.sub(a, b))


def cambio_pct(antes, despues):
    return (despues / antes - 1.0) * 100.0


def largos_tramos(refs, tramos):
    """refs: {nombre: [x,y,z]}; tramos: {nombre: (ref_a, ref_b)} -> largos en m (3D)."""
    return {k: largo(refs[a], refs[b]) for k, (a, b) in tramos.items()}


def error_referencia(a, b):
    """Error entre dos valores de referencia: |a-b| si son escalares, distancia si son puntos 3D (m)."""
    if isinstance(a, (int, float)):
        return abs(a - b)
    return largo(a, b)


def errores_cm(refs_a, refs_b, claves=None):
    claves = claves or sorted(k for k in refs_a if k in refs_b)
    return {k: round(error_referencia(refs_a[k], refs_b[k]) * 100, 2) for k in claves}


def desfases(joints, refs, pares):
    """Desfase (articulacion - referencia) por par {articulacion: referencia}; en m."""
    return {j: G.sub(joints[j], refs[r]) for j, r in pares.items()}


def trasladar_juego(refs_ajustado, desf, pares):
    """Articulaciones del juego en la malla ajustada: referencia medida en la malla ajustada + desfase del original."""
    return {j: G.add(refs_ajustado[r], desf[j]) for j, r in pares.items()}
