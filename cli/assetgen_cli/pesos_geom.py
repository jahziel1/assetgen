"""Geometria sin dependencias para la transferencia provisional de pesos de piel (`blender/remaster/transferir_pesos.py`):
coordenadas baricentricas, mapa hueso -> segmento derivado de la posicion de los vertices, pesos compatibles con el segmento,
suavizado en el grafo de la malla y estadistica del estiramiento de aristas. Pesos = dict {hueso: peso}; vectores = listas de 3 floats."""
import math

SEGMENTOS = ("daga", "cabeza", "capucha", "cuerpo", "brazo_I", "brazo_D", "pierna_I", "pierna_D")   # el orden es el codigo del atributo `segmento`


def _sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def coords_baricentricas(p, a, b, c):
    """(u, v, w) con p ~ u*a + v*b + w*c, para un p sobre (o muy cerca del plano de) el triangulo abc. Triangulo degenerado -> (1, 0, 0)."""
    v0, v1, v2 = _sub(b, a), _sub(c, a), _sub(p, a)
    d00, d01, d11, d20, d21 = _dot(v0, v0), _dot(v0, v1), _dot(v1, v1), _dot(v2, v0), _dot(v2, v1)
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-18:
        return (1.0, 0.0, 0.0)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return (1.0 - v - w, v, w)


def normalizar(w):
    """Pesos positivos que suman 1 (dict vacio o de suma 0 -> dict vacio)."""
    t = sum(x for x in w.values() if x > 0)
    return {h: x / t for h, x in w.items() if x > 0} if t > 1e-12 else {}


def limitar_pesos(w, n=3, permitidos=None, minimo=1e-4):
    """Los `n` mayores pesos entre los huesos `permitidos` (None = todos), renormalizados. Los pesos menores que `minimo` se descartan."""
    c = [(x, -h) for h, x in w.items() if x > minimo and (permitidos is None or h in permitidos)]
    c.sort(reverse=True)
    return normalizar({-m: x for x, m in c[:n]})


def interpolar_pesos(pesos_vertices, bary, permitidos, n=3):
    """Pesos de un punto de un triangulo: mezcla baricentrica de los de sus 3 vertices, solo con huesos `permitidos`."""
    mezcla = {}
    for w, b in zip(pesos_vertices, bary):
        for h, x in w.items():
            if h in permitidos:
                mezcla[h] = mezcla.get(h, 0.0) + b * x
    return limitar_pesos(mezcla, n, permitidos)


def mapa_hueso_segmento(huesos_pieza, centroides, x_hombro, z_hombro, umbral_brazo=0.75, margen_z=0.3, z_pierna=None):
    """Mapa hueso global -> segmento del cuerpo, a partir de las piezas del original y de la posicion de los vertices de cada hueso.

    huesos_pieza: {"cabeza": [...], "daga": [...], "faldon": [...]} huesos de las piezas sueltas (cabeza, arma, faldon/capucha que
      llevan su propio bind y no se mezclan con el cuerpo). centroides: {hueso: (x, y, z)} centroide ponderado por peso de sus vertices en
      el bind T (+X = izquierda del personaje). x_hombro, z_hombro: |x| y altura de la articulacion del hombro. Un hueso del resto es de
      brazo si su centroide cae a |x| >= umbral_brazo * x_hombro y no mas de `margen_z` por debajo del hombro (los huesos de cadera
      y falda laterales tienen la misma x pero van mucho mas abajo) (manga, antebrazo, mano; por esa regla quedan en el torso los huesos de la
      cintura escapular 17 y 18, que el original comparte con las piezas del brazo) y del cuerpo si no.
      z_pierna (opcional): los huesos de cuerpo con el centroide por debajo de esa altura pasan a pierna_I (x > 0) o pierna_D (x <= 0);
      por encima quedan en cuerpo (cadera, columna). Asi las dos piernas son segmentos distintos y ninguna toma huesos de la otra."""
    mapa = {}
    for seg, clave in (("cabeza", "cabeza"), ("daga", "daga"), ("capucha", "faldon")):
        for h in huesos_pieza.get(clave, []):
            mapa[h] = seg
    for h, c in centroides.items():
        if h in mapa:
            continue
        if abs(c[0]) >= umbral_brazo * x_hombro and c[2] >= z_hombro - margen_z:
            mapa[h] = "brazo_I" if c[0] > 0 else "brazo_D"
        elif z_pierna is not None and c[2] < z_pierna:
            mapa[h] = "pierna_I" if c[0] > 0 else "pierna_D"
        else:
            mapa[h] = "cuerpo"
    return mapa


def huesos_de(mapa, *segmentos):
    return sorted(h for h, s in mapa.items() if s in segmentos)


def suavizar_pesos(W, vecinos, grupo, fijos=(), iteraciones=6, lam=0.5, n=3):
    """Suaviza pesos por el grafo: cada vertice se mezcla con la media de sus vecinos del MISMO grupo (grupo[i] == grupo[j]);
    asi un segmento nunca toma pesos de otro. `fijos`: indices que no cambian. Cada vuelta se limita a `n` pesos y se renormaliza."""
    fijos = set(fijos)
    W = [dict(w) for w in W]
    for _ in range(iteraciones):
        nuevo = []
        for i, w in enumerate(W):
            vs = [j for j in vecinos[i] if grupo[j] == grupo[i]]
            if i in fijos or not vs:
                nuevo.append(w)
                continue
            m = {}
            for j in vs:
                for h, x in W[j].items():
                    m[h] = m.get(h, 0.0) + x / len(vs)
            mezcla = {h: (1 - lam) * w.get(h, 0.0) + lam * m.get(h, 0.0) for h in set(w) | set(m)}
            nuevo.append(limitar_pesos(mezcla, n) or w)
        W = nuevo
    return W


def percentil(valores, q):
    """Percentil q (0-100) con interpolacion lineal."""
    v = sorted(valores)
    if not v:
        raise ValueError("sin valores")
    k = (len(v) - 1) * q / 100.0
    i = int(math.floor(k))
    j = min(i + 1, len(v) - 1)
    return v[i] + (v[j] - v[i]) * (k - i)


def resumen_estiramiento(razones, qs=(1, 5, 50, 95, 99), alto=2.0, bajo=0.5):
    """razones = longitud de arista en pose / longitud en reposo. Percentiles y cuantas pasan de `alto` o bajan de `bajo`."""
    return {"percentiles": {str(q): round(percentil(razones, q), 4) for q in qs},
            "min": round(min(razones), 4), "max": round(max(razones), 4), "n": len(razones),
            "mayores": sum(1 for r in razones if r > alto), "menores": sum(1 for r in razones if r < bajo)}


def _m4_de_12(m):
    return [[m[0], m[3], m[6], m[9]], [m[1], m[4], m[7], m[10]], [m[2], m[5], m[8], m[11]], [0.0, 0.0, 0.0, 1.0]]


def _inv_afin(M):
    """Inversa de una afin 4x4 (base 3x3 invertible) por cofactores."""
    a, b, c = M[0][:3], M[1][:3], M[2][:3]
    cof = [[b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2], b[0] * c[1] - b[1] * c[0]],
           [c[1] * a[2] - c[2] * a[1], c[2] * a[0] - c[0] * a[2], c[0] * a[1] - c[1] * a[0]],
           [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]]
    det = a[0] * cof[0][0] + a[1] * cof[0][1] + a[2] * cof[0][2]
    inv = [[cof[j][i] / det for j in range(3)] for i in range(3)]
    t = [M[0][3], M[1][3], M[2][3]]
    return [inv[r] + [-sum(inv[r][k] * t[k] for k in range(3))] for r in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


def _mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def mul4_12(pose12, enlace12):
    """Matriz bind del modelo -> pantalla de un hueso: pose . enlace^-1 (12 floats cada una) -> 4x4."""
    return _mul(_m4_de_12(pose12), _inv_afin(_m4_de_12(enlace12)))


def movimiento_relativo(Mref, M):
    """(angulo en grados, traslacion) del movimiento del hueso M respecto al hueso de referencia: Mref^-1 . M."""
    R = _mul(_inv_afin(Mref), M)
    d = (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
         + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))
    esc = abs(d) ** (1.0 / 3.0) or 1.0
    c = max(-1.0, min(1.0, ((R[0][0] + R[1][1] + R[2][2]) / esc - 1.0) / 2.0))
    return math.degrees(math.acos(c)), math.sqrt(R[0][3] ** 2 + R[1][3] ** 2 + R[2][3] ** 2)


def suave(x, a, b):
    """Escalon suave (smoothstep) de 0 en x<=a a 1 en x>=b."""
    if b == a:
        return 1.0 if x >= b else 0.0
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3.0 - 2.0 * t)


def mezclar_pesos(partes, n=3):
    """Combinacion lineal de pesos: partes = [(coeficiente, {hueso: peso})]; devuelve los `n` mayores renormalizados."""
    tot = {}
    for c, w in partes:
        for h, x in w.items():
            tot[h] = tot.get(h, 0.0) + c * x
    return limitar_pesos(tot, n) or {}


def pesos_entrepierna(w_cadera, w_izq, w_der, x, z, z_ent, banda=0.08, caida_z=0.08, peso_cadera=0.5, n=3):
    """Pesos de un vertice del cuerpo en la zona de las piernas. w_cadera = pesos solo con huesos de cadera/columna; w_izq / w_der = pesos
    con cadera + huesos de la pierna izquierda (x > 0) / derecha. x = lado respecto al plano medio, z = altura, z_ent = altura de la entrepierna
    (donde las dos piernas se juntan). Por debajo de la entrepierna (mas de `caida_z`) cada pierna es un segmento puro, separado por el plano
    x = 0; en la banda |x| < `banda` por encima de la entrepierna se mezcla la pierna de cada lado con la cadera: hasta `peso_cadera` de cadera
    en el plano medio y el resto repartido entre las dos piernas, de forma continua."""
    kz = suave(z, z_ent - caida_z, z_ent)                       # 0 bien por debajo de la entrepierna, 1 a su altura o por encima
    ban = max(banda * kz, 1e-9)
    t = suave(x, -ban, ban)                                      # 0 = pierna derecha pura, 1 = izquierda pura
    u = kz * (1.0 - suave(abs(x), 0.0, banda))                  # peso de la cadera: maximo en el plano medio, 0 fuera de la banda
    return mezclar_pesos([((1.0 - u * peso_cadera) * t, w_izq), ((1.0 - u * peso_cadera) * (1.0 - t), w_der), (u * peso_cadera, w_cadera)], n)
