"""injertar_cabeza.py -- injerta una cabeza (con capucha/tocado) sobre un cuerpo, ambos glb de personaje, y deja el conjunto a escala.

    blender -b -P injertar_cabeza.py -- --cuerpo cuerpo.glb --cabeza cabeza.glb --salida DIR [--nombre n] [--alto 1.75] [...]

Pensado para modelos generados por imagen->3D (una sola malla por glb, textura propia): el cuerpo trae su cabeza y la cabeza
elegida llega con hombros/pecho de sobra. Los dos tienen la capucha (o tocado) con un ribete de otro color en el borde inferior.
Metodo:
  1. Borde de la capucha en la cabeza: el ribete es un lazo alrededor del cuello, asi que su altura se modela como funcion del azimut
     z = f(theta) (serie de Fourier de orden 4, ver assetgen_cli/injerto_geom.py) y se ajusta a los triangulos del ribete (color con
     matiz/saturacion/brillo en `--oro`) dentro de la caja `--zona-borde x_max,z_min,z_max`, con recorte robusto de atipicos (el
     ribete del chaleco, las costuras). La cabeza conserva lo que queda por encima de f - `--margen-cabeza`, dentro de |x| < x_max,
     y se descartan las islas pequenas.
  2. Ajuste de la cabeza al cuerpo: similitud (escala, giro, traslacion) por ICP recortado entre la cabeza conservada y la parte
     alta del cuerpo (fraccion `--region-cuerpo` del alto); varias escalas iniciales, gana la de menor error.
  3. Corte del cuerpo: se quita lo que queda dentro de la huella de la capucha (|x_cabeza| < x_max) por encima de f' + `--solape`,
     con f' = f llevada al espacio del cuerpo. Entre ambos cortes queda un solape (la capucha monta sobre el cuerpo).
  3b. Corte frontal: por delante (sector en azimut `--corte-frente sector,transicion,bajada,fundido_x,eje`, ver abajo) el corte de la
     cabeza y el del cuerpo bajan `bajada` unidades de la cabeza (con transicion suave en azimut y fundido en x): la cabeza conserva su
     cuello y el arranque del pecho hasta la V del chaleco y el cuerpo pierde lo que quedaria detras, de modo que el cuello de la cabeza
     tapa la union. El ajuste ICP usa la cabeza sin ese corte. Se quitan ademas los triangulos muy estirados (`--estirado`) y las islas.
  4. Se unen en una malla con dos materiales (cuerpo y cabeza), se escala a `--alto` con los pies en z = 0, centrado en x/y,
     mirando a -Y, y se exportan <nombre>.blend y <nombre>.glb. Se guarda <nombre>_ajuste.json (escala, giro, errores).
Parametros del corte frontal (unidades de la cabeza): sector = semiancho en grados alrededor de -Y (50), transicion = grados de fundido
suave (30), bajada = cuanto baja el corte bajo el ribete (0.12), fundido_x = ancho del fundido antes del borde |x| < x_max (0.08),
eje = desplazamiento hacia atras del vertice del sector respecto al centro del lazo del ribete (0.12; el cuello queda detras de el).
Imprime 'INJERTO {resumen}'."""
import json
import math
import os
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Matrix
from mathutils.kdtree import KDTree

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "..", "..", "cli"))
import tono_piel as T  # noqa: E402  (hsv, _malla, _imagen_base)
from assetgen_cli import injerto_geom as IG  # noqa: E402


# ---------------------------------------------------------------- lectura
def importar(ruta, nombre):
    antes = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=ruta)
    ob = [o for o in bpy.data.objects if o not in antes and o.type == "MESH"][0]
    ob.name = nombre
    ob.data.name = nombre
    return ob


def color_triangulos(ob):
    """-> (co mundo (n,3), uv, lt, vt, color base (nt,3) en el centro de cada triangulo)."""
    nodo = T._imagen_base(ob.data)
    img = nodo.image
    W, H = img.size
    px = np.empty(W * H * 4, np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(H, W, 4)
    co, uv, lt, vt = T._malla(ob)
    uc = uv[lt].mean(1)
    c = px[np.clip((uc[:, 1] * H).astype(int), 0, H - 1), np.clip((uc[:, 0] * W).astype(int), 0, W - 1), :3]
    return co, uv, lt, vt, c


# ---------------------------------------------------------------- superficie del borde
# El borde de la capucha es un lazo alrededor del cuello: su altura es funcion del azimut theta = atan2(y - y0, x) alrededor del eje de
# la cabeza. Modelo: z = c0 + sum_k (a_k cos k theta + b_k sin k theta), k = 1..ORDEN; c = [y0, c0, a1, b1, a2, b2, ...].
ORDEN = 4


def z_borde(c, P):
    """Altura del borde (`injerto_geom.z_borde`, vectorizada) en los puntos P (n,3)."""
    P = np.asarray(P)
    th = np.arctan2(P[:, 1] - c[0], P[:, 0])
    z = np.full(len(P), c[1], np.float64)
    for k in range(1, ORDEN + 1):
        z += c[2 * k] * np.cos(k * th) + c[2 * k + 1] * np.sin(k * th)
    return z


# ---------------------------------------------------------------- similitud / ICP
def umeyama(src, dst, escalar=True):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    A, B = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(B.T @ A / len(src))
    d = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[2] = -1
    R = U @ np.diag(d) @ Vt
    s = (S * d).sum() / (A ** 2).sum() * len(src) if escalar else 1.0
    return s, R, mu_d - s * R @ mu_s


def vecinos(kd, X):
    idx = np.empty(len(X), np.int64)
    dist = np.empty(len(X))
    for i, p in enumerate(X):
        _, j, d = kd.find(p)
        idx[i] = j
        dist[i] = d
    return idx, dist


def icp(src, dst, s0, R0, t0, iteraciones=30, recorte=0.7):
    kd = KDTree(len(dst))
    for i, p in enumerate(dst):
        kd.insert(p, i)
    kd.balance()
    s, R, t = s0, R0, t0
    err = None
    for it in range(iteraciones):
        X = s * src @ R.T + t
        idx, d = vecinos(kd, X)
        k = d <= np.quantile(d, recorte)
        s, R, t = umeyama(src[k], dst[idx[k]])
        err = float(np.sqrt((d[k] ** 2).mean()))
    X = s * src @ R.T + t
    idx, d = vecinos(kd, X)
    return s, R, t, d


# ---------------------------------------------------------------- edicion de mallas
def borrar_triangulos(ob, mascara_tri):
    """Borra de `ob` los triangulos de loop_triangles marcados; los poligonos tocados se borran enteros (poligono = triangulo aqui)."""
    me = ob.data
    pol_de_tri = np.empty(len(me.loop_triangles), np.int32)
    me.loop_triangles.foreach_get("polygon_index", pol_de_tri)
    borrar = np.zeros(len(me.polygons), bool)
    borrar[pol_de_tri[mascara_tri]] = True
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(me)
    bm.faces.ensure_lookup_table()
    bpy.ops.mesh.select_all(action="DESELECT")
    cara = [bm.faces[i] for i in np.nonzero(borrar)[0]]
    bmesh.ops.delete(bm, geom=cara, context="FACES")
    bmesh.update_edit_mesh(me)
    bpy.ops.object.mode_set(mode="OBJECT")
    return int(borrar.sum())


def islas_pequenas(ob, min_frac=0.02):
    """Mascara (sobre loop_triangles) de los triangulos que pertenecen a islas (partes sueltas, con los vertices soldados por posicion)
    de menos de `min_frac` de los triangulos de la isla mayor. -> (mascara, lista de (triangulos, bbox min, bbox max))."""
    co, _, _, vt = T._malla(ob)
    llave = np.round(co.astype(np.float64) * 1e5).astype(np.int64)
    _, w = np.unique(llave, axis=0, return_inverse=True)
    w = w.ravel()
    tri = w[vt]
    lab = np.arange(w.max() + 1)
    while True:
        m = lab[tri].min(1)
        nuevo = lab.copy()
        for k in range(3):
            np.minimum.at(nuevo, tri[:, k], m)
        nuevo = nuevo[nuevo]
        if np.array_equal(nuevo, lab):
            break
        lab = nuevo
    tl = lab[tri[:, 0]]
    ids, cnt = np.unique(tl, return_counts=True)
    chicas = ids[cnt < min_frac * cnt.max()]
    mask = np.isin(tl, chicas)
    info = []
    for i in chicas:
        k = tl == i
        v = co[vt[k]].reshape(-1, 3)
        info.append((int(k.sum()), v.min(0).round(3).tolist(), v.max(0).round(3).tolist()))
    print("  islas:", len(ids), "mayor", int(cnt.max()), "pequenas", len(chicas))
    return mask, info


def tris_estirados(ob, k, zona=None):
    """Mascara (sobre loop_triangles) de los triangulos cuya arista mayor supera `k` veces la mediana de las aristas mayores del objeto
    (caras muy estiradas, tipicas de los colgajos de un corte), solo donde `zona(centros)` sea cierto (todo el objeto si es None).
    k <= 0 la desactiva."""
    co, _, _, vt = T._malla(ob)
    if k <= 0:
        return np.zeros(len(vt), bool)
    P = co[vt].astype(np.float64)
    emax = np.max(np.stack([np.linalg.norm(P[:, i] - P[:, (i + 1) % 3], axis=1) for i in range(3)], 1), 1)
    m = emax > k * np.median(emax)
    return m if zona is None else m & zona(P.mean(1))


def aplicar_matriz(ob, M):
    ob.matrix_world = M @ ob.matrix_world
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def bajada_frente(P, y_eje, xmax, sector, trans, baja, fx):
    """Cuanto se baja el corte (unidades de la cabeza) en los puntos P (n,3): `baja` por delante (sector en grados alrededor de -Y),
    con transicion suave en azimut y un fundido en x para que muera antes del borde de la huella."""
    th = np.arctan2(P[:, 1] - y_eje, P[:, 0])
    w = np.zeros(len(P))
    k = np.abs(P[:, 0]) < xmax
    sr, tr = math.radians(sector), math.radians(trans)
    w[k] = [IG.peso_frente(t, sr, tr) * IG.peso_lateral(x, xmax, fx) for t, x in zip(th[k], P[k, 0])]
    return baja * w


def vertices(ob):
    me = ob.data
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3)


# ---------------------------------------------------------------- referencias de la cara
def punta_nariz(pts, z_min):
    """Altura de la punta de la nariz: vertice de menor y (el personaje mira a -Y) cerca del plano medio, por encima de z_min."""
    cx = np.median(pts[:, 0])
    ancho = pts[:, 0].max() - pts[:, 0].min()
    k = (pts[:, 2] > z_min) & (np.abs(pts[:, 0] - cx) < 0.04 * ancho)
    p = pts[k]
    return float(p[np.argmin(p[:, 1]), 2])


def menton_por_perfil(pts, z_nariz, umbral=0.1):
    """Altura del menton (barba) como el punto donde, bajando desde la nariz, el perfil frontal retrocede de golpe (cuello dentro de la
    capucha). Devuelve None si no hay salto. `umbral` es fraccion de la distancia coronilla-nariz."""
    top = pts[:, 2].max()
    paso = 0.01 * (top - z_nariz)
    cx = np.median(pts[:, 0])
    ancho = pts[:, 0].max() - pts[:, 0].min()
    sel = np.abs(pts[:, 0] - cx) < 0.04 * ancho
    z = z_nariz
    prev = None
    while z > z_nariz - 3 * (top - z_nariz):
        k = sel & (pts[:, 2] <= z) & (pts[:, 2] > z - paso)
        if k.any():
            y = pts[k, 1].min()
            if prev is not None and y - prev > umbral * (top - z_nariz):
                return float(z + paso)
            prev = y
        z -= paso
    return None


# ---------------------------------------------------------------- medidas
def medidas(ob, alto):
    co = vertices(ob).astype(np.float64)
    z = co[:, 2]
    zmin, zmax = z.min(), z.max()
    H = zmax - zmin
    out = {"alto_m": round(float(H), 4)}
    # hombros: ancho del modelo a 0.80 del alto (zona del deltoides, antes de que se ensanchen las mangas)
    for fz in (0.80,):
        k = np.abs(z - (zmin + fz * H)) < 0.01 * H
        out["ancho_hombros_m"] = round(float(co[k, 0].max() - co[k, 0].min()), 4)
    # cadera: altura de la entrepierna = primer z (de abajo arriba) donde la mitad trasera tiene vertices en el centro
    cen = (np.abs(co[:, 0]) < 0.015 * H) & (co[:, 1] > np.median(co[:, 1])) & (z < zmin + 0.62 * H) & (z > zmin + 0.25 * H)
    out["alto_cadera_m"] = round(float(z[cen].min() - zmin), 4) if cen.any() else None
    # brazos: eje principal (PCA) de los vertices laterales por debajo de los hombros; angulo con la vertical
    ang = {}
    pos = {}
    for lado, sg in (("derecha", 1), ("izquierda", -1)):
        k = (sg * co[:, 0] > 0.16 * H) & (z < zmin + 0.72 * H)
        P = co[k]
        c = P.mean(0)
        u, s, vt = np.linalg.svd(P - c, full_matrices=False)
        e = vt[0] * (1 if vt[0][2] < 0 else -1)          # apunta hacia abajo
        ang[lado] = round(math.degrees(math.acos(min(1, abs(e[2])))), 1)
        mano = co[sg * co[:, 0] > 0.88 * np.abs(co[:, 0]).max()]
        pos[lado] = [round(float(v), 4) for v in mano.mean(0)]
    out["angulo_brazos_grados"] = ang
    out["posicion_manos_m"] = pos
    return out


def main():
    a = sys.argv[sys.argv.index("--") + 1:]

    def arg(n, d=None):
        return a[a.index(n) + 1] if n in a else d

    cuerpo, cabeza, salida = arg("--cuerpo"), arg("--cabeza"), arg("--salida")
    if not (cuerpo and cabeza and salida):
        sys.exit("uso: --cuerpo c.glb --cabeza h.glb --salida DIR [--nombre n] [--alto 1.75]")
    nombre = arg("--nombre", "principe_alto")
    alto = float(arg("--alto", "1.75"))
    oro = [float(x) for x in arg("--oro", "0.085,0.17,0.25,0.4").split(",")]            # h_min, h_max, s_min, v_min del ribete
    xmax, zmin_b, zmax_b = [float(x) for x in arg("--zona-borde", "0.25,-0.45,0.0").split(",")]
    margen_cabeza = float(arg("--margen-cabeza", "0.025"))      # en unidades de la cabeza (por debajo del ribete)
    solape = float(arg("--solape", "0.003"))                    # en unidades del cuerpo (por encima del ribete llevado al cuerpo)
    region_cuerpo = float(arg("--region-cuerpo", "0.16"))
    # corte frontal: sector (grados a cada lado del frente), transicion (grados), bajada (unidades de la cabeza bajo el ribete),
    # fundido lateral en x (unidades de la cabeza antes del borde de la huella)
    # y del vertice del sector: el eje de la capucha (centro del lazo del ribete) queda delante del cuello, asi que el sector se mide desde
    # `eje` unidades de la cabeza detras de el.
    cf_sector, cf_trans, cf_baja, cf_fx, cf_eje = [float(x) for x in arg("--corte-frente", "50,30,0.12,0.08,0.12").split(",")]
    estirado = float(arg("--estirado", "10"))                   # arista mayor / mediana por encima de la cual un triangulo se descarta
    os.makedirs(salida, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    ob_c = importar(cuerpo, "cuerpo")
    ob_h = importar(cabeza, "cabeza")
    co_h, uv_h, lt_h, vt_h, col_h = color_triangulos(ob_h)
    th, ts, tv = T.hsv(col_h)
    tc = co_h[vt_h].mean(1)

    # 1. borde de la capucha en la cabeza
    es_oro = (th > oro[0]) & (th < oro[1]) & (ts > oro[2]) & (tv > oro[3])
    reg = es_oro & (tc[:, 2] > zmin_b) & (tc[:, 2] < zmax_b) & (np.abs(tc[:, 0]) < xmax + 0.02)
    cf, rms_borde, n_borde = IG.ajustar_borde(tc[reg].tolist(), ORDEN)
    cf = np.array(cf)
    print("borde: triangulos", int(reg.sum()), "usados", n_borde, "rms (unid. cabeza)", round(rms_borde, 4), "coef", np.round(cf, 3))
    zb_tri = z_borde(cf, tc)
    baja_h = bajada_frente(tc, cf[0] + cf_eje, xmax, cf_sector, cf_trans, cf_baja, cf_fx)
    print("borde z en el frente (unid. cabeza)", round(float(z_borde(cf, np.array([[0.0, cf[0] - 0.2, 0.0]]))[0]), 3))
    mantener_ajuste = (tc[:, 2] > zb_tri - margen_cabeza) & (np.abs(tc[:, 0]) < xmax)     # para el ICP: sin el corte frontal
    mantener_h = (tc[:, 2] > zb_tri - baja_h - margen_cabeza) & (np.abs(tc[:, 0]) < xmax)
    pts_h = co_h[np.unique(vt_h[mantener_ajuste])]
    print("cabeza: triangulos conservados", int(mantener_h.sum()), "de", len(vt_h))

    # 2. ajuste de similitud cabeza -> cuerpo
    co_c = vertices(ob_c).astype(np.float64)
    zt_c = co_c[:, 2].max()
    Hc = zt_c - co_c[:, 2].min()
    dst = co_c[co_c[:, 2] > zt_c - region_cuerpo * Hc]
    rng = np.random.default_rng(1)
    src = pts_h[rng.choice(len(pts_h), min(30000, len(pts_h)), replace=False)].astype(np.float64)
    if len(dst) > 120000:
        dst = dst[rng.choice(len(dst), 120000, replace=False)]
    nariz_h = punta_nariz(pts_h, pts_h[:, 2].max() - 0.7 * (pts_h[:, 2].max() - pts_h[:, 2].min()))
    nariz_c = punta_nariz(dst, zt_c - 0.7 * region_cuerpo * Hc)
    menton_h = menton_por_perfil(pts_h, nariz_h)
    mejor = None
    alto_h = pts_h[:, 2].max() - pts_h[:, 2].min()
    for f in (0.9, 1.1, 1.25, 1.4, 1.6):
        s0 = f * (region_cuerpo * Hc) / alto_h
        # traslacion inicial: coronilla con coronilla, centros en x/y de las cajas
        t0 = np.array([(dst[:, 0].max() + dst[:, 0].min()) / 2 - s0 * (src[:, 0].max() + src[:, 0].min()) / 2,
                       (dst[:, 1].max() + dst[:, 1].min()) / 2 - s0 * (src[:, 1].max() + src[:, 1].min()) / 2,
                       dst[:, 2].max() - s0 * src[:, 2].max()])
        s, R, t, d = icp(src, dst, s0, np.eye(3), t0)
        e = float(np.sqrt((d[d <= np.quantile(d, 0.7)] ** 2).mean()))
        print(f"  ICP inicio x{f}: escala {s:.4f} error rms (70%) {e * 1000:.2f} mm-cuerpo")
        if mejor is None or e < mejor[0]:
            mejor = (e, s, R, t, d)
    e, s, R, t, d = mejor
    M = Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            M[i][j] = s * R[i, j]
        M[i][3] = t[i]
    # 3. corte del cuerpo (en coordenadas de la cabeza)
    co_c_w, _, lt_c, vt_c = T._malla(ob_c)
    tcc = co_c_w[vt_c].mean(1)
    en_h = ((tcc - t) @ R) / s                       # inversa de la similitud (IG.invertir_similitud)
    zb_c = z_borde(cf, en_h)
    baja_c = bajada_frente(en_h, cf[0] + cf_eje, xmax, cf_sector, cf_trans, cf_baja, cf_fx)
    quitar_c = (np.abs(en_h[:, 0]) < xmax) & (en_h[:, 2] > zb_c - baja_c + solape / s)
    nq = borrar_triangulos(ob_c, quitar_c)
    print("cuerpo: triangulos quitados", nq)
    # recorte de la cabeza
    nh = borrar_triangulos(ob_h, ~mantener_h)
    def zona_h(c):          # franja del cuello y la cabeza (unidades de la cabeza): hasta 0.3 bajo el ribete
        return (c[:, 2] > z_borde(cf, c) - 0.3) & (np.abs(c[:, 0]) < xmax + 0.02)
    m_e = tris_estirados(ob_h, estirado, zona_h)
    nh += borrar_triangulos(ob_h, m_e) if m_e.any() else 0
    m_ec = tris_estirados(ob_c, estirado, lambda c: zona_h(((c - t) @ R) / s))
    nq += borrar_triangulos(ob_c, m_ec) if m_ec.any() else 0
    print("triangulos estirados quitados: cabeza", int(m_e.sum()), "cuerpo", int(m_ec.sum()))
    m_h, isl_h = islas_pequenas(ob_h)
    nh += borrar_triangulos(ob_h, m_h) if m_h.any() else 0
    m_c, isl_c = islas_pequenas(ob_c)
    nq += borrar_triangulos(ob_c, m_c) if m_c.any() else 0
    print("islas borradas cabeza", isl_h[:12], "cuerpo", isl_c[:12])
    aplicar_matriz(ob_h, M)

    # 4. unir, escalar, colocar
    ob_c.data.materials[0].name = "mat_cuerpo"
    ob_h.data.materials[0].name = "mat_cabeza"
    bpy.ops.object.select_all(action="DESELECT")
    ob_c.select_set(True)
    ob_h.select_set(True)
    bpy.context.view_layer.objects.active = ob_c
    bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    ob.name = nombre
    ob.data.name = nombre
    co = vertices(ob)
    zmin, zmax = float(co[:, 2].min()), float(co[:, 2].max())
    k = alto / (zmax - zmin)
    cx, cy = float((co[:, 0].max() + co[:, 0].min()) / 2), float((co[:, 1].max() + co[:, 1].min()) / 2)
    Mf = Matrix.Scale(k, 4) @ Matrix.Translation((-cx, -cy, -zmin))
    aplicar_matriz(ob, Mf)
    med = medidas(ob, alto)
    top_h = float(pts_h[:, 2].max())
    sk = float(s) * k
    med["alto_cabeza_coronilla_nariz_m"] = round((top_h - nariz_h) * sk, 4)
    med["alto_cabeza_coronilla_menton_m"] = round((top_h - menton_h) * sk, 4) if menton_h is not None else None
    ref = (zt_c - nariz_c) * k          # el cuerpo conserva su forma: unidad original -> k metros
    med["cuerpo_original_coronilla_nariz_m"] = round(ref, 4)
    med["cabeza_vs_cuerpo_original_pct"] = round(((top_h - nariz_h) * sk / ref - 1) * 100, 2)
    ajuste = {
        "escala_cabeza": round(float(s) * k, 5), "giro_cabeza_grados_xyz": IG.euler_xyz(R.tolist()),
        "error_icp_rms_mm": round(float(np.sqrt((d[d <= np.quantile(d, 0.7)] ** 2).mean())) * k * 1000, 2),
        "error_icp_mediana_mm": round(float(np.median(d)) * k * 1000, 2),
        "borde_rms_mm": round(rms_borde * float(s) * k * 1000, 2),
        "tris_cuerpo_quitados": nq, "tris_cabeza_quitados": nh,
    }
    ajuste["medidas"] = med
    with open(os.path.join(salida, nombre + "_ajuste.json"), "w") as f:
        json.dump(ajuste, f, indent=1)
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(salida, nombre + ".blend"))
    bpy.ops.export_scene.gltf(filepath=os.path.join(salida, nombre + ".glb"), export_format="GLB")
    print("INJERTO", json.dumps(ajuste))


if __name__ == "__main__":
    main()
