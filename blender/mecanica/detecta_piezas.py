"""Detección automática de piezas mecánicas en una malla fusionada (sin coordenadas fijas).
Unidad de corte: 'chart' = componente conexo de la malla SIN soldar (islas UV tras importar el glb).
Convención: arriba = +Z (glTF importado en Blender). Todo en unidades de la malla; los umbrales son
relativos a L = mayor dimensión de la caja envolvente.
Solo numpy + bpy. Funciones principales: cargar, orienta, detecta_ruedas, detecta_brazo,
detecta_cuerdas, detecta_rotor."""
import bpy, numpy as np, math

# ------------------------------------------------------------------ carga / charts
def cargar(path):
    """Importa el glb, une las mallas, aplica transformaciones. Devuelve (objeto, co, tri)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)
    ms = [o for o in bpy.data.objects if o.type == 'MESH']
    bpy.ops.object.select_all(action='DESELECT')
    for o in ms: o.select_set(True)
    bpy.context.view_layer.objects.active = ms[0]
    if len(ms) > 1: bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    ob.parent = None
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    me = ob.data
    n = len(me.vertices)
    co = np.empty(n * 3); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    tri = np.empty(len(me.loops), dtype=np.int64); me.loops.foreach_get('vertex_index', tri)
    return ob, co, tri.reshape(-1, 3)

def charts_de(n, tri):
    """Etiqueta por vértice del componente conexo (malla sin soldar)."""
    par = np.arange(n)
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    for u, v in np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]]]):
        ru, rv = find(u), find(v)
        if ru != rv: par[ru] = rv
    root = np.array([find(i) for i in range(n)])
    _, lab = np.unique(root, return_inverse=True)
    return lab

class Malla:
    """Contenedor con descriptores por chart. L = mayor dimensión; todo relativo a L."""
    def __init__(self, co, tri):
        self.co, self.tri = co, tri
        self.fc = co[tri].mean(1)
        self.vlab = charts_de(len(co), tri)
        self.flab = self.vlab[tri[:, 0]]
        self.nch = self.vlab.max() + 1
        self.mn, self.mx = co.min(0), co.max(0)
        self.L = float((self.mx - self.mn).max())
        self.nf = np.bincount(self.flab, minlength=self.nch)
        self.desc = [None] * self.nch
        order = np.argsort(self.vlab, kind='stable')
        bounds = np.searchsorted(self.vlab[order], np.arange(self.nch + 1))
        self.vidx = [order[bounds[i]:bounds[i + 1]] for i in range(self.nch)]
        self.faces_of = [np.where(self.flab == c)[0] for c in range(self.nch)] if self.nch < 20000 else None

    def pca(self, c):
        """(centro, eigvals_asc, eigvecs columnas asc, extensiones PCA ~ rango) del chart c."""
        if self.desc[c] is None:
            q = self.co[self.vidx[c]]
            cen = q.mean(0)
            if len(q) < 4:
                w, v = np.ones(3) * 1e-12, np.eye(3)
            else:
                w, v = np.linalg.eigh(np.cov((q - cen).T))
            ext = np.sqrt(np.maximum(w, 0)) * 3.46
            self.desc[c] = (cen, w, v, ext)
        return self.desc[c]

# ------------------------------------------------------------------ orientación
def orienta(M, ignorar=None):
    """arriba=+Z. Eje largo horizontal por PCA de la huella (vértices); si está a <12 grados de X o Y se alinea.
    Devuelve dict(up, fwd, lat, fwd_pca). El sentido de fwd se fija luego (la plantilla lo decide)."""
    up = np.array([0, 0, 1.0])
    xy = M.co[:, :2]
    c = xy - xy.mean(0)
    w, v = np.linalg.eigh(np.cov(c.T))
    f2 = v[:, 1]
    ang = math.degrees(math.atan2(f2[1], f2[0])) % 180
    snap = None
    for a in (0, 90, 180):
        if abs(ang - a) < 12: snap = a % 180
    if snap is not None:
        ang = snap
        f2 = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
    fwd = np.array([f2[0], f2[1], 0.0]); fwd /= np.linalg.norm(fwd)
    if fwd[0] + 1e-9 < 0 or (abs(fwd[0]) < 1e-9 and fwd[1] < 0): fwd = -fwd
    lat = np.cross(up, fwd)  # = u x f -> giro positivo: lo de arriba avanza
    return dict(up=up, fwd=fwd, lat=lat, snapped=snap is not None, ang=ang)

# ------------------------------------------------------------------ ruedas
def detecta_ruedas(M, min_caras=40, verbose=True):
    """Ruedas = charts discoidales (planos/anillos circulares, eje horizontal) cerca del suelo.
    Agrupa por eje (centro en el plano perpendicular) y radio; luego asigna por mayoría de caras
    los charts que caen dentro del cilindro de cada rueda.
    Devuelve lista de dicts: centro(3), eje(3, unit), radio, a0,a1 (rango axial), faces(array), charts."""
    L = M.L; zmin = M.mn[2]
    cand = []
    for c in range(M.nch):
        if M.nf[c] < min_caras: continue
        cen, w, v, ext = M.pca(c)
        s = np.sqrt(np.maximum(w, 1e-18))
        n = v[:, 0]
        if s[0] / s[2] > 0.22 or s[1] / s[2] < 0.55: continue     # plano y circular
        if abs(n[2]) > 0.35: continue                              # eje horizontal
        q = M.co[M.vidx[c]]
        d1, d2 = v[:, 1], v[:, 2]
        pr = np.stack([(q - cen) @ d1, (q - cen) @ d2], 1)
        r = float(np.percentile(np.linalg.norm(pr, axis=1), 99))
        if not (0.03 * L < r < 0.3 * L): continue
        if cen[2] - r > zmin + 0.12 * L: continue                  # cerca del suelo
        ax = n if abs(n[np.argmax(np.abs(n))]) > 0 else n
        cand.append(dict(c=c, cen=cen, n=n, r=r, nf=M.nf[c]))
    if verbose: print("ruedas: charts candidatos discoidales:", len(cand))
    if not cand: return []
    # eje dominante (media ponderada con signo alineado)
    ref = max(cand, key=lambda d: d['nf'])['n']
    W = np.zeros(3)
    for d in cand:
        W += d['nf'] * (d['n'] if d['n'] @ ref >= 0 else -d['n'])
    axis = W / np.linalg.norm(W)
    cand = [d for d in cand if abs(d['n'] @ axis) > 0.9]
    # radio de referencia: mediana ponderada por caras (las ruedas son lo más grande y discoidal)
    rs = np.array([d['r'] for d in cand]); wt_ = np.array([d['nf'] for d in cand], float)
    o = np.argsort(rs); cw = np.cumsum(wt_[o]); r_med = rs[o][np.searchsorted(cw, cw[-1] / 2)]
    cand = [d for d in cand if abs(d['r'] - r_med) < 0.3 * r_med]
    # base ortonormal en el plano perpendicular al eje
    e1 = np.cross(axis, [0, 0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(axis, e1)
    for d in cand:
        d['p'] = np.array([d['cen'] @ e1, d['cen'] @ e2]); d['a'] = float(d['cen'] @ axis)
    # clustering voraz por centro 2D y radio
    groups = []
    for d in sorted(cand, key=lambda d: -d['nf']):
        for g in groups:
            if np.linalg.norm(d['p'] - g['p']) < 0.6 * g['r'] and abs(d['r'] - g['r']) < 0.3 * g['r']:
                g['m'].append(d)
                wt_g = np.array([x['nf'] for x in g['m']], float)
                g['p'] = (np.array([x['p'] for x in g['m']]) * wt_g[:, None]).sum(0) / wt_g.sum()
                break
        else:
            groups.append(dict(p=d['p'].copy(), r=d['r'], m=[d]))
    # separa cada grupo por lado (coordenada axial) -> una rueda por lado
    amid = float(M.co @ axis @ np.ones(1)) if False else float((M.co @ axis).min() + (M.co @ axis).max()) / 2
    wheels = []
    for g in groups:
        for side in (-1, 1):
            m = [d for d in g['m'] if (d['a'] - amid) * side > 0]
            if not m: continue
            wt = np.array([d['nf'] for d in m], float)
            p = (np.array([d['p'] for d in m]) * wt[:, None]).sum(0) / wt.sum()
            r = max(d['r'] for d in m)
            a_lo = min(M.co[M.vidx[d['c']]] @ axis for d in m) if False else min((M.co[M.vidx[d['c']]] @ axis).min() for d in m)
            a_hi = max((M.co[M.vidx[d['c']]] @ axis).max() for d in m)
            wheels.append(dict(p=p, r=r, a0=a_lo - 0.012 * L, a1=a_hi + 0.012 * L, side=side, seeds=[d['c'] for d in m]))
    # asigna caras por mayoría de chart dentro del cilindro
    fa = M.fc @ axis
    fp = np.stack([M.fc @ e1, M.fc @ e2], 1)
    taken = np.zeros(len(M.fc), bool)
    out = []
    for w in sorted(wheels, key=lambda w: (w['side'], w['p'][0])):
        inside = (np.linalg.norm(fp - w['p'], axis=1) < w['r'] * 1.08) & (fa > w['a0']) & (fa < w['a1']) & (~taken)
        sel = np.zeros(len(M.fc), bool)
        cnt_in = np.bincount(M.flab[inside], minlength=M.nch)
        ok = cnt_in > 0.8 * M.nf
        sel = ok[M.flab] & (~taken)
        if sel.sum() < 100: continue
        taken |= sel
        cen = w['p'][0] * e1 + w['p'][1] * e2 + 0.5 * (w['a0'] + w['a1']) * axis
        out.append(dict(centro=cen, eje=axis, radio=w['r'], a0=w['a0'], a1=w['a1'], side=w['side'],
                        faces=np.where(sel)[0], e1=e1, e2=e2))
    # centra el eje usando bbox de caras asignadas (más estable que la media de semillas)
    for w in out:
        q = M.co[np.unique(M.tri[w['faces']])]
        p1, p2, pa = q @ w['e1'], q @ w['e2'], q @ w['eje']
        cc = ((p1.min() + p1.max()) / 2, (p2.min() + p2.max()) / 2, (pa.min() + pa.max()) / 2)
        w['centro'] = cc[0] * w['e1'] + cc[1] * w['e2'] + cc[2] * w['eje']
        w['radio'] = float(((p1.max() - p1.min()) + (p2.max() - p2.min())) / 4)
    if verbose:
        for w in out: print("  rueda centro", w['centro'].round(3), "r %.3f" % w['radio'], "lado", w['side'], "caras", len(w['faces']))
    return out

# ------------------------------------------------------------------ brazo
def mascara_cuerdas(M, e1_max=0.03):
    """Charts de cuerda/cable: delgados y de sección redonda (las dos dimensiones menores PCA < e1_max*L)."""
    m = np.zeros(M.nch, bool)
    for c in range(M.nch):
        if M.nf[c] < 12: continue
        _, _, _, ext = M.pca(c)
        if ext[1] < e1_max * M.L and ext[0] > 0.35 * ext[1] and ext[2] > 0.03 * M.L and abs(M.pca(c)[2][2, 2]) > 0.25: m[c] = True
    return m

def _kd(pts):
    from mathutils import kdtree
    k = kdtree.KDTree(len(pts))
    for i, p in enumerate(pts): k.insert(p, i)
    k.balance(); return k

def detecta_brazo(M, excluir_faces, ori=None, verbose=True):
    """Brazo/palanca: chart alargado (>0.3L), inclinado <55 grados, en la mitad alta. Se crece por caja orientada
    (semiancho lateral = el de la semilla + margen) absorbiendo charts cuya mayoría de caras cae dentro, y se
    extiende en la punta (cabeza). Pivote = zona del brazo en contacto con el marco, en su tercio trasero
    (kd-tree); si no hay contacto, el extremo trasero. Devuelve dict o None."""
    L = M.L
    cnt_ex = np.bincount(M.flab[excluir_faces], minlength=M.nch) if len(excluir_faces) else np.zeros(M.nch)
    excl = cnt_ex > 0.3 * M.nf
    cuerda = mascara_cuerdas(M)
    best = None
    for c in range(M.nch):
        if excl[c] or cuerda[c] or M.nf[c] < 60: continue
        cen, w, v, ext = M.pca(c)
        d = v[:, 2]
        if ext[2] < 0.3 * L or ext[2] < 2.5 * ext[1]: continue
        if abs(d[2]) > 0.8: continue
        if cen[2] < M.mn[2] + 0.5 * L: continue
        if best is None or ext[2] > best[1]: best = (c, ext[2], d, cen)
    if best is None:
        if verbose: print("brazo: no hay chart alargado/alto")
        return None
    c0, length, d, cen0 = best
    if verbose: print("brazo: semilla chart %d long %.3f centro %s dir %s" % (c0, length, cen0.round(3), d.round(3)))
    t = np.cross([0, 0, 1.0], d); t /= np.linalg.norm(t)
    w3 = np.cross(d, t)
    R = np.stack([d, t, w3], 1)
    fcl = (M.fc - cen0) @ R
    vl = (M.co - cen0) @ R
    sel_c = np.zeros(M.nch, bool); sel_c[c0] = True
    q0 = vl[M.vidx[c0]]
    tc = ((M.co @ t).min() + (M.co @ t).max()) * 0.5 - cen0 @ t           # plano medio lateral del objeto
    hw = np.abs(q0[:, 1] - tc).max() + 0.04 * L
    lo = np.array([q0[:, 0].min() - 0.04 * L, tc - hw, q0[:, 2].min() - 0.04 * L])
    hi = np.array([q0[:, 0].max() + 0.06 * L, tc + hw, q0[:, 2].max() + 0.04 * L])
    def absorbe(lo, hi):
        inb = ((fcl > lo) & (fcl < hi)).all(1)
        cin = np.bincount(M.flab[inb], minlength=M.nch)
        return (cin > 0.6 * M.nf) & (~excl)
    sel_c |= absorbe(lo, hi)
    # punta (cabeza): dos pasadas de extensión hacia el extremo, algo más anchas
    for _ in range(2):
        q = vl[np.isin(M.vlab, np.where(sel_c)[0])]
        s_top = q[:, 0].max(); s_bot = q[:, 0].min()
        # el extremo 'punta' es el más alejado del centro de masa del resto del objeto
        rest = M.co[~sel_c[M.vlab]].mean(0)
        sr = (rest - cen0) @ R
        tip_pos = abs(s_top - sr[0]) > abs(s_bot - sr[0])
        if tip_pos: lo2 = np.array([s_top - 0.06 * L, tc - hw - 0.03 * L, q[:, 2].min() - 0.05 * L]); hi2 = np.array([s_top + 0.08 * L, tc + hw + 0.03 * L, q[:, 2].max() + 0.05 * L])
        else: lo2 = np.array([s_bot - 0.08 * L, tc - hw - 0.03 * L, q[:, 2].min() - 0.05 * L]); hi2 = np.array([s_bot + 0.06 * L, tc + hw + 0.03 * L, q[:, 2].max() + 0.05 * L])
        sel_c |= absorbe(lo2, hi2)
    faces = np.where(sel_c[M.flab])[0]
    vv = np.unique(M.tri[faces])
    q = vl[vv]
    if verbose: print("brazo: %d charts, %d caras, s=[%.3f,%.3f], lat=[%.3f,%.3f], w=[%.3f,%.3f]" % (sel_c.sum(), len(faces), q[:, 0].min(), q[:, 0].max(), q[:, 1].min(), q[:, 1].max(), q[:, 2].min(), q[:, 2].max()))
    # --- orientación punta/pivote
    rest = M.co[~sel_c[M.vlab]].mean(0)
    sr = ((rest - cen0) @ R)[0]
    s_lo, s_hi = np.percentile(q[:, 0], 0.5), np.percentile(q[:, 0], 99.5)
    tip_pos = abs(s_hi - sr) > abs(s_lo - sr)
    sgn = 1.0 if tip_pos else -1.0           # s*sgn crece hacia la punta
    ss = q[:, 0] * sgn
    s_rear, s_front = ss.min(), ss.max()
    # --- pivote por contacto con el marco
    arm_set = set(vv.tolist())
    fvert = np.unique(M.tri[~sel_c[M.flab] & ~excl[M.flab] & ~cuerda[M.flab]])
    kd = _kd(M.co[fvert])
    near = np.array([kd.find(tuple(M.co[v]))[2] < 0.012 * L for v in vv])
    rear_zone = ss < s_rear + 0.4 * (s_front - s_rear)
    cp = near & rear_zone
    if verbose: print("brazo: vértices en contacto con el marco %d (zona trasera %d)" % (near.sum(), cp.sum()))
    if cp.sum() >= 15:
        sp = np.median(ss[cp]); wp = np.median(q[cp, 2]); how = "contacto"
    else:
        sp = s_rear + 0.04 * L; wp = q[:, 2].mean(); how = "extremo trasero"
    # eje del brazo (línea punta-pivote en el plano s-w) y pivote 3D
    piv_l = np.array([sp * sgn, tc, wp])
    pivote = cen0 + R @ piv_l
    tip_sel = ss > s_front - 0.05 * L
    tip_l = np.array([q[tip_sel, 0].mean(), tc, q[tip_sel, 2].mean()])
    punta = cen0 + R @ tip_l
    if verbose: print("brazo: pivote (%s) %s  punta %s" % (how, pivote.round(3), punta.round(3)))
    return dict(faces=faces, charts=np.where(sel_c)[0], seed=c0, pivote=pivote, punta=punta, how=how,
                length=float(np.linalg.norm(punta - pivote)), arm_verts=vv)

def detecta_cuerdas(M, brazo, excluir_faces, verbose=True):
    """Cuerdas atadas al brazo: charts 'cable' (delgados y redondos) conectados por proximidad a la
    región cercana al brazo. Agrupa por conectividad entre charts lejanos al brazo; cada cuerda recibe
    extremo-marco y extremo-brazo para pesos lineales. Devuelve lista de dicts(faces, frame_end, arm_end)."""
    L = M.L
    cuerda = mascara_cuerdas(M)
    sel = np.zeros(M.nch, bool); sel[brazo['charts']] = True
    cnt_ex = np.bincount(M.flab[excluir_faces], minlength=M.nch) if len(excluir_faces) else np.zeros(M.nch)
    cuerda &= ~sel & ~(cnt_ex > 0.3 * M.nf)
    cs = np.where(cuerda)[0]
    if len(cs) == 0: return []
    kd_arm = _kd(M.co[brazo['arm_verts']])
    near_r = 0.045 * L
    near = {}
    for c in cs:
        near[c] = min(kd_arm.find(tuple(M.co[v]))[2] for v in M.vidx[c]) < near_r
    # kd de vértices de cuerda
    allv = np.concatenate([M.vidx[c] for c in cs]); owner = np.concatenate([[c] * len(M.vidx[c]) for c in cs])
    kd = _kd(M.co[allv])
    par = {c: c for c in cs}
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    thr = 0.012 * L
    attach = {}
    for i, v in enumerate(allv):
        c = owner[i]
        for _, j, _ in kd.find_range(tuple(M.co[v]), thr):
            c2 = owner[j]
            if c2 == c: continue
            if not near[c] and not near[c2]:
                a, b = find(c), find(c2)
                if a != b: par[a] = b
            elif near[c] and not near[c2]:
                attach.setdefault(c, {}).setdefault(c2, 0); attach[c][c2] += 1
    clusters = {}
    for c in cs:
        if not near[c]: clusters.setdefault(find(c), []).append(c)
    anchored = {}
    for c, d in attach.items():
        c2 = max(d, key=d.get); anchored.setdefault(find(c2), []).append(c)
    ropes = []
    used_global = set()
    for root, members in clusters.items():
        if root not in anchored: continue
        mem = members + anchored[root]
        vids = np.unique(np.concatenate([M.vidx[c] for c in mem]))
        pts = M.co[vids]
        if len(pts) < 100: continue
        cen = pts.mean(0); w, v = np.linalg.eigh(np.cov((pts - cen).T)); ax = v[:, 2]
        pr = (pts - cen) @ ax
        e_a = pts[pr < pr.min() + 0.02 * L].mean(0); e_b = pts[pr > pr.max() - 0.02 * L].mean(0)
        da = kd_arm.find(tuple(e_a))[2]; db = kd_arm.find(tuple(e_b))[2]
        arm_end, frame_end = (e_a, e_b) if da < db else (e_b, e_a)
        if np.linalg.norm(arm_end - frame_end) < 0.12 * L: continue
        # restos: charts pequeños y delgados pegados a la cuerda que la máscara estricta dejó fuera
        for _ in range(2):
            kdr = _kd(M.co[np.unique(np.concatenate([M.vidx[c] for c in mem]))])
            usados = set(mem) | set(brazo['charts'].tolist())
            for c in range(M.nch):
                if c in usados or c in used_global or M.nf[c] > 120 or M.nf[c] < 4 or (cnt_ex[c] > 0.3 * M.nf[c]): continue
                if M.pca(c)[3][1] > 0.04 * L: continue
                if any(kdr.find(tuple(M.co[v]))[2] < 0.006 * L for v in M.vidx[c][::2]): mem.append(c)
        used_global.update(mem)
        fm = np.isin(M.flab, mem)
        ropes.append(dict(charts=mem, faces=np.where(fm)[0], arm_end=arm_end, frame_end=frame_end))
        if verbose: print("cuerda: %d charts %d caras extremo-marco %s extremo-brazo %s" % (len(mem), fm.sum(), frame_end.round(3), arm_end.round(3)))
    return ropes

# ------------------------------------------------------------------ rotor
def detecta_rotor(M, verbose=True):
    """Rotor (aspas): >=3 charts alargados cuyos ejes largos convergen en un punto (cubo) y son coplanares
    (el eje de giro = normal a ese plano). Hub por RANSAC sobre pares de líneas. Devuelve dict(centro, eje,
    radio, faces) o None. Todas las piezas dentro del cilindro (radio y losa axial) se agrupan por mayoría."""
    L = M.L
    cand = []
    for c in range(M.nch):
        if M.nf[c] < 30: continue
        cen, w, v, ext = M.pca(c)
        if ext[2] < 0.12 * L or ext[2] < 2.0 * ext[1]: continue
        cand.append((c, cen, v[:, 2], ext))
    if len(cand) < 3:
        if verbose: print("rotor: pocos charts alargados", len(cand))
        return None
    cand.sort(key=lambda x: -x[3][2]); cand = cand[:60]
    def closest(p1, d1, p2, d2):
        n = np.cross(d1, d2)
        if np.linalg.norm(n) < 0.2: return None
        A = np.stack([d1, -d2, n], 1)
        try: ts = np.linalg.solve(A, p2 - p1)
        except Exception: return None
        return (p1 + ts[0] * d1 + p2 + ts[1] * d2) / 2
    best = None
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            h = closest(cand[i][1], cand[i][2], cand[j][1], cand[j][2])
            if h is None: continue
            inl = []
            for (c, cen, d, ext) in cand:
                r = cen - h; rn = np.linalg.norm(r)
                if rn < 0.03 * L: continue
                dist_line = np.linalg.norm(r - (r @ d) * d)
                if dist_line < 0.05 * L and abs(r @ d) / rn > 0.85: inl.append((c, cen, d, ext))
            if len(inl) >= 3:
                score = sum(x[3][2] for x in inl)
                if best is None or score > best[0]: best = (score, h, inl)
    if best is None:
        if verbose: print("rotor: no convergen ejes de aspas")
        return None
    _, hub, inl = best
    dirs = np.array([x[2] for x in inl])
    w, v = np.linalg.eigh(dirs.T @ dirs)
    axis = v[:, 0]
    if axis[np.argmax(np.abs(axis))] < 0: axis = -axis
    if abs(abs(axis) @ np.array([1, 0, 0])) > 0.995 or abs(abs(axis) @ np.array([0, 1, 0])) > 0.995 or abs(abs(axis) @ np.array([0, 0, 1])) > 0.995:
        axis = np.eye(3)[np.argmax(np.abs(axis))] * np.sign(axis[np.argmax(np.abs(axis))])
    vs = np.unique(np.concatenate([M.vidx[x[0]] for x in inl]))
    P = M.co[vs] - hub
    ax_c = P @ axis; rad = np.linalg.norm(P - np.outer(ax_c, axis), axis=1)
    R = float(rad.max()); a0, a1 = ax_c.min() - 0.02 * L, ax_c.max() + 0.02 * L
    fp = M.fc - hub; fa = fp @ axis; fr = np.linalg.norm(fp - np.outer(fa, axis), axis=1)
    inside = (fr < R * 1.05) & (fa > a0) & (fa < a1)
    cin = np.bincount(M.flab[inside], minlength=M.nch)
    ok = cin > 0.6 * M.nf
    faces = np.where(ok[M.flab])[0]
    if verbose: print("rotor: hub %s eje %s radio %.3f aspas %d caras %d" % (hub.round(3), axis.round(3), R, len(inl), len(faces)))
    return dict(centro=hub, eje=axis, radio=R, faces=faces, aspas=[x[0] for x in inl])
