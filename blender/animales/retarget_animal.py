"""retarget_animal.py -- da animaciones de cuadrupedos (Quaternius "Ultimate Animated Animals", CC0) a un cuadrupedo propio.

    blender -b -P retarget_animal.py -- <nuestro.glb> <fuente_quaternius.gltf|glb> <salida.glb>
        [--metodo a|b] [--adelante auto|+x|-x|+y|-y] [--save-blend f.blend] [--sin-ik]

Nuestro modelo: malla (con o sin rig; el rig se descarta en el metodo b) de un animal parado sobre 4 patas, con Z arriba.
La fuente es el glTF de Quaternius de un animal (Wolf.gltf, Horse.gltf, Fox.gltf...) con su armature y sus clips.
No usa nombres de hueso: todo se deduce de la geometria (patas por la suela de la malla, cabeza por el contorno).

METODO B (por defecto) -- reemplazo de esqueleto:
  1. mide ambas mallas (frente, 4 pies, lomo, hocico) y deforma la fuente (malla + huesos en reposo) con una funcion
     lineal por tramos por eje (adelante / lateral / vertical) para que cuerpo y patas coincidan con los de nuestra malla;
  2. transfiere los pesos de piel de la malla de Quaternius a la nuestra (Data Transfer, cara mas cercana interpolada),
     normaliza y limita a 4 influencias;
  3. se descarta el rig original y se usan el esqueleto de Quaternius y SUS clips (las traslaciones locales se reescalan).
METODO A (EXPERIMENTAL, no recomendado) -- retarget al rig que ya trae nuestro modelo (Puppeteer): los huesos se emparejan por posicion
  (columna/cola/4 patas, cadenas por fraccion de longitud) y el delta de rotacion en espacio mundo de cada hueso fuente se aplica al destino.
  En las pruebas con lobo y caballo el cuerpo queda mal orientado y las patas se descolocan: se dejo para comparar, el resultado util es el B.

Autoprueba sin modelos:  blender -b -P retarget_animal.py -- --autotest

Clips: se conservan los de la fuente; los ciclos llevan sufijo _Loop (Godot los importa en bucle).
"""
import json
import math
import os
import re
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

FPS = 30
# clips que NO son ciclos (el resto de clips de Quaternius si lo son)
NO_BUCLE = re.compile(r"(death|attack|hitreact|jump|kick|headbutt|fly_?to|land|toidle|stand|sit|lay|roll|spin|bounce)", re.I)
SI_BUCLE = re.compile(r"^(idle|idle_\d|idle_headlow|walk|gallop|run|trot|swim|fly|eating|flying|jog|canter)\w*$", re.I)


def log(*a):
    print("[animal]", *a, flush=True)


# --------------------------------------------------------------------------- utilidades de escena
def importar(path):
    antes = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=path)
    return [o.name for o in bpy.data.objects if o.name not in antes]


def quitar_formas_hueso():
    """El importador de glTF crea 'Icosphere' (forma de visualizacion de huesos). Fuera."""
    for o in bpy.data.objects:
        if o.type == "ARMATURE":
            for b in o.pose.bones:
                b.custom_shape = None
    for o in list(bpy.data.objects):
        if o.type == "MESH" and ("Icosphere" in o.name or "Icosphere" in o.data.name):
            bpy.data.objects.remove(o, do_unlink=True)


def puntos(meshes):
    """(N,3) vertices en mundo."""
    out = []
    for m in meshes:
        n = len(m.data.vertices)
        co = np.zeros(n * 3)
        m.data.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        M = np.array(m.matrix_world)
        out.append(co @ M[:3, :3].T + M[:3, 3])
    return np.concatenate(out)


def escribir_puntos(m, P):
    """P en mundo -> co local de la malla."""
    Mi = np.array(m.matrix_world.inverted())
    co = P @ Mi[:3, :3].T + Mi[:3, 3]
    m.data.vertices.foreach_set("co", co.ravel())
    m.data.update()


def hijos_malla(arm):
    return [o for o in bpy.data.objects if o.type == "MESH" and (o.parent == arm or any(md.type == "ARMATURE" and md.object == arm for md in o.modifiers))]


# --------------------------------------------------------------------------- medidas del animal
def eje_adelante(P, forzado="auto"):
    """Vector horizontal unitario hacia donde mira el animal (+-X o +-Y)."""
    m = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}
    if forzado in m:
        return Vector((m[forzado][0], m[forzado][1], 0))
    ext = P.max(0) - P.min(0)
    ax = 0 if ext[0] >= ext[1] else 1
    c = P[:, ax]
    lo, hi = c.min(), c.max()
    L = hi - lo
    zt = {}
    for sg, sel in (("+", c > hi - 0.18 * L), ("-", c < lo + 0.18 * L)):
        zt[sg] = float(np.percentile(P[sel, 2], 98)) if sel.any() else -1e9
    # la cabeza (cuello + orejas) queda mas alta que el nacimiento de la cola
    s = 1 if zt["+"] >= zt["-"] else -1
    v = Vector((0, 0, 0))
    v[ax] = s
    log(f"adelante auto: eje {'XY'[ax]} signo {s:+d} (z extremo +={zt['+']:.3f}  -={zt['-']:.3f})")
    return v


class Medidas:
    """Medidas del animal en el marco (f=adelante, l=lateral, z=arriba)."""

    def __init__(self, P, u, nombre=""):
        self.u = Vector(u)
        self.lat = Vector((0, 0, 1)).cross(self.u)
        self.P = P
        f, l, z = self.coords(P)
        self.zmin, self.zmax = float(z.min()), float(z.max())
        H = self.zmax - self.zmin
        L = float(f.max() - f.min())
        self.lc = float((l.min() + l.max()) / 2)
        self.ancho = float(l.max() - l.min())
        self.nose_f = float(f.max())
        self.f_min = float(f.min())
        # pies: suela de la malla (7 % inferior de la altura), separada por lado y por frente/atras
        sel = z < self.zmin + 0.07 * H
        feet = {}
        for s in (1, -1):
            idx = sel & ((l - self.lc) * s > 0.08 * (l.max() - l.min()))
            if idx.sum() < 4:
                continue
            ff = np.sort(f[idx])
            gaps = np.diff(ff)
            k = int(np.argmax(gaps))
            if gaps[k] < 0.10 * L:
                continue
            cut = (ff[k] + ff[k + 1]) / 2
            for nom, mk in (("hind", idx & (f < cut)), ("fore", idx & (f >= cut))):
                feet[(nom, s)] = (float(f[mk].mean()), float(l[mk].mean()))
        self.feet = feet
        self.ok = all((n, s) in feet for n in ("hind", "fore") for s in (1, -1))
        if self.ok:
            self.hind_f = (feet[("hind", 1)][0] + feet[("hind", -1)][0]) / 2
            self.fore_f = (feet[("fore", 1)][0] + feet[("fore", -1)][0]) / 2
            self.half_l = np.mean([abs(feet[(n, s)][1] - self.lc) for n in ("hind", "fore") for s in (1, -1)])
        else:  # respaldo por bbox
            self.hind_f = float(f.min() + 0.25 * L)
            self.fore_f = float(f.min() + 0.70 * L)
            self.half_l = float((l.max() - l.min()) * 0.3)
            log(f"AVISO ({nombre}): no se detectaron los 4 pies; se usan proporciones de la bbox")
        # altura del lomo: mediana del maximo en 8 tramos entre pies traseros y delanteros
        edges = np.linspace(self.hind_f, self.fore_f, 9)
        tops = []
        for a, b in zip(edges[:-1], edges[1:]):
            mk = (f >= a) & (f < b)
            if mk.any():
                tops.append(z[mk].max())
        self.back_z = float(np.median(tops)) if tops else self.zmax * 0.8
        log(f"{nombre}: ground={self.zmin:.3f} lomo={self.back_z:.3f} top={self.zmax:.3f} pies f=({self.hind_f:.3f},{self.fore_f:.3f}) "
            f"medio-ancho patas={self.half_l:.3f} hocico f={self.nose_f:.3f} L={L:.3f} {'ok' if self.ok else 'RESPALDO'}")

    def coords(self, P):
        return P @ np.array(self.u), P @ np.array(self.lat), P[:, 2]

    def desde(self, f, l, z):
        return np.outer(f, np.array(self.u)) + np.outer(l, np.array(self.lat)) + np.outer(z, [0, 0, 1])


def pwl(x, xs, ys):
    """Lineal por tramos con extrapolacion lineal. Devuelve (y, pendiente)."""
    x = np.asarray(x, float)
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    sl = np.diff(ys) / np.diff(xs)
    i = np.clip(np.searchsorted(xs, x) - 1, 0, len(sl) - 1)
    return ys[i] + (x - xs[i]) * sl[i], sl[i]


class Warp:
    """Funcion fuente -> destino (separable por ejes, lineal por tramos)."""

    def __init__(self, S, T):
        self.S, self.T = S, T
        bs = (S.fore_f - S.hind_f)
        bt = (T.fore_f - T.hind_f)
        sb = bt / bs
        head_s = (T.nose_f - T.fore_f) / max(S.nose_f - S.fore_f, 1e-6)
        head_s = min(max(head_s, 0.5 * sb), 2.0 * sb)
        nose_t = T.fore_f + head_s * (S.nose_f - S.fore_f)
        self.fx, self.fy = [S.hind_f, S.fore_f, S.nose_f], [T.hind_f, T.fore_f, nose_t]
        dz = S.back_z - S.zmin
        self.zx, self.zy = [S.zmin, S.back_z], [T.zmin, T.zmin + (T.back_z - T.zmin)]
        self.sl = T.half_l / S.half_l
        log(f"warp: cuerpo f x{sb:.3f}  cabeza x{head_s:.3f}  lateral x{self.sl:.3f}  vertical x{(T.back_z - T.zmin) / dz:.3f}")

    def __call__(self, P):
        """P (N,3) mundo fuente -> (P' mundo destino, J (N,3,3) jacobiano mundo)."""
        S, T = self.S, self.T
        f, l, z = S.coords(P)
        f2, sf = pwl(f, self.fx, self.fy)
        z2, sz = pwl(z, self.zx, self.zy)
        l2 = T.lc + (l - S.lc) * self.sl
        P2 = T.desde(f2, l2, z2)
        # jacobiano: B_T diag(s) B_S^T
        BS = np.array([list(S.u), list(S.lat), [0, 0, 1]]).T
        BT = np.array([list(T.u), list(T.lat), [0, 0, 1]]).T
        D = np.zeros((len(P), 3, 3))
        D[:, 0, 0], D[:, 1, 1], D[:, 2, 2] = sf, self.sl, sz
        J = BT[None] @ D @ BS.T[None]
        return P2, J


# --------------------------------------------------------------------------- acciones (API de Blender 5: capas/tiras/canales)
def fcurves(act):
    for layer in act.layers:
        for st in layer.strips:
            for cb in st.channelbags:
                yield from cb.fcurves


def reescalar_traslaciones(acts, M):
    """M: {hueso: matriz 3x3 numpy} que se aplica a (x,y,z) de pose.bones[h].location en cada clave."""
    pat = re.compile(r'pose\.bones\["(.+)"\]\.location')
    for act in acts:
        por = {}
        for fc in fcurves(act):
            mt = pat.fullmatch(fc.data_path)
            if mt and mt.group(1) in M:
                por.setdefault(mt.group(1), {})[fc.array_index] = fc
        for h, d in por.items():
            if len(d) != 3:
                continue
            n = len(d[0].keyframe_points)
            V = np.zeros((n, 3))
            for c in range(3):
                co = np.zeros(n * 2)
                d[c].keyframe_points.foreach_get("co", co)
                V[:, c] = co[1::2]
            W = V @ M[h].T
            for c in range(3):
                kp = d[c].keyframe_points
                for i in range(n):
                    dv = W[i, c] - kp[i].co[1]
                    kp[i].co[1] = W[i, c]
                    kp[i].handle_left[1] += dv
                    kp[i].handle_right[1] += dv
                d[c].update()


def nombre_clip(n):
    n = re.sub(r"^(AnimalArmature|Armature)\|", "", n)
    n = n.replace(" ", "_")
    if n.endswith("_Loop"):
        return n
    if NO_BUCLE.search(n):
        return n
    if SI_BUCLE.match(n):
        return n + "_Loop"
    return n


def limpiar_acciones(arm, conservar=None):
    """Acciones con usuario falso y nombre final; las que no sean de `arm` se borran."""
    usadas = []
    for ac in list(bpy.data.actions):
        if conservar is not None and ac.name not in conservar:
            bpy.data.actions.remove(ac)
            continue
        nuevo = nombre_clip(ac.name)
        if nuevo != ac.name:
            ac.name = nuevo
        ac.use_fake_user = True
        usadas.append(ac)
    return usadas


def exportar(arm, meshes, out):
    bpy.ops.object.select_all(action="DESELECT")
    for o in [arm] + list(meshes):
        o.select_set(True)
    bpy.context.view_layer.objects.active = arm
    # una pista NLA por clip (el modo ACTIONS del exportador no ve las acciones del importador de glTF)
    arm.animation_data_clear()
    arm.animation_data_create()
    for ac in sorted(bpy.data.actions, key=lambda a: a.name):
        tr = arm.animation_data.nla_tracks.new()
        tr.name = ac.name
        st = tr.strips.new(ac.name, 0, ac)
        st.name = ac.name
        if ac.slots:
            st.action_slot = ac.slots[0]
    bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=True, export_animations=True,
                              export_animation_mode="NLA_TRACKS", export_skins=True, export_image_format="AUTO",
                              export_force_sampling=True, export_optimize_animation_size=False, export_yup=True)


# --------------------------------------------------------------------------- cola
def cadena_cola(huesos, M, ancho_l):
    """huesos: [(nombre, padre|None, head(np3), tail(np3))] en mundo; M: Medidas (marco f,l,z).
    La cola = cadena de huesos desde el hueso hoja mas trasero sobre la linea central hasta justo antes de una ramificacion.
    Devuelve [(nombre, head, tail)] de raiz a punta (o [])."""
    hijos = {}
    for n, p, h, t in huesos:
        hijos.setdefault(p, []).append(n)
    info = {n: (p, h, t) for n, p, h, t in huesos}

    def fl(v):
        return float(np.dot(v, np.array(M.u))), float(np.dot(v, np.array(M.lat)))
    hojas = [n for n in info if n not in hijos]
    cand = [n for n in hojas if abs(fl(info[n][2])[1] - M.lc) < 0.15 * ancho_l]
    if not cand:
        return []
    punta = min(cand, key=lambda n: fl(info[n][2])[0])
    cad = [punta]
    while info[cad[-1]][0] is not None and len(hijos.get(info[cad[-1]][0], [])) == 1 and len(cad) < 14:
        cad.append(info[cad[-1]][0])
    cad.reverse()
    return [(n, info[n][1], info[n][2]) for n in cad]


def remuestrear(pts, fracs):
    """Puntos de la polilinea `pts` (k,3) a las fracciones de longitud de arco `fracs` (0..1)."""
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    t = np.asarray(fracs) * cum[-1]
    return np.stack([np.interp(t, cum, pts[:, c]) for c in range(3)], 1)


def procesar_pesos(m, suavizar=3, factor=0.5, maxinf=4):
    """Pesos de la malla -> numpy; suavizado por adyacencia (la malla fuente es de baja resolucion: los pesos por cara salen en
    bloques), maximo `maxinf` influencias por vertice, normalizado; se reescriben los grupos."""
    n = len(m.data.vertices)
    nom = [g.name for g in m.vertex_groups]
    W = np.zeros((n, len(nom)), np.float32)
    for v in m.data.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    e = np.zeros(len(m.data.edges) * 2, np.int32)
    m.data.edges.foreach_get("vertices", e)
    e = e.reshape(-1, 2)
    deg = np.zeros(n, np.float32)
    np.add.at(deg, e[:, 0], 1)
    np.add.at(deg, e[:, 1], 1)
    deg = np.maximum(deg, 1)[:, None]
    for _ in range(suavizar):
        acc = np.zeros_like(W)
        np.add.at(acc, e[:, 0], W[e[:, 1]])
        np.add.at(acc, e[:, 1], W[e[:, 0]])
        W = (1 - factor) * W + factor * acc / deg
    if W.shape[1] > maxinf:
        corte = np.sort(W, axis=1)[:, -maxinf][:, None]
        W = np.where(W >= corte, W, 0)
    tot = W.sum(1, keepdims=True)
    W = np.where(tot > 1e-8, W / np.maximum(tot, 1e-8), 0)
    m.vertex_groups.clear()
    for gi, nm in enumerate(nom):
        idx = np.nonzero(W[:, gi] > 1e-3)[0]
        if len(idx) == 0:
            continue
        g = m.vertex_groups.new(name=nm)
        for vi in idx:
            g.add([int(vi)], float(W[vi, gi]), "REPLACE")


def sobrescribir_cola(m, pesos, ti):
    """Los vertices que en el rig original pertenecen a la cola (suma de pesos en su cadena > 0.5) toman pesos por proyeccion
    sobre la cadena de cola ya re-posada: la transferencia por cara mas cercana falla con colas gruesas (las pega a la pierna)."""
    n = len(m.data.vertices)
    suma = np.zeros(n)
    for nom in ti["nuestros"]:
        for vi, w in pesos.get(nom, {}).items():
            suma[vi] += w
    mask = np.where(suma > 0.5)[0]
    if len(mask) == 0:
        log("AVISO: ningun vertice marcado como cola en el rig original")
        return
    P = puntos([m])[mask]
    pts = ti["pts"]
    K = len(pts) - 1
    a, b = pts[:-1], pts[1:]
    ab = b - a
    t = np.clip(np.einsum("nsk,sk->ns", P[:, None, :] - a[None], ab) / np.einsum("sk,sk->s", ab, ab)[None], 0, 1)
    d = np.linalg.norm(P[:, None, :] - (a[None] + t[:, :, None] * ab[None]), axis=2)
    k = np.argmin(d, axis=1)
    j = k + t[np.arange(len(mask)), k]          # coordenada continua a lo largo de la cadena (0..K)
    nombres = ti["nombres"]
    for vi in mask:
        for g in m.vertex_groups:
            try:
                g.remove([int(vi)])
            except RuntimeError:
                pass
    grp = {}
    def gr(nm):
        if nm not in grp:
            grp[nm] = m.vertex_groups.get(nm) or m.vertex_groups.new(name=nm)
        return grp[nm]
    for vi, jj in zip(mask, j):
        w = {}
        for kk in range(K):
            wk = 1 - abs(jj - (kk + 0.5))
            if wk > 0:
                w[nombres[kk]] = wk
        if jj < 0.5 and ti["padre"]:
            w[ti["padre"]] = 0.5 - jj
        tot = sum(w.values())
        for nm, wk in w.items():
            gr(nm).add([int(vi)], wk / tot, "REPLACE")
    log(f"cola: {len(mask)} vertices re-pesados por proyeccion sobre {K} huesos")


# --------------------------------------------------------------------------- METODO B
def metodo_b(ours_path, src_path, adelante="auto"):
    src_n = importar(src_path)
    our_n = importar(ours_path)
    quitar_formas_hueso()
    src_objs = [bpy.data.objects[n] for n in src_n if n in bpy.data.objects]
    our_objs = [bpy.data.objects[n] for n in our_n if n in bpy.data.objects]
    src_arm = next(o for o in src_objs if o.type == "ARMATURE")
    our_arm = next((o for o in our_objs if o.type == "ARMATURE"), None)
    our_meshes = [o for o in our_objs if o.type == "MESH"]
    src_meshes = [o for o in hijos_malla(src_arm)]
    acciones = list(bpy.data.actions)
    # --- nuestra malla: sin rig, en el suelo (se guardan sus pesos originales: sirven para segmentar la cola)
    pesos_nuestros = {}
    for m in our_meshes:
        for md in [md for md in m.modifiers if md.type == "ARMATURE"]:
            m.modifiers.remove(md)
        nom_g = {g.index: g.name for g in m.vertex_groups}
        pesos_nuestros[m.name] = {}
        for vi, v in enumerate(m.data.vertices):
            for g in v.groups:
                pesos_nuestros[m.name].setdefault(nom_g[g.group], {})[vi] = g.weight
        m.vertex_groups.clear()
        wm = m.matrix_world.copy()
        m.parent = None
        m.matrix_world = wm
    our_bones = []
    if our_arm:
        MW = np.array(our_arm.matrix_world)
        for b in our_arm.data.bones:
            our_bones.append((b.name, b.parent.name if b.parent else None, MW[:3, :3] @ np.array(b.head_local) + MW[:3, 3],
                              MW[:3, :3] @ np.array(b.tail_local) + MW[:3, 3]))
        bpy.data.objects.remove(our_arm, do_unlink=True)
    PT = puntos(our_meshes)
    dz = -PT[:, 2].min()
    our_bones = [(n, p, h + [0, 0, dz], t + [0, 0, dz]) for n, p, h, t in our_bones]
    for m in our_meshes:
        P = puntos([m])
        P[:, 2] += dz
        escribir_puntos(m, P)
    PT = puntos(our_meshes)
    PS = puntos(src_meshes)
    uT = eje_adelante(PT, adelante)
    uS = eje_adelante(PS, "auto")
    T = Medidas(PT, uT, "nuestro")
    S = Medidas(PS, uS, "fuente")
    W = Warp(S, T)

    # --- deformar malla fuente y huesos en reposo
    for m in src_meshes:
        P2, _ = W(puntos([m]))
        escribir_puntos(m, P2)
    bpy.context.view_layer.objects.active = src_arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = src_arm.data.edit_bones
    viejo = {}
    for b in eb:
        viejo[b.name] = (np.array(b.head), np.array(b.tail), np.array(b.z_axis), np.array(b.matrix.to_3x3()))
    nombres = list(viejo)
    H0 = np.array([viejo[n][0] for n in nombres])
    T0 = np.array([viejo[n][1] for n in nombres])
    Z0 = np.array([viejo[n][2] for n in nombres])
    H1, JH = W(H0)
    T1, _ = W(T0)
    for b in eb:
        b.use_connect = False
    for i, n in enumerate(nombres):
        b = eb[n]
        b.head, b.tail = Vector(H1[i]), Vector(T1[i])
        b.align_roll(Vector(JH[i] @ Z0[i]))
    nuevo = {n: np.array(eb[n].matrix.to_3x3()) for n in nombres}
    # traslaciones locales: loc' = R'^-1 J R loc (con el reposo ya deformado, antes de re-posar la cola)
    Mloc = {n: np.linalg.inv(nuevo[n]) @ JH[i] @ viejo[n][3] for i, n in enumerate(nombres)}
    # --- cola: la cola de Quaternius sale recta hacia atras; la nuestra puede colgar. Se re-posa la cadena para que siga la nuestra
    tail_T, tail_info = {}, None
    if our_bones:
        ct = cadena_cola(our_bones, T, T.ancho)
        S3 = Medidas(puntos(src_meshes), T.u, "fuente-deformada")
        bones_s = [(b.name, b.parent.name if b.parent else None, np.array(b.head), np.array(b.tail)) for b in eb]
        cs = cadena_cola(bones_s, S3, S3.ancho)
        if ct and cs and len(ct) >= 2:
            ptsT = [c[1] for c in ct] + [ct[-1][2]]
            ptsS = [c[1] for c in cs] + [cs[-1][2]]
            seg = np.linalg.norm(np.diff(np.array(ptsS), axis=0), axis=1)
            fr = np.r_[0, np.cumsum(seg)] / seg.sum()
            nuevo_p = remuestrear(ptsT, fr)
            log(f"cola: fuente {len(cs)} huesos ({cs[0][0]}..{cs[-1][0]}), nuestra {len(ct)} huesos ({ct[0][0]}..{ct[-1][0]}); largo {seg.sum():.3f} -> {np.linalg.norm(np.diff(np.array(ptsT), axis=0), axis=1).sum():.3f}")
            R_old = {}
            for k, (n, h, t) in enumerate(cs):
                b = eb[n]
                R_old[n] = np.array(b.matrix)
                d_old = Vector(t - h).normalized()
                d_new = Vector(nuevo_p[k + 1] - nuevo_p[k]).normalized()
                z_new = d_old.rotation_difference(d_new) @ Vector(b.z_axis)
                b.head, b.tail = Vector(nuevo_p[k]), Vector(nuevo_p[k + 1])
                b.align_roll(z_new)
            tail_T = {n: np.array(eb[n].matrix) @ np.linalg.inv(R_old[n]) for n, _, _ in cs}
            padre = eb[cs[0][0]].parent.name if eb[cs[0][0]].parent else None
            tail_info = {"nombres": [c[0] for c in cs], "pts": nuevo_p, "padre": padre, "nuestros": [c[0] for c in ct]}
        else:
            log("AVISO: no se pudo identificar la cola en alguno de los dos esqueletos; se deja como esta")
    bpy.ops.object.mode_set(mode="OBJECT")
    reescalar_traslaciones(acciones, Mloc)
    if tail_T:  # mezcla lineal de la malla fuente con las transformaciones de la cola
        for sm in src_meshes:
            P = puntos([sm])
            D = np.zeros_like(P)
            gi = {g.index: g.name for g in sm.vertex_groups}
            for vi, v in enumerate(sm.data.vertices):
                for g in v.groups:
                    n = gi[g.group]
                    if n in tail_T and g.weight > 0:
                        Tm = tail_T[n]
                        D[vi] += g.weight * ((Tm[:3, :3] @ P[vi] + Tm[:3, 3]) - P[vi])
            escribir_puntos(sm, P + D)

    # --- transferencia de pesos fuente -> nuestra malla
    for sm in src_meshes:
        sm.parent = None  # la fuente ya esta en mundo; el padre no importa para la transferencia
    for m in our_meshes:
        for sm in src_meshes:
            for g in sm.vertex_groups:
                if g.name not in m.vertex_groups:
                    m.vertex_groups.new(name=g.name)
            md = m.modifiers.new("dt_pesos", "DATA_TRANSFER")  # el operador object.data_transfer no escribe nada en modo -b; el modificador si
            md.object = sm
            md.use_vert_data = True
            md.data_types_verts = {"VGROUP_WEIGHTS"}
            md.vert_mapping = "POLYINTERP_NEAREST"
            md.layers_vgroup_select_src = "ALL"
            md.layers_vgroup_select_dst = "NAME"
            md.mix_mode, md.mix_factor = "REPLACE", 1.0
            bpy.context.view_layer.objects.active = m
            bpy.ops.object.modifier_apply(modifier=md.name)
        procesar_pesos(m)
        log(f"vertices con pesos tras transferir: {sum(1 for v in m.data.vertices if v.groups)}/{len(m.data.vertices)}")
        if tail_info:
            sobrescribir_cola(m, pesos_nuestros.get(m.name, {}), tail_info)
        log(f"grupos de vertices finales: {len(m.vertex_groups)}")
        m.parent = src_arm
        m.matrix_parent_inverse = Matrix.Identity(4)
        md = m.modifiers.new("Armature", "ARMATURE")
        md.object = src_arm
    for sm in src_meshes:
        bpy.data.objects.remove(sm, do_unlink=True)
    return src_arm, our_meshes, {"adelante": list(uT), "metodo": "b"}


# --------------------------------------------------------------------------- METODO A
def rotq(m):
    return m.to_3x3().normalized().to_quaternion()


def bones_world(arm, validos=None):
    MW = np.array(arm.matrix_world)
    out = []
    for b in arm.data.bones:
        if validos is not None and b.name not in validos:
            continue
        p = b.parent.name if b.parent and (validos is None or b.parent.name in validos) else None
        out.append((b.name, p, MW[:3, :3] @ np.array(b.head_local) + MW[:3, 3], MW[:3, :3] @ np.array(b.tail_local) + MW[:3, 3]))
    return out


def cadenas_animal(huesos, M):
    """Cadenas semanticas por posicion: patas (4), cola, columna+cuello+cabeza. Cada cadena = lista de nombres raiz->punta."""
    hijos, info = {}, {}
    for n, p, h, t in huesos:
        hijos.setdefault(p, []).append(n)
        info[n] = (p, h, t)
    u, lat = np.array(M.u), np.array(M.lat)
    hojas = [n for n in info if n not in hijos]
    H = M.zmax - M.zmin
    out = {"patas": {}}
    for key, (ff, ll) in M.feet.items():
        cand = [n for n in hojas if info[n][2][2] < M.zmin + 0.5 * H and abs(info[n][2] @ lat - M.lc) > 0.12 * M.ancho]
        cand.sort(key=lambda n: (info[n][2] @ u - ff) ** 2 + (info[n][2] @ lat - ll) ** 2)
        cad = None
        for leaf in cand:   # la cadena debe colgar del cuerpo (los controles IK son raices sueltas)
            c = [leaf]
            while info[c[-1]][0] is not None and len(hijos[info[c[-1]][0]]) == 1 and len(c) < 10:
                c.append(info[c[-1]][0])
            if info[c[-1]][0] is not None:
                cad = c
                break
        if cad is None:
            continue
        out["patas"][key] = cad[::-1]
    out["cola"] = [c[0] for c in cadena_cola(huesos, M, M.ancho)]
    centro = [n for n in hojas if abs(info[n][2] @ lat - M.lc) < 0.15 * M.ancho]
    nariz = max(centro, key=lambda n: info[n][2] @ u)
    cad = [nariz]
    while info[cad[-1]][0] is not None:
        cad.append(info[cad[-1]][0])
    out["columna"] = cad[::-1]
    out["info"] = info
    return out


def polilinea(info, cad):
    pts = [info[n][1] for n in cad] + [info[cad[-1]][2]]
    return np.array(pts)


def emparejar_cadenas(CT, CS, rz):
    """-> {hueso_destino: (lista de (hueso_fuente, peso), vec_destino, vec_fuente_en_mundo_destino)}"""
    pares = []
    for k in set(CT["patas"]) & set(CS["patas"]):
        pares.append((CT["patas"][k], CS["patas"][k]))
    for nm in ("cola", "columna"):
        if CT[nm] and CS[nm]:
            pares.append((CT[nm], CS[nm]))
    res = {}
    for ct, cs in pares:
        PT, PS = polilinea(CT["info"], ct), polilinea(CS["info"], cs)
        fT = np.r_[0, np.cumsum(np.linalg.norm(np.diff(PT, axis=0), axis=1))]
        fT /= fT[-1]
        fS = np.r_[0, np.cumsum(np.linalg.norm(np.diff(PS, axis=0), axis=1))]
        fS /= fS[-1]
        cenS = (fS[:-1] + fS[1:]) / 2
        for i, n in enumerate(ct):
            c = (fT[i] + fT[i + 1]) / 2
            x = float(np.interp(c, cenS, np.arange(len(cs))))
            k0 = int(math.floor(x))
            k1 = min(k0 + 1, len(cs) - 1)
            a = x - k0
            # vector fuente en el mismo tramo de fracciones
            sA = remuestrear(PS, [fT[i]])[0]
            sB = remuestrear(PS, [fT[i + 1]])[0]
            vs = Vector(rz @ (sB - sA))
            vt = Vector(PT[i + 1] - PT[i])
            res[n] = ([(cs[k0], 1 - a), (cs[k1], a)], vt, vs)
    return res


def metodo_a(ours_path, src_path, adelante="auto"):
    src_n = importar(src_path)
    our_n = importar(ours_path)
    quitar_formas_hueso()
    src = next(bpy.data.objects[n] for n in src_n if n in bpy.data.objects and bpy.data.objects[n].type == "ARMATURE")
    tgt = next(bpy.data.objects[n] for n in our_n if n in bpy.data.objects and bpy.data.objects[n].type == "ARMATURE")
    tmeshes = hijos_malla(tgt)
    smeshes = hijos_malla(src)
    acciones = list(bpy.data.actions)
    for pb in tgt.pose.bones:
        pb.rotation_mode = "QUATERNION"
    # bajar nuestro modelo al suelo (huesos y malla)
    PT = puntos(tmeshes)
    dz0 = -float(PT[:, 2].min())
    bpy.context.view_layer.objects.active = tgt
    bpy.ops.object.mode_set(mode="EDIT")
    for eb in tgt.data.edit_bones:
        eb.head.z += dz0
        eb.tail.z += dz0
    bpy.ops.object.mode_set(mode="OBJECT")
    for m in tmeshes:
        P = puntos([m])
        P[:, 2] += dz0
        escribir_puntos(m, P)
    bpy.context.view_layer.update()
    PT, PS = puntos(tmeshes), puntos(smeshes)
    uT, uS = eje_adelante(PT, adelante), eje_adelante(PS, "auto")
    T, S = Medidas(PT, uT, "nuestro"), Medidas(PS, uS, "fuente")
    # solo huesos con piel (fuera controles IK / polos) y sus ancestros
    pesados = set()
    for m in smeshes:
        nom = {g.index: g.name for g in m.vertex_groups}
        pesados |= {nom[g.group] for v in m.data.vertices for g in v.groups if g.weight > 0.01}
    cerrado = set(pesados)
    for b in src.data.bones:
        if b.name in pesados:
            cerrado |= {a.name for a in b.parent_recursive}
    CS = cadenas_animal(bones_world(src, cerrado), S)
    CT = cadenas_animal(bones_world(tgt), T)
    ang_z = math.atan2(uT.y, uT.x) - math.atan2(uS.y, uS.x)
    Rz = Matrix.Rotation(ang_z, 3, "Z")
    rz = np.array(Rz)
    qz = Rz.to_quaternion()
    log("patas destino:", {f"{k[0]}{'+' if k[1] > 0 else '-'}": v for k, v in CT["patas"].items()})
    log("patas fuente:", {f"{k[0]}{'+' if k[1] > 0 else '-'}": v for k, v in CS["patas"].items()})
    log("cola destino:", CT["cola"], " fuente:", CS["cola"])
    log("columna destino:", CT["columna"], " fuente:", CS["columna"])
    mapa = emparejar_cadenas(CT, CS, rz)
    B = tgt.data.bones
    orden = sorted(B, key=lambda b: len(b.parent_recursive))
    rest_rel = {b.name: (b.parent.matrix_local.inverted() @ b.matrix_local if b.parent else b.matrix_local.copy()) for b in orden}
    t_rest = {b: rotq(B[b].matrix_local) for b in mapa}
    ang = {b: vt.rotation_difference(vs) if vt.length > 1e-6 and vs.length > 1e-6 else Quaternion() for b, (_, vt, vs) in mapa.items()}
    ang[CT["columna"][0]] = Quaternion()   # la raiz de Quaternius ("Body") es un pivote vertical: no se alinea por vector
    s_rest = {n: rotq(src.data.bones[n].matrix_local) for v in mapa.values() for n, _ in v[0]}
    raiz = CT["columna"][0]
    raiz_s = CS["columna"][0]
    escala = (T.back_z - T.zmin) / (S.back_z - S.zmin)
    t_head = Vector(CT["info"][raiz][1])
    s_head = Vector(CS["info"][raiz_s][1])
    sc = bpy.context.scene
    nuevas = {}
    for ac in acciones:
        src.animation_data_create()
        src.animation_data.action = ac
        if ac.slots:
            src.animation_data.action_slot = ac.slots[0]
        f0, f1 = ac.frame_range
        n = int(round(f1 - f0)) + 1
        keys = {b: [] for b in mapa}
        locs = []
        zs = []
        for i in range(n):
            sc.frame_set(int(f0) + i)
            dwq = {nm: rotq(src.pose.bones[nm].matrix) @ s_rest[nm].inverted() for nm in s_rest}
            pose = {}
            for b in orden:
                Mp = pose[b.parent.name] if b.parent else Matrix.Identity(4)
                base = Mp @ rest_rel[b.name]
                if b.name not in mapa:
                    pose[b.name] = base
                    continue
                (s0, w0), (s1, w1) = mapa[b.name][0]
                dw = dwq[s0].slerp(dwq[s1], w1) if s0 != s1 else dwq[s0]
                dw = qz @ dw @ qz.inverted()
                lq = rotq(base).inverted() @ (dw @ ang[b.name] @ t_rest[b.name])
                if keys[b.name] and lq.dot(keys[b.name][-1]) < 0:
                    lq.negate()
                loc = Vector()
                if b.name == raiz:
                    hs = src.matrix_world @ src.pose.bones[raiz_s].head
                    want = t_head + Rz @ (hs - s_head) * escala
                    loc = base.to_3x3().inverted() @ (want - base.translation)
                    locs.append(loc.copy())
                pose[b.name] = base @ Matrix.Translation(loc) @ lq.to_matrix().to_4x4()
                keys[b.name].append(lq)
        nuevas[ac.name] = (n, keys, locs)
    src.animation_data.action = None
    # escribir acciones en el destino, con ajuste de suelo constante por clip (el punto mas bajo de la malla toca Z=0)
    for ac in list(bpy.data.actions):
        bpy.data.actions.remove(ac)
    info = {"metodo": "a", "mapeados": len(mapa), "dz_suelo_inicial": round(dz0, 4)}
    for nm, (n, keys, locs) in nuevas.items():
        act = escribir_accion(tgt, nm, n, keys, locs, [0.0] * n, raiz)
        z = []
        for i in sorted(set(list(range(0, n, max(1, n // 12))) + [n - 1])):
            sc.frame_set(i)
            z.append(zmin_malla(tmeshes))
        zm = min(z)
        dzs = (-zm if (zm < -0.01 or zm > 0.004) else 0.0)
        escribir_accion(tgt, nm, n, keys, locs, [dzs] * n, raiz)
        info.setdefault("ajuste_z_cm", {})[nm] = round(dzs * 100, 1)
    tgt.animation_data.action = None
    for o in [o for o in bpy.data.objects if o in [src] + smeshes]:
        bpy.data.objects.remove(o, do_unlink=True)
    sc.frame_set(0)
    return tgt, tmeshes, info


def zmin_malla(meshes):
    dg = bpy.context.evaluated_depsgraph_get()
    z = 1e9
    for m in meshes:
        ev = m.evaluated_get(dg)
        me = ev.to_mesh()
        co = np.zeros(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        z = min(z, float((co @ np.array(m.matrix_world)[:3, :3].T)[:, 2].min()))
        ev.to_mesh_clear()
    return z


def escribir_accion(tgt, nombre, n, keys, locs, dz, hips):
    """Accion `nombre` en tgt: cuaterniones locales por hueso + traslacion de la raiz (+ dz mundo)."""
    from bpy_extras import anim_utils
    if nombre in bpy.data.actions:
        bpy.data.actions.remove(bpy.data.actions[nombre])
    B = tgt.data.bones
    Rinv = B[hips].matrix_local.to_3x3().inverted()
    act = bpy.data.actions.new(nombre)
    act.use_fake_user = True
    slot = act.slots.new(id_type="OBJECT", name=tgt.name)
    tgt.animation_data_create()
    tgt.animation_data.action = act
    tgt.animation_data.action_slot = slot
    cb = anim_utils.action_ensure_channelbag_for_slot(act, slot)
    lin = bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items["LINEAR"].value

    def write(path, idx, grupo, vals):
        fc = cb.fcurves.ensure(path, index=idx, group_name=grupo)
        fc.keyframe_points.add(len(vals))
        fc.keyframe_points.foreach_set("co", [v for i, x in enumerate(vals) for v in (float(i), float(x))])
        fc.keyframe_points.foreach_set("interpolation", [lin] * len(vals))
        fc.update()
    for b, qs in keys.items():
        tgt.pose.bones[b].rotation_mode = "QUATERNION"
        for c in range(4):
            write(f'pose.bones["{b}"].rotation_quaternion', c, b, [q[c] for q in qs])
    L = [locs[i] + Rinv @ Vector((0, 0, dz[i])) for i in range(n)]
    for c in range(3):
        write(f'pose.bones["{hips}"].location', c, hips, [v[c] for v in L])
    return act


def autotest():
    x, d = pwl([-1, 0.5, 2, 5], [0, 1, 3], [10, 20, 40])
    assert np.allclose(x, [0, 15, 30, 60]), x
    p = remuestrear([[0, 0, 0], [1, 0, 0], [1, 1, 0]], [0, 0.5, 1])
    assert np.allclose(p, [[0, 0, 0], [1, 0, 0], [1, 1, 0]]), p
    assert nombre_clip("Walk") == "Walk_Loop" and nombre_clip("Death") == "Death" and nombre_clip("Idle_HitReact1") == "Idle_HitReact1"
    assert nombre_clip("Idle_2_HeadLow") == "Idle_2_HeadLow_Loop" and nombre_clip("Gallop_Jump") == "Gallop_Jump" and nombre_clip("Attack_Kick") == "Attack_Kick"
    print("ok autotest retarget_animal")


def main():
    if "--autotest" in sys.argv:
        return autotest()
    a = sys.argv[sys.argv.index("--") + 1:]
    pos = [x for x in a if not x.startswith("--")]
    opt = {}
    i = 0
    while i < len(a):
        if a[i].startswith("--") and i + 1 < len(a) and not a[i + 1].startswith("--"):
            opt[a[i]] = a[i + 1]
            i += 2
        else:
            if a[i].startswith("--"):
                opt[a[i]] = True
            i += 1
    pos = [x for x in a if not x.startswith("--") and x not in opt.values()]
    if len(pos) != 3:
        raise SystemExit("uso: blender -b -P retarget_animal.py -- <nuestro.glb> <fuente.gltf> <salida.glb> [--metodo a|b] [--adelante auto|+x|-x|+y|-y]")
    ours, src, out = pos
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = FPS
    bpy.context.scene.render.fps_base = 1.0
    metodo = opt.get("--metodo", "b")
    if metodo == "a":
        log("AVISO: el metodo a es experimental y da malos resultados en cuadrupedos; use b")
    r = (metodo_b if metodo == "b" else metodo_a)(ours, src, opt.get("--adelante", "auto"))
    arm, meshes, info = r
    acts = limpiar_acciones(arm)
    info["clips"] = sorted(a.name for a in acts)
    if "--save-blend" in opt:
        bpy.ops.wm.save_as_mainfile(filepath=opt["--save-blend"], copy=True)
    exportar(arm, meshes, out)
    json.dump(info, open(os.path.splitext(out)[0] + "_info.json", "w"), indent=1)
    print("@@RESUMEN@@" + json.dumps(info))


if __name__ == "__main__":
    main()
