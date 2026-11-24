"""Plantillas de animación mecánica genéricas: vehiculo, catapulta, rotor.
Cada plantilla: detecta -> separa por caras -> objetos con origen en el pivote -> clips (NLA) -> exporta glb.
Sin coordenadas fijas: todo sale de detecta_piezas."""
import bpy, bmesh, math, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detecta_piezas import *

D = math.radians

# ------------------------------------------------------------------ utilidades de construcción
def sub_objeto(src, faces_mask, name):
    """Copia de src conservando UV/material, solo con las caras de la máscara."""
    me = src.data
    bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
    kill = [f for f in bm.faces if not faces_mask[f.index]]
    bmesh.ops.delete(bm, geom=kill, context='FACES')
    nm = bpy.data.meshes.new(name); bm.to_mesh(nm); bm.free()
    for m in me.materials: nm.materials.append(m)
    ob = bpy.data.objects.new(name, nm); bpy.context.scene.collection.objects.link(ob)
    return ob

def normales_afuera(ob, tol=1e-4, umbral_vol=2e-3):
    """Orienta hacia fuera las caras de una pieza separada. Las islas UV (charts) no sirven para decidir el sentido
    (cada isla suelta saldria al azar): se sueldan vertices coincidentes (tolerancia relativa a la pieza), se calcula el
    volumen con signo de cada componente soldada y, si es negativo (normales hacia dentro), se voltean todas sus caras
    (y sus normales personalizadas). Las componentes abiertas/planas (|volumen| casi nulo) no se tocan.
    Devuelve el numero de caras volteadas."""
    me = ob.data
    n_v, n_f = len(me.vertices), len(me.polygons)
    if n_f == 0:
        return 0
    co = coords(me)
    tri_l = me.loop_triangles
    tri = np.empty(len(tri_l) * 3, dtype=np.int64); tri_l.foreach_get('vertices', tri); tri = tri.reshape(-1, 3)
    pol = np.empty(len(tri_l), dtype=np.int64); tri_l.foreach_get('polygon_index', pol)
    diag = max(float(np.ptp(co, axis=0).max()), 1e-9)
    key = np.round(co / (tol * diag)).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    lab = charts_de(int(inv.max()) + 1, inv[tri])
    lt = lab[inv[tri[:, 0]]]
    v0, v1, v2 = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
    vol6 = np.einsum('ij,ij->i', v0, np.cross(v1, v2))
    vol = np.bincount(lt, weights=vol6, minlength=int(lt.max()) + 1) / 6.0
    bbox_vol = float(np.prod(np.maximum(np.ptp(co, axis=0), 1e-9)))
    # solo componentes cerradas (cada arista compartida por 2 caras): en las abiertas el signo del volumen no significa nada
    ti = inv[tri]
    e = np.sort(np.concatenate([ti[:, [0, 1]], ti[:, [1, 2]], ti[:, [2, 0]]]), axis=1)
    elab = np.concatenate([lt, lt, lt])
    ek, ecnt = np.unique(np.concatenate([e, elab[:, None]], axis=1), axis=0, return_counts=True)
    abierta = np.zeros(len(vol), bool)
    abierta[ek[ecnt != 2, 2]] = True
    flip_comp = (vol < -umbral_vol * bbox_vol * 0.05) & ~abierta
    # el sentido de cada cara lo decide su componente; una cara = varios triangulos de la misma componente
    flip_face = np.zeros(n_f, bool)
    flip_face[pol[flip_comp[lt]]] = True
    nflip = int(flip_face.sum())
    if nflip == 0:
        return 0
    cn = np.array([c.vector[:] for c in me.corner_normals]) if me.has_custom_normals else None
    bm = bmesh.new(); bm.from_mesh(me); bm.faces.ensure_lookup_table()
    for i in np.where(flip_face)[0]:
        bm.faces[i].normal_flip()
    bm.to_mesh(me); bm.free()
    if cn is not None:
        lens = np.array([p.loop_total for p in me.polygons])
        cn[np.repeat(flip_face, lens)] *= -1
        me.normals_split_custom_set([tuple(v) for v in cn])
    me.update()
    return nflip

def coords(me):
    v = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', v); return v.reshape(-1, 3)

def set_origin(ob, pt):
    pt = np.asarray(pt, float); me = ob.data
    me.vertices.foreach_set('co', (coords(me) - pt).ravel()); me.update()
    ob.location = tuple(pt)

def quat(axis, ang):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    s = math.sin(ang / 2); return (math.cos(ang / 2), a[0] * s, a[1] * s, a[2] * s)

def _fcurves(act, owner):
    try:
        from bpy_extras import anim_utils
        return list(anim_utils.action_get_channelbag_for_slot(act, owner.animation_data.action_slot).fcurves)
    except Exception:
        return list(act.fcurves)

def key_rot(ob, accion, pista, frames_angulos, eje, pb=None, interp='LINEAR'):
    """Keys de rotación (cuaternión) sobre eje (local del objeto/hueso). Crea acción y la pasa a pista NLA."""
    tgt = pb if pb is not None else ob
    tgt.rotation_mode = 'QUATERNION'
    ob.animation_data_create()
    act = bpy.data.actions.new(accion); ob.animation_data.action = act
    path = pb.path_from_id('rotation_quaternion') if pb is not None else 'rotation_quaternion'
    for fr, ang, it in frames_angulos:
        tgt.rotation_quaternion = quat(eje, ang)
        ob.keyframe_insert(data_path=path, frame=fr)
    fcs = [f for f in _fcurves(act, ob) if f.data_path == path]
    fr_list = [f[0] for f in frames_angulos]
    for fc in fcs:
        for kp, (fr, ang, it) in zip(fc.keyframe_points, frames_angulos):
            kp.interpolation = it[0] if isinstance(it, tuple) else it
            kp.easing = it[1] if isinstance(it, tuple) else 'AUTO'
        fc.update()
    tr = ob.animation_data.nla_tracks.new(); tr.name = pista
    st = tr.strips.new(accion, int(fr_list[0]), act); st.name = pista
    ob.animation_data.action = None
    tgt.rotation_quaternion = (1, 0, 0, 0)

def clip_rodar(wheels_obj, nvueltas=2, frames=60, spin=None):
    """Giro lineal continuo (loop perfecto): nvueltas en `frames`, keys cada 30 grados."""
    n = 12 * nvueltas
    kf = [(i * frames / n, i * 2 * math.pi / 12, 'LINEAR') for i in range(n + 1)]
    for ob, eje in wheels_obj:
        key_rot(ob, "rodar_" + ob.name, "rodar", kf, eje)

def exporta(out):
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.splitext(out)[0] + ".blend")
    bpy.ops.export_scene.gltf(filepath=out, export_format='GLB', export_animations=True,
        export_animation_mode='NLA_TRACKS', export_nla_strips=True, export_force_sampling=True,
        export_optimize_animation_size=False, export_image_format='AUTO', export_apply=False, export_yup=True)
    print("EXPORT OK", out)

def spin_axis(eje, fwd, up=np.array([0, 0, 1.0])):
    """Eje de giro con signo tal que el ángulo positivo hace avanzar lo de arriba hacia fwd."""
    a = np.cross(up, fwd); a /= np.linalg.norm(a)
    return eje if eje @ a >= 0 else -eje

def ruedas_a_objetos(src, M, W, fwd, root, prefijo="rueda"):
    objs = []
    for i, w in enumerate(W):
        mask = np.zeros(len(M.tri), bool); mask[w['faces']] = True
        ob = sub_objeto(src, mask, "%s_%d" % (prefijo, i + 1)); ob.parent = root
        set_origin(ob, w['centro'])
        objs.append((ob, spin_axis(w['eje'], fwd)))
    return objs

def preparar(path, verbose=True):
    src, co, tri = cargar(path)
    M = Malla(co, tri); ori = orienta(M)
    if verbose: print("malla: %d caras, %d charts, L=%.3f; orientación: fwd(eje largo)=%s snapped=%s" % (len(tri), M.nch, M.L, ori['fwd'].round(2), ori['snapped']))
    return src, M, ori

def _root(nombre):
    r = bpy.data.objects.new(nombre, None); bpy.context.scene.collection.objects.link(r); return r

# ------------------------------------------------------------------ plantilla: vehiculo
def limpia_ruedas(M, W, min_caras=12, verbose=True):
    """Tras separar las ruedas por mayoría de chart: (1) los trozos cortados de una pieza grande del chasis (componente
    soldada que solo en parte cae en la rueda) vuelven al chasis; (2) las islas diminutas completas (< min_caras caras)
    se eliminan de toda la malla (motas). Devuelve la máscara de caras a descartar."""
    fl = componentes_soldadas(M)
    nfc = np.bincount(fl)
    descarta = (nfc < min_caras)[fl]
    for w in W:
        f = w['faces']
        en_rueda = np.bincount(fl[f], minlength=len(nfc))
        principal = int(np.argmax(en_rueda))
        trozo = (en_rueda > 0) & (en_rueda < 0.5 * nfc) & (np.arange(len(nfc)) != principal)   # fragmento de algo mayor
        w['faces'] = f[~trozo[fl[f]] & ~descarta[f]]
        if verbose and (trozo.sum() or descarta[f].sum()):
            print("  rueda: %d fragmentos devueltos al chasis, %d caras de motas eliminadas" % (trozo.sum(), descarta[f].sum()))
    if verbose: print("  motas eliminadas del modelo: %d caras (%d islas)" % (descarta.sum(), int((nfc < min_caras).sum())))
    return descarta

def vehiculo(entrada, salida, adelante=None):
    src, M, ori = preparar(entrada)
    fwd = np.array(adelante, float) if adelante is not None else ori['fwd']
    W = detecta_ruedas(M)
    if not W: raise RuntimeError("vehiculo: no se detectaron ruedas")
    descarta = limpia_ruedas(M, W)
    wf = np.concatenate([w['faces'] for w in W])
    root = _root("vehiculo")
    objs = ruedas_a_objetos(src, M, W, fwd, root)
    for ob, _ in objs: print("  normales volteadas", ob.name, normales_afuera(ob))
    mask = np.ones(len(M.tri), bool); mask[wf] = False; mask &= ~descarta
    chasis = sub_objeto(src, mask, "chasis"); chasis.parent = root
    print("  normales volteadas chasis", normales_afuera(chasis))
    bpy.data.objects.remove(src)
    clip_rodar(objs)
    sc = bpy.context.scene; sc.render.fps = 30; sc.frame_start = 0; sc.frame_end = 60
    exporta(salida)
    return dict(ruedas=len(W))

# ------------------------------------------------------------------ plantilla: catapulta
def catapulta(entrada, salida):
    src, M, ori = preparar(entrada)
    W = detecta_ruedas(M)
    wf = np.concatenate([w['faces'] for w in W]) if W else np.array([], int)
    A = detecta_brazo(M, wf, ori)
    if A is None: raise RuntimeError("catapulta: no se detectó brazo")
    cuerdas = detecta_cuerdas(M, A, wf)
    # adelante = horizontal de pivote->punta, alineado al eje largo si está cerca
    h = A['punta'] - A['pivote']; h[2] = 0; h /= np.linalg.norm(h)
    fwd = h
    for ax in (np.array([1, 0, 0.]), np.array([0, 1, 0.])):
        for sg in (1, -1):
            if h @ (sg * ax) > math.cos(D(12)): fwd = sg * ax
    lat = np.cross([0, 0, 1.0], fwd)
    print("catapulta: adelante", fwd.round(2), "pivote", A['pivote'].round(3), "punta", A['punta'].round(3), "ruedas", len(W), "cuerdas", len(cuerdas))
    root = _root("catapulta")
    nf = len(M.tri)
    wheel_objs = ruedas_a_objetos(src, M, W, fwd, root)
    brazo_m = np.zeros(nf, bool); brazo_m[A['faces']] = True
    cuer_m = [np.zeros(nf, bool) for _ in cuerdas]
    for m, r in zip(cuer_m, cuerdas): m[r['faces']] = True
    marco_m = np.ones(nf, bool); marco_m[wf] = False; marco_m[A['faces']] = False
    for m in cuer_m: marco_m &= ~m
    # no se puede pisar: cuerdas se calcularon excluyendo brazo
    marco = sub_objeto(src, marco_m, "marco"); marco.parent = root
    brazo = sub_objeto(src, brazo_m, "brazo")
    cobjs = [sub_objeto(src, m, "cuerda_%d" % (i + 1)) for i, m in enumerate(cuer_m)]
    bpy.data.objects.remove(src)
    # --- rig: fijo + hueso_brazo con eje = lateral, cabeza en el pivote
    ad = bpy.data.armatures.new("rig_data"); rig = bpy.data.objects.new("rig", ad)
    bpy.context.scene.collection.objects.link(rig); rig.parent = root
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    b0 = ad.edit_bones.new("fijo"); b0.head = (0, 0, 0); b0.tail = (0, 0, 0.1)
    b1 = ad.edit_bones.new("hueso_brazo"); b1.head = tuple(A['pivote']); b1.tail = tuple(A['pivote'] + lat * 0.1); b1.parent = b0
    bpy.ops.object.mode_set(mode='OBJECT')
    def skin(ob, w):
        vf = ob.vertex_groups.new(name="fijo"); va = ob.vertex_groups.new(name="hueso_brazo")
        for i, wi in enumerate(w):
            if wi < 0.9999: vf.add([i], 1.0 - float(wi), 'REPLACE')
            if wi > 0.0001: va.add([i], float(wi), 'REPLACE')
        md = ob.modifiers.new("Armature", 'ARMATURE'); md.object = rig; ob.parent = rig
    skin(brazo, np.ones(len(brazo.data.vertices)))
    for ob, r in zip(cobjs, cuerdas):
        p = coords(ob.data); ax = r['arm_end'] - r['frame_end']
        w = np.clip((p - r['frame_end']) @ ax / (ax @ ax), 0, 1)
        skin(ob, w)
    # --- clips
    s = 1.0  # signo: giro positivo sobre `lat` baja la punta (punta adelante, arriba = +Z)
    disp = [(0, 0.0, ('SINE', 'EASE_IN_OUT')), (10, D(11), 'CONSTANT'), (13, D(11), ('CUBIC', 'EASE_IN')),
            (18, D(-33), ('SINE', 'EASE_OUT')), (21, D(-29), ('SINE', 'EASE_IN_OUT')), (24, D(-33), ('SINE', 'EASE_IN_OUT')),
            (27, D(-33), ('SINE', 'EASE_IN_OUT')), (45, 0.0, ('SINE', 'EASE_IN_OUT'))]
    # el hueso gira sobre su eje local Y (= su dirección = lat)
    key_rot(rig, "disparar_brazo", "disparar", disp, (0, 1, 0), pb=rig.pose.bones["hueso_brazo"])
    clip_rodar(wheel_objs)
    sc = bpy.context.scene; sc.render.fps = 30; sc.frame_start = 0; sc.frame_end = 60
    exporta(salida)
    return dict(ruedas=len(W), cuerdas=len(cuerdas), pivote=A['pivote'].tolist())

# ------------------------------------------------------------------ plantilla: rotor
def rotor(entrada, salida):
    src, M, ori = preparar(entrada)
    R = detecta_rotor(M)
    if R is None: raise RuntimeError("rotor: no se detectaron aspas radiales")
    root = _root("rotor")
    mask = np.zeros(len(M.tri), bool); mask[R['faces']] = True
    rot = sub_objeto(src, mask, "rotor"); rot.parent = root
    base = sub_objeto(src, ~mask, "base"); base.parent = root
    bpy.data.objects.remove(src)
    set_origin(rot, R['centro'])
    n = 12; frames = 60
    kf = [(i * frames / n, i * 2 * math.pi / n, 'LINEAR') for i in range(n + 1)]
    key_rot(rot, "girar_rotor", "girar", kf, R['eje'])
    sc = bpy.context.scene; sc.render.fps = 30; sc.frame_start = 0; sc.frame_end = 60
    exporta(salida)
    return dict(aspas=len(R['aspas']))

PLANTILLAS = dict(vehiculo=vehiculo, catapulta=catapulta, rotor=rotor)
