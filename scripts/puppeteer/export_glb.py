"""Exporta GLB riggeado (y animado) con bpy desde: glb original (malla+UV+textura), rig.txt de Puppeteer y,
opcionalmente, la animacion optimizada (npz con local_quats, root_quats, root_pos).
Uso: python export_glb.py --src in.glb --rig rig.txt --out out.glb [--anim anim.npz --fps 10]
Las coordenadas se conservan tal cual (export_yup=False), asi que el GLB sale en el mismo marco que el de entrada."""
import sys, argparse, numpy as np, trimesh, bpy, os, tempfile
from mathutils import Vector, Matrix

ap = argparse.ArgumentParser()
ap.add_argument('--src'); ap.add_argument('--rig'); ap.add_argument('--out')
ap.add_argument('--anim', default=None); ap.add_argument('--fps', type=float, default=10)
ap.add_argument('--maxinf', type=int, default=4)
a = ap.parse_args()

# ---------- malla original (mismo orden de vertices que el obj usado para el skinning)
sc = trimesh.load(a.src, process=False)
m = sc.dump()[0] if isinstance(sc, trimesh.Scene) else sc
V = np.asarray(m.vertices, dtype=np.float64); F = np.asarray(m.faces)
uv = np.asarray(m.visual.uv); img = m.visual.material.baseColorTexture if hasattr(m.visual.material, 'baseColorTexture') else m.visual.material.image
print('malla', V.shape, F.shape, 'img', None if img is None else img.size)

# ---------- rig
names, jpos, hier, skin, root = [], [], [], {}, None
for line in open(a.rig):
    p = line.split()
    if not p: continue
    if p[0] == 'joints': names.append(p[1]); jpos.append([float(x) for x in p[2:5]])
    elif p[0] == 'root': root = p[1]
    elif p[0] == 'hier': hier.append((p[1], p[2]))
    elif p[0] == 'skin': skin[int(p[1])] = [(p[i], float(p[i+1])) for i in range(2, len(p)-1, 2)]
idx = {n: i for i, n in enumerate(names)}
J = len(names); jpos = np.array(jpos)
parent = [-1]*J
for p_, c_ in hier: parent[idx[c_]] = idx[p_]
children = [[] for _ in range(J)]
for c, p_ in enumerate(parent):
    if p_ >= 0: children[p_].append(c)
rooti = idx[root]
print('joints', J, 'root', root, 'vertices con skin', len(skin))

# ---------- escena
bpy.ops.wm.read_factory_settings(use_empty=True)
mesh = bpy.data.meshes.new('mesh')
mesh.from_pydata(V.tolist(), [], F.tolist())
mesh.update()
uvl = mesh.uv_layers.new(name='UVMap')
loops_v = np.empty(len(mesh.loops), dtype=np.int64); mesh.loops.foreach_get('vertex_index', loops_v)
uvl.data.foreach_set('uv', uv[loops_v].reshape(-1).astype(np.float32))
obj = bpy.data.objects.new('Malla', mesh); bpy.context.collection.objects.link(obj)
# material con textura
tmp = tempfile.mkdtemp(); tp = os.path.join(tmp, 'tex.png'); img.convert('RGBA').save(tp)
bimg = bpy.data.images.load(tp); bimg.pack()
mat = bpy.data.materials.new('mat'); mat.use_nodes = True
nt = mat.node_tree; bsdf = nt.nodes['Principled BSDF']
tn = nt.nodes.new('ShaderNodeTexImage'); tn.image = bimg
nt.links.new(tn.outputs['Color'], bsdf.inputs['Base Color'])
mesh.materials.append(mat)
for p_ in mesh.polygons: p_.use_smooth = True

# armadura (en coordenadas del GLB)
bbox = V.max(0) - V.min(0); minlen = 0.02*bbox.max()
arm_d = bpy.data.armatures.new('Armature'); arm = bpy.data.objects.new('Armature', arm_d)
bpy.context.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm; arm.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
for j in range(J):
    b = arm_d.edit_bones.new(names[j]); head = Vector(jpos[j])
    if children[j]:
        tail = Vector(np.mean([jpos[c] for c in children[j]], axis=0))
    elif parent[j] >= 0:
        d = jpos[j]-jpos[parent[j]]; n = np.linalg.norm(d)
        tail = head + Vector(d/n*minlen*2) if n > 1e-9 else head + Vector((0, 0, minlen*2))
    else: tail = head + Vector((0, 0, minlen*2))
    if (tail-head).length < minlen: 
        dd = (tail-head); dd = dd.normalized() if dd.length > 1e-9 else Vector((0, 0, 1)); tail = head + dd*minlen
    b.head = head; b.tail = tail
for j in range(J):
    if parent[j] >= 0: arm_d.edit_bones[names[j]].parent = arm_d.edit_bones[names[parent[j]]]
bpy.ops.object.mode_set(mode='OBJECT')

# pesos (top-N por vertice, normalizados)
vgs = {n: obj.vertex_groups.new(name=n) for n in names}
nskin = 0
for v, lst in skin.items():
    lst = sorted(lst, key=lambda x: -x[1])[:a.maxinf]
    s = sum(w for _, w in lst)
    if s <= 0: continue
    nskin += 1
    for n, w in lst:
        if w/s > 1e-4: vgs[n].add([v], w/s, 'REPLACE')
print('vertices con pesos', nskin, '/', len(V))
mod = obj.modifiers.new('Armature', 'ARMATURE'); mod.object = arm
obj.parent = arm

# ---------- animacion
rest = np.array([np.array(arm_d.bones[n].matrix_local) for n in names])  # (J,4,4) armature space
def q2R(q):  # (w,x,y,z) -> 3x3
    q = q/np.linalg.norm(q, axis=-1, keepdims=True); w, x, y, z = np.moveaxis(q, -1, 0)
    R = np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), 2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), 2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1)
    return R.reshape(q.shape[:-1]+(3, 3))
if a.anim:
    d = np.load(a.anim)
    lq, rq, rp = d['local_quats'], d['root_quats'], d['root_pos']  # (T,J,4),(T,4),(T,3)  (marco normalizado)
    center, scale = d['center'], float(d['scale'])
    T = lq.shape[0]
    # jerarquia en marco normalizado (misma que Puppeteer): pos local en reposo
    jn = (jpos-center)/scale
    rl = np.zeros((J, 3))
    for j in range(J): rl[j] = jn[j] if parent[j] < 0 else jn[j]-jn[parent[j]]
    # orden topologico
    order = []; q_ = [rooti]
    while q_:
        c = q_.pop(0); order.append(c); q_ += children[c]
    A = np.eye(4); A[:3, :3] /= scale; A[:3, 3] = -center/scale
    Ai = np.linalg.inv(A)
    bind_n = np.tile(np.eye(4), (J, 1, 1)); bind_n[:, :3, 3] = jn
    arm.animation_data_create()
    scn = bpy.context.scene; scn.render.fps = int(a.fps); scn.frame_start = 1; scn.frame_end = T
    for n in names: arm.pose.bones[n].rotation_mode = 'QUATERNION'
    rest_local = [None]*J
    for j in range(J):
        rest_local[j] = rest[j] if parent[j] < 0 else np.linalg.inv(rest[parent[j]]) @ rest[j]
    for t in range(T):
        Rr = np.eye(4); Rr[:3, :3] = q2R(rq[t]); Rr[:3, 3] = rp[t]
        R = q2R(lq[t])  # (J,3,3)
        G = [None]*J
        for j in order:
            Tj = np.eye(4); Tj[:3, :3] = R[j]; Tj[:3, 3] = rl[j]
            G[j] = Tj if parent[j] < 0 else G[parent[j]] @ Tj   # matriz global (rot global acumulada + pos)
        # reconstruir como FK de Puppeteer: rot global = prod rots, pos = rot_parent*local_pos + pos_parent
        # (G arriba es equivalente: G_c = G_p @ [R_c | rl_c])
        pose_arm = [None]*J
        for j in order:
            M_n = Rr @ G[j] @ np.linalg.inv(bind_n[j])           # deformacion en marco normalizado
            M = Ai @ M_n @ A                                      # a marco original del GLB
            pose_arm[j] = M @ rest[j]
            pm = arm.pose.bones[names[j]]
            if parent[j] < 0: basis = np.linalg.inv(rest_local[j]) @ pose_arm[j]
            else: basis = np.linalg.inv(rest_local[j]) @ np.linalg.inv(pose_arm[parent[j]]) @ pose_arm[j]
            pm.matrix_basis = Matrix(basis.tolist())
            pm.keyframe_insert('location', frame=t+1); pm.keyframe_insert('rotation_quaternion', frame=t+1)
    print('animacion', T, 'cuadros')

bpy.ops.object.select_all(action='DESELECT')
obj.select_set(True); arm.select_set(True)
bpy.ops.export_scene.gltf(filepath=a.out, export_format='GLB', use_selection=True, export_yup=False,
                          export_animations=bool(a.anim), export_skins=True, export_image_format='AUTO',
                          export_apply=False, export_force_sampling=True)
print('GLB ->', a.out, os.path.getsize(a.out))
