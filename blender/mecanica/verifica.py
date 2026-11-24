"""Reimporta un glb animado: clips, texturas, pieles; renderiza hoja de 6 cuadros del clip principal (<=1000 px).
blender -b -P verifica.py -- <glb> <clip_principal> <hoja.png> [camara: x,y,z,tx,ty,tz]"""
import bpy, sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import *
a = sys.argv[sys.argv.index('--') + 1:]
glb, clip, hoja = a[0], a[1], a[2]
cam = [float(x) for x in a[3].split(',')] if len(a) > 3 else [0.2, -2.1, 0.55, 0.02, 0, 0.17]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
print("OBJS", [(o.name, o.type) for o in bpy.data.objects])
print("SKINS", [(o.name, [m.type for m in o.modifiers], len(o.vertex_groups)) for o in bpy.data.objects if o.type == "MESH" and o.modifiers])
print("ACTIONS", [(x.name, [round(v) for v in x.frame_range]) for x in bpy.data.actions])
print("IMAGES", [(i.name, tuple(i.size)) for i in bpy.data.images])
for m in bpy.data.materials:
    print("MAT", m.name, [n.image.name for n in m.node_tree.nodes if n.type == 'TEX_IMAGE' and n.image])
tiene_tex = any(i.size[0] > 0 for i in bpy.data.images)
print("TEXTURA_OK", tiene_tex)
sc = bpy.context.scene
rng = None
for o in bpy.data.objects:
    ad = o.animation_data
    if not ad: continue
    tr = [(t.name, [s.action.name for s in t.strips]) for t in ad.nla_tracks]
    print("ANIM", o.name, "action", ad.action.name if ad.action else None, "tracks", tr)
    for t in ad.nla_tracks:
        if clip not in t.name: t.mute = True
        else:
            for s in t.strips:
                rng = (s.frame_start, s.frame_end) if rng is None else (min(rng[0], s.frame_start), max(rng[1], s.frame_end))
    if ad.action and clip not in ad.action.name and ad.nla_tracks: pass
print("rango clip", rng)
if rng is None: rng = (0, 24)
setup_render(330, 'BLENDER_EEVEE')
sc.render.engine = 'BLENDER_EEVEE'
sc.world = bpy.data.worlds.new("W2"); sc.world.use_nodes = True
bg = sc.world.node_tree.nodes['Background']; bg.inputs[0].default_value = (0.75, 0.78, 0.85, 1); bg.inputs[1].default_value = 1.0
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", 'SUN')); sun.data.energy = 3.5
sun.rotation_euler = (math.radians(50), math.radians(10), math.radians(-30)); sc.collection.objects.link(sun)
sc.view_settings.view_transform = 'Standard'
make_cam(tuple(cam[:3]), tuple(cam[3:]), lens=50)
tiles = []
fr_list = np.linspace(rng[0], rng[1] * (0.95 if 'rodar' in clip or 'girar' in clip else 1.0), 6)
if 'disparar' in clip:   # cuadros representativos: reposo, carga, salto, tope, rebote, regreso
    fr_list = [rng[0] + f for f in (0, 10, 15, 18, 24, 40)]
for fr in fr_list:
    sc.frame_set(int(fr), subframe=fr - int(fr))
    tiles.append(render_to_array(os.path.join(__import__("tempfile").gettempdir(), "mec_t.png")))
    for o in bpy.data.objects:
        if o.type == 'ARMATURE' and 'hueso_brazo' in o.pose.bones:
            pb = o.pose.bones['hueso_brazo']; print("frame", round(fr, 1), "hueso_brazo rot deg", [round(math.degrees(x), 1) for x in pb.rotation_quaternion.to_euler()])
save_sheet(tiles, 3, hoja)
print("HOJA", hoja)
