"""Hoja de contacto de un GLB (animado o no) con bpy Workbench. Uso: render_sheet.py in.glb out.png [n_cuadros=4] [azim_deg=0] [size=250]"""
import sys, bpy, math, numpy as np
from mathutils import Vector
glb, out = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
azim = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
size = int(sys.argv[5]) if len(sys.argv) > 5 else 250
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
sc = bpy.context.scene
meshes = [o for o in sc.objects if o.type == 'MESH']
# bbox del primer cuadro (reposo)
sc.frame_set(sc.frame_start)
dg = bpy.context.evaluated_depsgraph_get()
pts = []
for o in meshes:
    oe = o.evaluated_get(dg); me = oe.to_mesh()
    pts += [oe.matrix_world @ v.co for v in me.vertices]
pts = np.array([[p.x, p.y, p.z] for p in pts]); lo, hi = np.percentile(pts, 0.5, axis=0), np.percentile(pts, 99.5, axis=0); c = (lo+hi)/2; ext = (hi-lo).max()
cam_d = bpy.data.cameras.new('c'); cam = bpy.data.objects.new('c', cam_d); sc.collection.objects.link(cam); sc.camera = cam
cam_d.type = 'ORTHO'; cam_d.ortho_scale = ext*1.9
a = math.radians(azim); dist = ext*4
cam.location = Vector((c[0]+dist*math.sin(a), c[1]-dist*math.cos(a), c[2]+ext*0.05))
d = Vector(c) - cam.location; cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
sc.render.engine = 'BLENDER_WORKBENCH'
sh = sc.display.shading; sh.light = 'STUDIO'; sh.color_type = 'TEXTURE'; sh.show_shadows = False
sc.render.resolution_x = sc.render.resolution_y = size
sc.render.film_transparent = False
sc.world = bpy.data.worlds.new('w'); sc.world.color = (0.55, 0.6, 0.65)
f0, f1 = sc.frame_start, sc.frame_end
if bpy.data.actions:
    r = bpy.data.actions[0].frame_range; f0, f1 = int(r[0]), int(r[1]); print('rango accion', f0, f1, 'fps', sc.render.fps)
frames = [f0] if (f1 == f0 or n == 1) else [int(round(f0 + (f1-f0)*i/(n-1))) for i in range(n)]
from PIL import Image
ims = []
for k, f in enumerate(frames):
    sc.frame_set(f); sc.render.filepath = f'/tmp/_sheet_{k}.png'; bpy.ops.render.render(write_still=True)
    ims.append(Image.open(f'/tmp/_sheet_{k}.png').convert('RGB'))
sheet = Image.new('RGB', (size*len(ims), size)); [sheet.paste(im, (i*size, 0)) for i, im in enumerate(ims)]
sheet.save(out); print('frames', frames, '->', out)
