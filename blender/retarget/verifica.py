"""verifica.py -- reimporta un .glb animado, comprueba animaciones y textura y renderiza una hoja de contacto.

    blender -b -P verifica.py -- modelo.glb hoja.png [--clips Walk_Loop,Sword_Attack] [--n 4] [--celda 250x330] [--esperadas 5]
Imprime 'VERIF {...}' (n de animaciones, nombres, imagenes/texturas, malla) y guarda la hoja (4 columnas x filas = clips, <=1000 px de ancho).
Render con Workbench (color = textura), camara 3/4 ortografica. Los cuadros se eligen repartidos a lo largo del clip (sin repetir el ultimo = primero).
"""
import json
import math
import os
import sys
import tempfile

import bpy
import numpy as np
from mathutils import Vector

a = sys.argv[sys.argv.index("--") + 1:]


def arg(n, d=None):
    return a[a.index(n) + 1] if n in a else d


glb, hoja = a[0], a[1]
clips = arg("--clips", "Walk_Loop,Sword_Attack").split(",")
N = int(arg("--n", "4"))
cw, ch = (int(x) for x in arg("--celda", "250x330").split("x"))

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
mallas = [o for o in bpy.data.objects if o.type == "MESH"]
rep = {"animaciones": len(bpy.data.actions), "nombres": sorted(x.name for x in bpy.data.actions),
       "huesos": len(arm.data.bones), "mallas": [(m.name, len(m.data.vertices), len(m.vertex_groups)) for m in mallas]}
tex = []
for m in mallas:
    for mat in m.data.materials:
        if mat and mat.node_tree:
            for n in mat.node_tree.nodes:
                if n.type == "TEX_IMAGE" and n.image:
                    lig = any(l.from_node == n for l in mat.node_tree.links)
                    tex.append({"material": mat.name, "imagen": n.image.name, "tam": list(n.image.size), "conectada": lig})
rep["texturas"] = tex
rep["imagenes"] = [(i.name, list(i.size)) for i in bpy.data.images if i.name != "Render Result"]
for ac in bpy.data.actions:
    rep.setdefault("rango", {})[ac.name] = [round(ac.frame_range[0], 2), round(ac.frame_range[1], 2)]
print("VERIF " + json.dumps(rep))

sc = bpy.context.scene
sc.render.engine = "BLENDER_WORKBENCH"
sh = sc.display.shading
sh.light = "STUDIO"
sh.color_type = "TEXTURE"
sh.show_shadows = False
sh.background_type = "WORLD"
sc.display.render_aa = "8"
sc.render.resolution_x, sc.render.resolution_y = cw, ch
sc.render.film_transparent = False
sc.world = bpy.data.worlds.new("w")
sc.world.color = (0.55, 0.62, 0.7)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
bpy.context.collection.objects.link(cam)
cam.data.type = "ORTHO"
cam.data.ortho_scale = 1.4
sc.camera = cam
# camara 3/4 frontal: el personaje mira a -Y => camara en (-Y, +X) un poco elevada
dist = 6.0
az = math.radians(35)
cam.location = (dist * math.sin(az), -dist * math.cos(az), 0.5 + 1.5)
tgt = Vector((0, 0, 0.5))
d = tgt - Vector(cam.location)
cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
cam.data.clip_end = 50
filas = []
animadas = {x.name: x for x in bpy.data.actions}
for cn in clips:
    ac = animadas[cn]
    arm.animation_data_create()
    arm.animation_data.action = ac
    if ac.slots:
        arm.animation_data.action_slot = ac.slots[0]
    f0, f1 = ac.frame_range
    fr = [f0 + (f1 - f0) * i / N for i in range(N)]  # no incluye el ultimo (=primero en bucles)
    fila = []
    for f in fr:
        sc.frame_set(int(math.floor(f)), subframe=f - math.floor(f))
        # centrar horizontalmente en la cadera para que quepa
        hp = arm.matrix_world @ arm.pose.bones[0].head
        sc.render.filepath = os.path.join(tempfile.gettempdir(), "_assetgen_cel.png")
        bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(sc.render.filepath)
        px = np.zeros(im.size[0] * im.size[1] * 4, dtype=np.float32)
        im.pixels.foreach_get(px)
        fila.append(px.reshape(im.size[1], im.size[0], 4))
        bpy.data.images.remove(im)
    filas.append(np.concatenate(fila, axis=1))
H = np.concatenate(filas[::-1], axis=0)  # los pixels de Blender van de abajo a arriba: el primer clip arriba
img = bpy.data.images.new("hoja", H.shape[1], H.shape[0], alpha=True)
img.pixels.foreach_set(H.ravel())
img.filepath_raw = hoja
img.file_format = "PNG"
img.save()
print("HOJA", hoja, H.shape[1], "x", H.shape[0])
