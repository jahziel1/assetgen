"""cara_render.py -- render ortografico de frente de una cabeza (para detectar landmarks 2D) de un glTF o de la base MakeHuman.

    blender -b -P cara_render.py -- (--glb cabeza.glb | --base [--cache DIR]) --salida img.png --centro x,y,z --escala m [--res 1024]

Camara ortografica mirando a +Y desde -Y (el frente en Blender), centrada en `--centro` (metros), `--escala` = ancho visible en metros.
El glTF se renderiza con su textura (Workbench, luz plana). La base MakeHuman, en tono piel con sus ojos (la textura de ojos de la
cache). Escribe img.png e img.json con {centro, escala, res} (lo que hace falta para llevar pixeles 2D a rayos 3D)."""
import json
import math
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cara_comun as C

a = sys.argv[sys.argv.index("--") + 1:]


def arg(n, d=None):
    return a[a.index(n) + 1] if n in a else d


salida = arg("--salida")
res = int(arg("--res", "1024"))
centro = [float(x) for x in arg("--centro", "0,0,0").split(",")]
escala = float(arg("--escala", "0.5"))
if not salida or not (arg("--glb") or "--base" in a):
    sys.exit("uso: --glb f.glb | --base  --salida img.png --centro x,y,z --escala m")

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "BLENDER_WORKBENCH"
sc.render.resolution_x = sc.render.resolution_y = res
sh = sc.display.shading
sh.light = "FLAT"
sh.color_type = "TEXTURE"
sc.world = bpy.data.worlds.new("w")
sc.world.color = (0.5, 0.5, 0.5)

if arg("--glb"):
    bpy.ops.import_scene.gltf(filepath=arg("--glb"))
else:
    from assets_makehuman import asegurar
    rutas = asegurar(arg("--cache"))
    sh.light = "STUDIO"
    sh.color_type = "OBJECT"
    base = C.leer_obj(rutas["base.obj"])
    cuerpo = [c for c, g in zip(base["caras"], base["grupo"]) if g == "body"]
    me = bpy.data.meshes.new("base")
    me.from_pydata(C.a_blender(base["v"]).tolist(), [], cuerpo)
    me.update()
    ob = bpy.data.objects.new("base", me)
    ob.color = (0.75, 0.52, 0.40, 1)
    sc.collection.objects.link(ob)
    # ojos: proxy mhclo sobre la base, con su textura
    ov = C.leer_obj(rutas["eyes_high-poly.obj"])
    pe = C.a_blender(C.ojos_sobre_base(C.leer_mhclo(rutas["eyes_high-poly.mhclo"]), base["v"]))
    me2 = bpy.data.meshes.new("ojos")
    me2.from_pydata(pe.tolist(), [], ov["caras"])
    me2.update()
    capa = me2.uv_layers.new(name="UV")
    li = 0
    for cv in ov["caras_vt"]:
        for k in cv:
            u = ov["vt"][k]
            capa.data[li].uv = (u[0], u[1])
            li += 1
    img = bpy.data.images.load(rutas["eyes_brown_eye.png"])
    mat = bpy.data.materials.new("ojo")
    mat.use_nodes = True
    t = mat.node_tree.nodes.new("ShaderNodeTexImage")
    t.image = img
    mat.node_tree.links.new(t.outputs["Color"], mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
    me2.materials.append(mat)
    ob2 = bpy.data.objects.new("ojos", me2)
    sc.collection.objects.link(ob2)
    sh.color_type = "MATERIAL"
    ob.data.materials.append(bpy.data.materials.new("piel"))
    ob.data.materials[0].diffuse_color = (0.75, 0.52, 0.40, 1)

cam = bpy.data.objects.new("c", bpy.data.cameras.new("c"))
sc.collection.objects.link(cam)
sc.camera = cam
cam.data.type = "ORTHO"
cam.data.ortho_scale = escala
cam.location = (centro[0], centro[1] - 3.0, centro[2])
cam.rotation_euler = (math.pi / 2, 0, 0)
sc.render.filepath = salida
bpy.ops.render.render(write_still=True)
json.dump(dict(centro=centro, escala=escala, res=res), open(os.path.splitext(salida)[0] + ".json", "w"))
print("RENDER", salida)
