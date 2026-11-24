"""renderizar.py -- renders de referencia de un original en reposo: frente, lado (izquierdo del personaje) y espalda.

    blender -b -P renderizar.py -- --salida DIR (--entrada modelo.glb|.gltf|.fbx | --geom geom.json) [--frente x,y,z] [--tam 1024] [--textura img.png] [--hoja vistas.jpg]

Camara ortografica con la MISMA escala en las tres vistas, fondo gris neutro, Workbench (color = textura si hay), sin sombras.
`--frente` es un vector horizontal en el espacio canonico del archivo (Y arriba; glTF/fbx importados), por defecto +z.
`--geom` es el JSON que escribe el adaptador 3dmigoto (piezas con pos/nrm/uv/tris ya en espacio canonico Y arriba, +z frente).
Escribe frente.png, lado.png, espalda.png y render.json (escala, altura y centro usados). Imprime 'RENDER {...}'."""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

a = sys.argv[sys.argv.index("--") + 1:]


def arg(n, d=None):
    return a[a.index(n) + 1] if n in a else d


salida = arg("--salida")
entrada, geom_json = arg("--entrada"), arg("--geom")
tam = int(arg("--tam", "1024"))
textura = arg("--textura")
frente_c = [float(x) for x in arg("--frente", "0,0,1").split(",")]
if not salida or not (entrada or geom_json):
    sys.exit("uso: --salida DIR (--entrada archivo | --geom json)")
os.makedirs(salida, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)


def canon_a_blender(v):          # (x, y, z) canonico (Y arriba, +z frente) -> Blender (Z arriba, frente -Y)
    return (v[0], -v[2], v[1])


tiene_textura = False
if geom_json:
    g = json.load(open(geom_json))
    mat = None
    if textura and os.path.exists(textura):
        img = bpy.data.images.load(textura)
        mat = bpy.data.materials.new("original")
        mat.use_nodes = True
        nt = mat.node_tree
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        nt.links.new(tex.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
        tiene_textura = True
    for p in g["piezas"]:
        pos = np.array(p["pos"], dtype=np.float64)
        pb = np.stack([pos[:, 0], -pos[:, 2], pos[:, 1]], axis=1)
        me = bpy.data.meshes.new(p["nombre"])
        me.from_pydata(pb.tolist(), [], [tuple(t) for t in p["tris"]])
        me.update()
        if p.get("uv"):
            uv = np.array(p["uv"], dtype=np.float64)
            uv[:, 1] = 1.0 - uv[:, 1]
            capa = me.uv_layers.new(name="UV")
            vi = np.zeros(len(me.loops), dtype=np.int32)
            me.loops.foreach_get("vertex_index", vi)
            capa.data.foreach_set("uv", uv[vi].reshape(-1))
        if p.get("nrm") and p["nrm"][0] is not None:
            nr = np.array(p["nrm"], dtype=np.float64)
            nb = np.stack([nr[:, 0], -nr[:, 2], nr[:, 1]], axis=1)
            ln = np.linalg.norm(nb, axis=1, keepdims=True)
            nb = nb / np.where(ln > 1e-9, ln, 1)
            try:
                me.normals_split_custom_set_from_vertices(nb.tolist())
            except Exception:
                pass
        ob = bpy.data.objects.new(p["nombre"], me)
        bpy.context.scene.collection.objects.link(ob)
        if mat:
            me.materials.append(mat)
else:
    ext = os.path.splitext(entrada)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=entrada)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=entrada)
    else:
        sys.exit("formato no soportado: " + ext)
    for m in bpy.data.materials:
        if m.use_nodes and any(n.type == "TEX_IMAGE" and n.image for n in m.node_tree.nodes):
            tiene_textura = True

mallas = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not mallas:
    sys.exit("RENDER_ERROR sin mallas")
for o in bpy.context.scene.objects:
    if o.type not in ("MESH", "CAMERA", "LIGHT"):
        o.hide_render = True
bpy.context.view_layer.update()
mn = np.array([1e18] * 3)
mx = np.array([-1e18] * 3)
for o in mallas:
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c)
        mn = np.minimum(mn, w[:])
        mx = np.maximum(mx, w[:])
centro = (mn + mx) / 2
altura = float(mx[2] - mn[2])
horiz = float(max(mx[0] - mn[0], mx[1] - mn[1]))
escala = 1.12 * max(altura, horiz)

sc = bpy.context.scene
sc.render.engine = "BLENDER_WORKBENCH"
sh = sc.display.shading
sh.light = "STUDIO"
sh.color_type = "TEXTURE" if tiene_textura else "SINGLE"
sh.single_color = (0.78, 0.78, 0.78)
sh.show_shadows = False
sh.show_cavity = False
sh.show_object_outline = False
sc.display.render_aa = "8"
sc.render.resolution_x = sc.render.resolution_y = tam
sc.render.resolution_percentage = 100
sc.render.film_transparent = False
sc.view_settings.view_transform = "Standard"
sc.view_settings.look = "None"
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGB"
mundo = bpy.data.worlds.new("fondo")
mundo.color = (0.5, 0.5, 0.5)
sc.world = mundo

cam_d = bpy.data.cameras.new("cam")
cam_d.type = "ORTHO"
cam_d.ortho_scale = escala
cam_d.clip_start, cam_d.clip_end = 0.01, 10000
cam = bpy.data.objects.new("cam", cam_d)
sc.collection.objects.link(cam)
sc.camera = cam

fb = Vector(canon_a_blender(frente_c))
fb.z = 0
fb.normalize()
arriba = Vector((0, 0, 1))
izq = arriba.cross(fb).normalized()          # izquierda del personaje
dist = max(altura, horiz) * 3 + 5
vistas = {"frente": fb, "lado": izq, "espalda": -fb}
for nombre, dire in vistas.items():
    cam.location = Vector(centro) + dire * dist
    cam.rotation_euler = (Vector(centro) - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = os.path.join(salida, nombre + ".png")
    bpy.ops.render.render(write_still=True)

# ---- hoja de contacto sin Pillow: las 3 vistas reducidas, una al lado de otra (JPEG <= 1000 px de ancho)
hoja = arg("--hoja")
if hoja:
    celda = 333
    tira = np.zeros((celda, celda * len(vistas), 4), dtype=np.float32)
    for k, nombre in enumerate(vistas):
        im = bpy.data.images.load(os.path.join(salida, nombre + ".png"))
        im.scale(celda, celda)
        px = np.empty(celda * celda * 4, dtype=np.float32)
        im.pixels.foreach_get(px)
        tira[:, k * celda:(k + 1) * celda, :] = px.reshape(celda, celda, 4)
    out = bpy.data.images.new("hoja", celda * len(vistas), celda, alpha=False)
    out.pixels.foreach_set(tira.reshape(-1))
    out.file_format = "JPEG"
    out.filepath_raw = hoja
    out.save()

info = {"escala_ortografica": escala, "altura": altura, "centro_blender": [float(v) for v in centro], "textura": tiene_textura,
        "vistas": list(vistas), "tam": tam}
json.dump(info, open(os.path.join(salida, "render.json"), "w"), indent=1)
print("RENDER " + json.dumps(info))
