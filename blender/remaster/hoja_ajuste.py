"""hoja_ajuste.py -- hoja de contacto del ajuste de proporciones: el modelo antes (alineado con escala y traslacion globales) y despues,
de frente y de lado, con la silueta del original encima (rojo translucido) y las referencias marcadas (amarillo = nuevo, rojo = original),
y primeros planos del hombro y del fin de la manga.

    blender -b -P hoja_ajuste.py -- --antes modelo_T.glb --despues modelo_T_ajustado.glb --original original.glb --referencias referencias.json
        --alineacion alineacion.json --salida hoja.jpg [--panel 333]

Orden: fila 1 = antes frente, despues frente, primer plano del hombro izquierdo (despues, de frente); fila 2 = antes lado, despues lado,
primer plano del fin de manga (despues, de frente). Imprime 'HOJA ruta'."""
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


def importar(ruta):
    antes = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=ruta)
    return [o for o in bpy.data.objects if o not in antes and o.type == "MESH" and len(o.data.vertices) > 100]


def main():
    panel = int(arg("--panel", "333"))
    R = panel * 2
    bpy.ops.wm.read_factory_settings(use_empty=True)
    m_antes = importar(arg("--antes"))
    m_desp = importar(arg("--despues"))
    m_orig = importar(arg("--original"))
    al = json.load(open(arg("--alineacion")))
    s, t = al["escala"], al["traslacion_m"]
    ref = json.load(open(arg("--referencias")))["referencias"]
    rojo = bpy.data.materials.new("rojo")
    rojo.diffuse_color = (0.85, 0.1, 0.1, 1)
    for o in m_orig:
        o.data.materials.clear()
        o.data.materials.append(rojo)

    def esferas(puntos, color, r):
        out = []
        for p in puntos:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=p, segments=10, ring_count=6)
            e = bpy.context.active_object
            m = bpy.data.materials.new("c")
            m.diffuse_color = color
            e.data.materials.append(m)
            out.append(e)
        return out
    pts = {k: v for k, v in ref.items() if isinstance(v["original"], list) and v["original"][0] != 0.0}
    e_orig = esferas([v["original"] for v in pts.values()], (1.0, 0.0, 0.0, 1), 0.011)
    e_antes = esferas([[s * v["nuevo_antes"][0] + t[0], s * v["nuevo_antes"][1] + t[1], s * v["nuevo_antes"][2] + t[2]] for v in pts.values()],
                      (1.0, 0.85, 0.0, 1), 0.014)
    e_desp = esferas([v["nuevo_despues"] for v in pts.values()], (1.0, 0.85, 0.0, 1), 0.014)
    for o in m_antes:
        o.scale, o.location = (s, s, s), tuple(t)
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.render.resolution_x = sc.render.resolution_y = R
    sh = sc.display.shading
    sh.light, sh.color_type, sh.show_cavity = "FLAT", "TEXTURE", False
    sc.render.film_transparent = True
    sc.render.image_settings.file_format, sc.render.image_settings.color_mode = "PNG", "RGBA"
    sc.view_settings.view_transform = "Standard"
    cd = bpy.data.cameras.new("c")
    cd.type = "ORTHO"
    cam = bpy.data.objects.new("c", cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    tmp = tempfile.mkdtemp()

    def capa(grupo, centro, ortho, az, el=0):
        for o in bpy.data.objects:
            if o.type == "MESH":
                o.hide_render = True
        for o in grupo:
            o.hide_render = False
        cd.ortho_scale = ortho
        A, E = math.radians(az), math.radians(el)
        d = Vector((math.sin(A) * math.cos(E), -math.cos(A) * math.cos(E), math.sin(E)))
        cam.location = Vector(centro) + d * 10
        cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
        ruta = os.path.join(tmp, "p.png")
        sc.render.filepath = ruta
        bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(ruta)
        px = np.array(im.pixels[:], np.float32).reshape(R, R, 4)
        bpy.data.images.remove(im)
        return px

    def componer(capas):
        out = np.zeros((R, R, 3), np.float32) + 0.62
        for px, alfa in capas:
            m = px[..., 3:4] * alfa
            out = out * (1 - m) + px[..., :3] * m
        return out

    def baja(x):
        return x.reshape(panel, 2, panel, 2, 3).mean((1, 3))

    def vista(modelo, marcas, centro, ortho, az):
        return baja(componer([(capa(modelo, centro, ortho, az), 1.0), (capa(m_orig, centro, ortho, az), 0.45), (capa(marcas + e_orig, centro, ortho, az), 1.0)]))
    cc = (0, 0, 0.95)
    pan = [vista(m_antes, e_antes, cc, 2.15, 0), vista(m_desp, e_desp, cc, 2.15, 0),
           vista(m_desp, e_desp, (0.30, 0.0, 1.50), 0.55, 0),
           vista(m_antes, e_antes, cc, 2.15, 90), vista(m_desp, e_desp, cc, 2.15, 90),
           vista(m_desp, e_desp, (0.60, 0.0, 1.50), 0.55, 0)]
    big = np.vstack([np.hstack(pan[i:i + 3]) for i in (0, 3)][::-1])
    H, W = big.shape[:2]
    img = bpy.data.images.new("hoja", W, H)
    img.pixels = np.concatenate([big, np.ones((H, W, 1), np.float32)], 2).ravel()
    img.file_format = "JPEG"
    img.filepath_raw = arg("--salida")
    img.save()
    print("HOJA", arg("--salida"), W, H)


if __name__ == "__main__":
    main()
