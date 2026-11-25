"""hoja_T.py -- hoja de contacto de un modelo en pose T alineado con el original: modelo solo (frente, lado), superposicion con el original
(silueta roja translucida) y articulaciones marcadas (amarillo = nuevo, rojo = original), y primeros planos de hombros y axilas.

    blender -b -P hoja_T.py -- --nuevo modelo_T.glb --original original.glb --art-nuevo articulaciones_nuevo_T.json
        --art-original articulaciones_original.json --alineacion alineacion.json --salida hoja.jpg [--panel 333]

Cada panel se renderiza a doble resolucion con Workbench (textura de color, fondo transparente) y se baja 2x. Orden: fila 1 = nuevo frente,
nuevo lado, superposicion frente; fila 2 = superposicion lado, primer plano hombro izquierdo (3/4 desde abajo), hombro derecho (frente,
desde abajo). Imprime 'HOJA ruta'."""
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
    return [o for o in bpy.data.objects if o not in antes]


def main():
    panel = int(arg("--panel", "333"))
    R = panel * 2
    bpy.ops.wm.read_factory_settings(use_empty=True)
    nuevo = importar(arg("--nuevo"))
    orig = importar(arg("--original"))
    al = json.load(open(arg("--alineacion")))
    jn = json.load(open(arg("--art-nuevo")))["articulaciones"]
    jo = json.load(open(arg("--art-original"))) ["articulaciones"]
    s, t = al["escala"], al["traslacion_m"]
    mallas_n = [o for o in nuevo if o.type == "MESH"]
    mallas_o = [o for o in orig if o.type == "MESH" and len(o.data.vertices) > 100]
    for o in [o for o in orig if o.type == "MESH" and o not in mallas_o]:
        bpy.data.objects.remove(o, do_unlink=True)
    rojo = bpy.data.materials.new("rojo")
    rojo.diffuse_color = (0.85, 0.1, 0.1, 1)
    for o in mallas_o:
        o.data.materials.clear()
        o.data.materials.append(rojo)
    # esferas de articulaciones
    esf = []

    def esfera(p, color, r=0.014):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=p, segments=12, ring_count=8)
        e = bpy.context.active_object
        m = bpy.data.materials.new("c")
        m.diffuse_color = color
        e.data.materials.append(m)
        esf.append(e)
    for k, p in jn.items():
        esfera([s * p[0] + t[0], s * p[1] + t[1], s * p[2] + t[2]], (1.0, 0.85, 0.0, 1))
    for k, p in jo.items():
        esfera(p, (1.0, 0.0, 0.0, 1), 0.01)
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

    def aplica_alineacion(on):
        for o in mallas_n:
            o.scale = (s, s, s) if on else (1, 1, 1)
            o.location = tuple(t) if on else (0, 0, 0)
        bpy.context.view_layer.update()

    def capa(grupo, centro, ortho, az, el):
        for o in bpy.data.objects:
            if o.type in ("MESH", "ARMATURE"):
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

    pan = []
    cc = (0, 0, 0.95)
    aplica_alineacion(False)
    pan.append(baja(componer([(capa(mallas_n, cc, 2.15, 0, 0), 1.0)])))
    pan.append(baja(componer([(capa(mallas_n, cc, 2.15, 90, 0), 1.0)])))
    aplica_alineacion(True)
    for az in (0, 90):
        px_n = capa(mallas_n, cc, 2.15, az, 0)
        px_o = capa(mallas_o, cc, 2.15, az, 0)
        px_j = capa(esf, cc, 2.15, az, 0)
        pan.append(baja(componer([(px_n, 1.0), (px_o, 0.45), (px_j, 1.0)])))
    pan = [pan[0], pan[1], pan[2], pan[3]]
    cd_foco = (0.22, 0.03, 1.3)
    aplica_alineacion(False)
    pan.append(baja(componer([(capa(mallas_n, cd_foco, 0.6, 40, -20), 1.0)])))
    pan.append(baja(componer([(capa(mallas_n, (-0.22, 0.03, 1.3), 0.6, 0, -20), 1.0)])))
    filas = [np.hstack(pan[i:i + 3]) for i in (0, 3)]
    big = np.vstack(filas[::-1])  # bpy guarda de abajo arriba
    H, W = big.shape[:2]
    rgba = np.concatenate([big, np.ones((H, W, 1), np.float32)], 2)
    img = bpy.data.images.new("hoja", W, H)
    img.pixels = rgba.ravel()
    img.file_format = "JPEG"
    img.filepath_raw = arg("--salida")
    img.save()
    print("HOJA", arg("--salida"), W, H)


if __name__ == "__main__":
    main()
