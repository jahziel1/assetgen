"""hoja_modelo.py -- hoja de contacto de un modelo con textura: cuerpo entero (frente, 3/4, lado, espalda) y primeros planos.

    blender -b -P hoja_modelo.py -- --entrada m.glb|m.blend --salida hoja.jpg [--ancho 1000] [--foco x,y,z --foco-ancho 0.35]
                                    [--frente-menos-y]

Convencion de Blender (z arriba, el personaje mira a -Y). La camara es ortografica, Workbench con la textura de color, fondo gris,
sin sombras. Fila 1: cuerpo entero (misma escala). Fila 2 (si hay --foco): primeros planos centrados en el punto `foco` (frente, 3/4
derecha, lado, 3/4 izquierda) con ancho de campo `--foco-ancho` en metros. Imprime 'HOJA ruta'."""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

a = sys.argv[sys.argv.index("--") + 1:]


def arg(n, d=None):
    return a[a.index(n) + 1] if n in a else d


def render(path, centro, ortho, azim, elev, w, h):
    sc = bpy.context.scene
    cam = bpy.data.objects.get("cam_hoja")
    if cam is None:
        cd = bpy.data.cameras.new("cam_hoja")
        cd.type = "ORTHO"
        cam = bpy.data.objects.new("cam_hoja", cd)
        sc.collection.objects.link(cam)
    cam.data.ortho_scale = ortho
    cam.data.clip_end = 100
    az, el = math.radians(azim), math.radians(elev)
    d = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
    cam.location = Vector(centro) + d * 20
    cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.filepath = path
    bpy.ops.render.render(write_still=True)
    im = bpy.data.images.load(path)
    px = np.empty(w * h * 4, np.float32)
    im.pixels.foreach_get(px)
    bpy.data.images.remove(im)
    return px.reshape(h, w, 4)[..., :3]


def bajar(p):      # 2x2 -> 1 (antialias)
    return (p[0::2, 0::2] + p[1::2, 0::2] + p[0::2, 1::2] + p[1::2, 1::2]) / 4


def main():
    ent, sal = arg("--entrada"), arg("--salida")
    ancho = int(arg("--ancho", "1000"))
    if not ent or not sal:
        sys.exit("uso: --entrada m.glb|m.blend --salida hoja.jpg [--ancho 1000] [--foco x,y,z --foco-ancho m]")
    if ent.endswith(".blend"):
        bpy.ops.wm.open_mainfile(filepath=ent)
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=ent)
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light, sh.color_type, sh.show_cavity = "FLAT", "TEXTURE", False
    sc.render.image_settings.file_format = "PNG"
    sc.render.film_transparent = False
    sc.world = bpy.data.worlds.new("w_hoja")
    sc.world.color = (0.5, 0.5, 0.5)
    sc.view_settings.view_transform = "Standard"
    cos = []
    for o in bpy.data.objects:
        if o.type == "MESH":
            M = np.array(o.matrix_world)
            v = np.empty(len(o.data.vertices) * 3, np.float32)
            o.data.vertices.foreach_get("co", v)
            v = v.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
            cos.append(v)
    co = np.concatenate(cos)
    lo, hi = co.min(0), co.max(0)
    alto = float(hi[2] - lo[2])
    centro = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2]
    cw = ancho // 4                      # ancho de celda
    ch = int(cw * 1.6)                   # alto de celda de cuerpo entero
    tmp = os.path.join(os.path.dirname(os.path.abspath(sal)), "_hoja_tmp.png")
    celdas = [bajar(render(tmp, centro, alto * 1.06, az, 0, cw * 2, ch * 2)) for az in (0, 40, 90, 180)]
    fila1 = np.concatenate(celdas, 1)
    filas = [fila1]
    if arg("--foco"):
        foco = [float(x) for x in arg("--foco").split(",")]
        fa = float(arg("--foco-ancho", "0.35"))
        fch = cw
        pl = [bajar(render(tmp, foco, fa, az, el, cw * 2, fch * 2)) for az, el in ((0, 0), (35, 8), (90, 0), (-35, 8))]
        filas.append(np.concatenate(pl, 1))
    hoja = np.concatenate(filas[::-1], 0)         # los pixeles de Blender van de abajo arriba: la fila 1 va al final
    h, w = hoja.shape[:2]
    out = np.ones((h, w, 4), np.float32)
    out[..., :3] = hoja
    im = bpy.data.images.new("hoja", w, h, alpha=False)
    im.pixels.foreach_set(out.ravel())
    im.filepath_raw = sal
    im.file_format = "JPEG"
    sc.render.image_settings.quality = 88
    im.save()
    os.remove(tmp)
    print("HOJA", sal, w, h)


if __name__ == "__main__":
    main()
