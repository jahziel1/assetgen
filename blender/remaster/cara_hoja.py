"""cara_hoja.py -- hoja de contacto de cara_base.py (Workbench, gris con cavidades), importada desde cara_base.py.

Columnas: (a) cabeza generada, (b) cara base ajustada con ojos, (c) cara (naranja) con ojos encima de la generada (gris).
Filas: frente, 3/4 y primer plano de los ojos. Cada tile se renderiza a 2x y se reduce."""
import math
import os
import tempfile

import bpy
import numpy as np
from mathutils import Vector


def _mat(nombre, color):
    m = bpy.data.materials.new(nombre)
    m.diffuse_color = color
    return m


def hoja(sc, ob_gen, ob_cara, ojos, ctr, ancho, salida, centros_ojos, tam=336, ss=2):
    gris = _mat("hoja_gris", (0.72, 0.72, 0.72, 1))
    naranja = _mat("hoja_naranja", (0.95, 0.55, 0.30, 1))
    orig_gen = list(ob_gen.data.materials)
    orig_cara = list(ob_cara.data.materials)
    ob_gen.data.materials.clear()
    ob_gen.data.materials.append(gris)
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.render.resolution_x = sc.render.resolution_y = tam * ss
    sc.render.film_transparent = False
    sh = sc.display.shading
    sh.light = "STUDIO"
    sh.color_type = "TEXTURE"
    sh.show_cavity = True
    sh.cavity_type = "BOTH"
    sh.show_specular_highlight = False
    sc.world = bpy.data.worlds.new("hoja_w")
    sc.world.color = (0.45, 0.45, 0.45)
    cam = bpy.data.objects.new("hoja_cam", bpy.data.cameras.new("hoja_cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.type = "ORTHO"
    cam.data.clip_end = 100
    centro_ojos = (Vector(centros_ojos[0]) + Vector(centros_ojos[1])) / 2
    ctr = Vector(ctr)
    vistas = [("frente", 0, ctr, 1.5 * ancho), ("tres_cuartos", 35, ctr, 1.5 * ancho), ("ojos", 0, centro_ojos, 0.62 * ancho)]
    columnas = ["gen", "cara", "sobre"]
    tmp = tempfile.mkdtemp()
    filas = []
    for nom, az, tgt, esc in vistas:
        fila = []
        for col in columnas:
            ob_gen.hide_render = col == "cara"
            ob_cara.hide_render = col == "gen"
            for o in ojos:
                o.hide_render = col == "gen"
            ob_cara.data.materials.clear()
            ob_cara.data.materials.append(naranja if col == "sobre" else gris)
            cam.data.ortho_scale = esc
            t = math.radians(az)
            cam.location = tgt + Vector((math.sin(t) * 3, -math.cos(t) * 3, 0))
            cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
            ruta = os.path.join(tmp, "%s_%s.png" % (nom, col))
            sc.render.filepath = ruta
            bpy.ops.render.render(write_still=True)
            im = bpy.data.images.load(ruta)
            w, h = im.size
            px = np.array(im.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1, :, :3]
            px = px.reshape(tam, ss, tam, ss, 3).mean(axis=(1, 3))
            fila.append(px)
            bpy.data.images.remove(im)
        filas.append(np.hstack(fila))
    mosaico = np.vstack(filas)
    H, W = mosaico.shape[:2]
    out = np.ones((H, W, 4), dtype=np.float32)
    out[..., :3] = mosaico[::-1]
    im = bpy.data.images.new("hoja", W, H)
    im.pixels.foreach_set(out.reshape(-1))
    im.file_format = "JPEG"
    im.filepath_raw = salida
    im.save()
    bpy.data.images.remove(im)
    # restaurar
    ob_gen.hide_render = False
    ob_cara.hide_render = False
    for o in ojos:
        o.hide_render = False
    ob_gen.data.materials.clear()
    for m in orig_gen:
        ob_gen.data.materials.append(m)
    ob_cara.data.materials.clear()
    for m in orig_cara:
        ob_cara.data.materials.append(m)
    print("HOJA", salida, W, H)
