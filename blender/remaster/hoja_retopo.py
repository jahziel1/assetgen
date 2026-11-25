"""hoja_retopo.py -- hoja de contacto de la retopologia: malla alta vs malla del juego (con textura) de frente, espalda y 3/4, y alambre
de cara, hombro y mano de la malla del juego.

    blender -b -P hoja_retopo.py -- --trab DIR --juego juego.blend --salida hoja.jpg [--ancho 1000] [--alto-blend altas.blend]

Workbench (sin trazado de rayos): rapido y sin GPU. Las vistas usan camara ortografica con el encuadre de la caja de la malla."""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

a = sys.argv[sys.argv.index("--") + 1:]


def arg(n, d=None):
    return a[a.index(n) + 1] if n in a else d


def log(*m):
    print(*m, flush=True)


def caja(obs):
    pts = [o.matrix_world @ Vector(c) for o in obs for c in o.bound_box]
    mn = Vector([min(p[i] for p in pts) for i in range(3)]); mx = Vector([max(p[i] for p in pts) for i in range(3)])
    return mn, mx


def camara(centro, azim_deg, elev_deg, escala, dist=6.0):
    cam = bpy.data.objects.get("cam") or bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    if cam.name not in bpy.context.scene.objects:
        bpy.context.scene.collection.objects.link(cam)
    cam.data.type = "ORTHO"; cam.data.ortho_scale = escala; cam.data.clip_end = 50
    az, el = math.radians(azim_deg), math.radians(elev_deg)
    # azim 0 = mira hacia +y desde -y (el frente del personaje esta en -y)
    off = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el))) * dist
    cam.location = centro + off
    cam.rotation_euler = (centro - cam.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam


def render(ruta, w, h, color_tex=True, por_objeto=False):
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.film_transparent = False
    sc.world = sc.world or bpy.data.worlds.new("w")
    sc.world.color = (0.16, 0.17, 0.19)
    sh = sc.display.shading
    sh.light = "STUDIO"; sh.color_type = "OBJECT" if por_objeto else ("TEXTURE" if color_tex else "SINGLE"); sh.single_color = (0.72, 0.72, 0.74)
    sh.show_specular_highlight = False
    sc.view_settings.view_transform = "Standard"
    sc.render.image_settings.file_format = "PNG"; sc.render.filepath = ruta
    bpy.ops.render.render(write_still=True)


def solo(visibles):
    for o in bpy.data.objects:
        if o.type == "MESH":
            o.hide_render = o not in visibles


def main():
    trab, juego, salida = arg("--trab"), arg("--juego"), arg("--salida")
    ancho = int(arg("--ancho", 1000)); celda = ancho // 3
    bpy.ops.wm.open_mainfile(filepath=juego)
    bajos = [o for o in bpy.data.objects if o.type == "MESH"]
    # malla alta con sus texturas (si hay altas.blend)
    altos = []
    ab = arg("--alto-blend", os.path.join(trab, "altas.blend"))
    with bpy.data.libraries.load(ab, link=False) as (src, dst):
        dst.objects = list(src.objects)
    for o in dst.objects:
        bpy.context.scene.collection.objects.link(o); altos.append(o)
    mn, mx = caja(altos + bajos)
    centro = (mn + mx) / 2; alto_h = mx.z - mn.z
    escala_cuerpo = max(alto_h, mx.x - mn.x) * 1.04
    vistas = [("frente", 0), ("3/4", 38), ("espalda", 180)]
    tmp = os.path.join(trab, "_hoja"); os.makedirs(tmp, exist_ok=True)
    celdas = []
    for fila, grupo in (("alto", altos), ("juego", bajos)):
        for nom, az in vistas:
            solo(grupo); camara(centro, az, 6, escala_cuerpo)
            r = os.path.join(tmp, "%s_%s.png" % (fila, nom.replace("/", "")))
            render(r, celda, celda); celdas.append(r)
    # alambre del juego: copia gris lisa + copia con modificador de alambre (color de objeto oscuro), sin textura
    cab = next((o for o in bajos if "cabeza" in o.name), bajos[0]); cue = next((o for o in bajos if "cuerpo" in o.name), bajos[-1])

    def con_alambre(obs):
        extra = []
        for o in obs:
            o.color = (0.80, 0.80, 0.83, 1)
            c = o.copy(); c.data = o.data; c.name = "alambre_" + o.name; bpy.context.scene.collection.objects.link(c)
            m = c.modifiers.new("alambre", "WIREFRAME"); m.thickness = 0.0007; m.use_replace = True; m.offset = 0.0
            c.color = (0.02, 0.02, 0.05, 1); extra.append(c)
        return extra

    c0, c1 = caja([cab]); cabc = (c0 + c1) / 2
    zonas = [("cara", [cab], Vector((cabc.x, c0.y, cabc.z)), 0, (c1.z - c0.z) * 1.15),
             ("hombro", [cue], Vector((0.27, 0.0, 1.52)), 20, 0.55),
             ("mano", [cue], Vector((mx.x - 0.11, 0.0, 1.50)), 0, 0.30)]
    for nom, grupo, c, az, esc in zonas:
        ex = con_alambre(grupo)
        solo(grupo + ex); camara(c, az, 8, esc)
        r = os.path.join(tmp, "alambre_%s.png" % nom)
        render(r, celda, celda, color_tex=False, por_objeto=True); celdas.append(r)
        for c_ in ex:
            bpy.data.objects.remove(c_)
    # montaje 3x3 con numpy
    filas = []
    for i in range(3):
        fila = []
        for j in range(3):
            im = bpy.data.images.load(celdas[i * 3 + j])
            px = np.empty(im.size[0] * im.size[1] * 4, np.float32); im.pixels.foreach_get(px)
            fila.append(px.reshape(im.size[1], im.size[0], 4)); bpy.data.images.remove(im)
        filas.append(np.concatenate(fila, axis=1))
    lienzo = np.concatenate(filas[::-1], axis=0)       # el origen de la imagen esta abajo: la primera fila queda arriba
    out = bpy.data.images.new("hoja", lienzo.shape[1], lienzo.shape[0], alpha=False)
    out.pixels.foreach_set(lienzo.ravel()); out.filepath_raw = salida; out.file_format = "JPEG"
    bpy.context.scene.render.image_settings.quality = 88
    out.save()
    log("hoja", salida, lienzo.shape)


main()
