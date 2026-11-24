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


def hoja_unida(sc, ob_cabeza, ojos, ctr, ancho, salida, centros_ojos, tam=336, ss=2):
    """Hoja de la cabeza unida (3 columnas x 2 filas, Workbench, gris con cavidades; los ojos conservan su textura).
    Fila 1: frente, 3/4 derecho, perfil. Fila 2: 3/4 izquierdo, primer plano de los ojos de frente y de 3/4."""
    gris = _mat("hoja_gris", (0.72, 0.72, 0.72, 1))
    orig = [s.material for s in ob_cabeza.material_slots]
    for s in ob_cabeza.material_slots:
        s.material = gris
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
    ctr = Vector(ctr)
    centro_ojos = (Vector(centros_ojos[0]) + Vector(centros_ojos[1])) / 2
    vistas = [("frente", 0, ctr, 1.5 * ancho), ("tres_cuartos_der", 35, ctr, 1.5 * ancho), ("perfil", 90, ctr, 1.5 * ancho),
              ("tres_cuartos_izq", -35, ctr, 1.5 * ancho), ("ojos", 0, centro_ojos, 0.62 * ancho),
              ("ojos_3_4", 40, centro_ojos, 0.62 * ancho)]
    tmp = tempfile.mkdtemp()
    tiles = []
    for nom, az, tgt, esc in vistas:
        cam.data.ortho_scale = esc
        t = math.radians(az)
        cam.location = tgt + Vector((math.sin(t) * 3, -math.cos(t) * 3, 0))
        cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
        ruta = os.path.join(tmp, nom + ".png")
        sc.render.filepath = ruta
        bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(ruta)
        w, h = im.size
        px = np.array(im.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1, :, :3]
        tiles.append(px.reshape(tam, ss, tam, ss, 3).mean(axis=(1, 3)))
        bpy.data.images.remove(im)
    mosaico = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    H, W = mosaico.shape[:2]
    out = np.ones((H, W, 4), dtype=np.float32)
    out[..., :3] = mosaico[::-1]
    im = bpy.data.images.new("hoja", W, H)
    im.pixels.foreach_set(out.reshape(-1))
    im.file_format = "JPEG"
    im.filepath_raw = salida
    im.save()
    bpy.data.images.remove(im)
    for s, m in zip(ob_cabeza.material_slots, orig):
        s.material = m
    print("HOJA", salida, W, H)


def _luces(sc, ctr):
    """Luz suave: sol principal con penumbra amplia desde arriba a la izquierda, relleno tenue desde la derecha y ambiente."""
    sc.world = bpy.data.worlds.new("hoja_w_ee")
    sc.world.use_nodes = True
    bg = sc.world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.55, 0.56, 0.58, 1)
    bg.inputs["Strength"].default_value = 0.9
    for nom, rot, e, ang in (("sol", (math.radians(55), 0, math.radians(35)), 2.6, 25), ("relleno", (math.radians(75), 0, math.radians(-60)), 0.9, 40)):
        luz = bpy.data.objects.new("hoja_" + nom, bpy.data.lights.new("hoja_" + nom, "SUN"))
        luz.data.energy = e
        luz.data.angle = math.radians(ang)
        luz.rotation_euler = rot
        luz.location = tuple(ctr)
        sc.collection.objects.link(luz)


def hoja_final(sc, objs_final, glb_antes, retrato, ctr, ancho, salida, centros_ojos, tam=336, ss=2):
    """Hoja de la cabeza final con textura y luz suave (Eevee; Workbench con textura si Eevee no esta). Fila 1: retrato original | cabeza antes
    (glb_antes) | cabeza final, de frente. Fila 2: 3/4 y perfil de la final y primer plano de los ojos."""
    try:
        sc.render.engine = "BLENDER_EEVEE"
    except TypeError:
        sc.render.engine = "BLENDER_EEVEE_NEXT"
    sc.render.resolution_x = sc.render.resolution_y = tam * ss
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    try:
        sc.eevee.taa_render_samples = 24
    except AttributeError:
        pass
    ctr = Vector(ctr)
    _luces(sc, ctr)
    cam = bpy.data.objects.new("hoja_cam_f", bpy.data.cameras.new("hoja_cam_f"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.data.type = "ORTHO"
    cam.data.clip_end = 100
    centro_ojos = (Vector(centros_ojos[0]) + Vector(centros_ojos[1])) / 2
    pre = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=glb_antes)
    antes = [o for o in bpy.data.objects if o.name not in pre]
    tmp = tempfile.mkdtemp()

    def tile(nom, az, tgt, esc, visibles_antes):
        for o in objs_final:
            o.hide_render = visibles_antes
        for o in antes:
            o.hide_render = not visibles_antes
        cam.data.ortho_scale = esc
        t = math.radians(az)
        cam.location = tgt + Vector((math.sin(t) * 3, -math.cos(t) * 3, 0))
        cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
        ruta = os.path.join(tmp, nom + ".png")
        sc.render.filepath = ruta
        bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(ruta)
        w, h = im.size
        px = np.array(im.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1, :, :3]
        bpy.data.images.remove(im)
        return px.reshape(tam, ss, tam, ss, 3).mean(axis=(1, 3))

    ret = bpy.data.images.load(retrato)
    ret.scale(tam, tam)
    rp = np.array(ret.pixels[:], dtype=np.float32).reshape(tam, tam, 4)[::-1, :, :3]
    # el retrato se muestra en sRGB tal cual: se pasa a lineal para que el "Standard" de la hoja lo devuelva igual
    rp = np.where(rp <= 0.04045, rp / 12.92, ((rp + 0.055) / 1.055) ** 2.4) if ret.colorspace_settings.name != "sRGB" else rp
    t_antes = tile("antes", 0, ctr, 1.5 * ancho, True)
    t_frente = tile("frente", 0, ctr, 1.5 * ancho, False)
    t_34 = tile("tres_cuartos", 35, ctr, 1.5 * ancho, False)
    t_perfil = tile("perfil", 90, ctr, 1.5 * ancho, False)
    t_ojos = tile("ojos", 0, centro_ojos, 0.62 * ancho, False)
    mosaico = np.vstack([np.hstack([rp, t_antes, t_frente]), np.hstack([t_34, t_perfil, t_ojos])])
    H, W = mosaico.shape[:2]
    out = np.ones((H, W, 4), dtype=np.float32)
    out[..., :3] = mosaico[::-1]
    im = bpy.data.images.new("hoja_f", W, H)
    im.colorspace_settings.name = "sRGB"
    im.pixels.foreach_set(out.reshape(-1))
    im.file_format = "JPEG"
    im.filepath_raw = salida
    im.save()
    bpy.data.images.remove(im)
    print("HOJA", salida, W, H)
