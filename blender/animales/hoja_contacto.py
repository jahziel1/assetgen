"""hoja_contacto.py -- renderiza (Workbench) fotogramas de clips de un GLB animado, vista lateral, y los junta en una hoja JPG.

    blender -b -P hoja_contacto.py -- <animal.glb> <hoja.jpg> [--clips Walk_Loop,Gallop_Loop] [--n 6] [--cols 3] [--ancho 990] [--solo-frames dir]
Cada clip ocupa un bloque de n cuadros (cols columnas); la hoja se compone con numpy (filas en el orden de --clips).
Una linea gris marca el suelo (Z=0) para ver pies que flotan o se hunden.
"""
import math
import os
import subprocess
import sys

import bpy
import numpy as np
from mathutils import Vector


def args():
    a = sys.argv[sys.argv.index("--") + 1:]
    opt, pos, i = {}, [], 0
    while i < len(a):
        if a[i].startswith("--"):
            opt[a[i]] = a[i + 1] if i + 1 < len(a) else True
            i += 2
        else:
            pos.append(a[i])
            i += 1
    return pos, opt


def main():
    pos, opt = args()
    glb, out = pos[0], pos[1]
    n = int(opt.get("--n", 6))
    cols = int(opt.get("--cols", 3))
    ancho = int(opt.get("--ancho", 990))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.fps = 30
    bpy.ops.import_scene.gltf(filepath=glb)
    for o in list(bpy.data.objects):   # formas de hueso que crea el importador
        if o.type == "MESH" and ("Icosphere" in o.name or not o.vertex_groups):
            bpy.data.objects.remove(o, do_unlink=True)
    arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    acts = {a.name: a for a in bpy.data.actions}
    clips = opt.get("--clips", "Walk_Loop,Gallop_Loop").split(",")
    clips = [c for c in clips if c in acts] or sorted(acts)[:2]
    # marco: bbox en reposo (sin animar) -> eje largo
    dg = bpy.context.evaluated_depsgraph_get()

    def bbox():
        pts = []
        for m in meshes:
            ev = m.evaluated_get(bpy.context.evaluated_depsgraph_get())
            me = ev.to_mesh()
            co = np.zeros(len(me.vertices) * 3)
            me.vertices.foreach_get("co", co)
            M = np.array(m.matrix_world)
            co = co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
            pts.append(co)
            ev.to_mesh_clear()
        P = np.concatenate(pts)
        return P.min(0), P.max(0)

    lo, hi = bbox()
    ext = hi - lo
    ax = 0 if ext[0] >= ext[1] else 1  # eje largo
    cen = (lo + hi) / 2
    escala = max(ext[ax], ext[2] * 1.5) * 1.5
    # materiales sin metal para que Workbench muestre la textura
    sc.render.engine = "BLENDER_WORKBENCH"
    sh = sc.display.shading
    sh.light = "STUDIO"
    sh.color_type = "TEXTURE"
    sh.show_shadows = False
    sc.render.film_transparent = False
    w = bpy.data.worlds.new("w")
    w.color = (0.93, 0.93, 0.93)
    sc.world = w
    sc.render.resolution_x = ancho // cols
    sc.render.resolution_y = int(ancho // cols * 0.66)
    sc.render.image_settings.file_format = "PNG"
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = escala
    d = Vector((0, -1, 0)) if ax == 0 else Vector((1, 0, 0))   # mira de lado: eje largo = horizontal en la imagen
    cam.location = Vector(cen) - d * 10
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    sc.camera = cam
    # suelo: caja fina a Z=0
    bpy.ops.mesh.primitive_cube_add(size=1)
    suelo = bpy.context.active_object
    suelo.scale = (ext[0] * 3, ext[1] * 3, 0.006 * max(1, ext[2]))
    suelo.location = (cen[0], cen[1], -0.003 * max(1, ext[2]))
    suelo.color = (0.45, 0.45, 0.45, 1)
    tmp = opt.get("--solo-frames") or os.path.splitext(out)[0] + "_frames"
    os.makedirs(tmp, exist_ok=True)
    files = []
    arm.animation_data_create()
    for c in clips:
        act = acts[c]
        arm.animation_data.action = act
        if act.slots:
            arm.animation_data.action_slot = act.slots[0]
        f0, f1 = act.frame_range
        for i in range(n):
            f = f0 + (f1 - f0) * i / n  # ciclo completo sin repetir el ultimo cuadro
            sc.frame_set(int(math.floor(f)), subframe=f - math.floor(f))
            p = os.path.join(tmp, f"{c}_{i}.png")
            sc.render.filepath = p
            bpy.ops.render.render(write_still=True)
            files.append((c, p))
    # composicion con numpy + imagenes de Blender 
    w1, h1 = sc.render.resolution_x, sc.render.resolution_y
    filas = math.ceil(n / cols)
    H, Wt = h1 * filas * len(clips), w1 * cols
    hoja = np.ones((H, Wt, 4), np.float32)
    for k, (c, p) in enumerate(files):
        ci = clips.index(c)
        i = k % n
        x, y = (i % cols) * w1, (ci * filas + i // cols) * h1
        im = bpy.data.images.load(p)
        a = np.array(im.pixels[:], np.float32).reshape(h1, w1, 4)   # origen abajo-izquierda
        hoja[H - y - h1:H - y, x:x + w1] = a
        hoja[H - y - h1:H - y, x:x + 2] = 1.0   # separadores finos
        hoja[H - y - 2:H - y, x:x + w1] = 1.0
        bpy.data.images.remove(im)
    img = bpy.data.images.new("hoja", Wt, H)
    img.pixels.foreach_set(hoja.ravel())
    img.file_format = "JPEG"
    img.filepath_raw = out
    img.save()
    print("HOJA", out, (Wt, H), "filas:", clips)
    for _, p in files:
        os.remove(p)
    try:
        os.rmdir(tmp)
    except OSError:
        pass
    print("HOJA", out, hoja.size)


main()
