"""verificar_glb.py -- reimporta un .glb final (humanoide o mecanico), lista clips y texturas y renderiza una hoja de cuadros
de su clip principal (<=1000 px de ancho).

    blender -b -P verificar_glb.py -- modelo.glb hoja.png [--clip Walk_Loop] [--n 6] [--celda 320x320] [--json salida.json]

Imprime 'VERIF {...}'. Sirve para cualquier GLB: clips por accion (armadura) o por pistas NLA (objetos animados).
Si no se da --clip se elige: Walk_Loop > un clip con 'rod'/'gir' > el primero. Render Workbench (color = textura), camara 3/4 ortografica
encuadrada por la caja de la malla.
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
N = int(arg("--n", "6"))
cw, ch = (int(x) for x in arg("--celda", "320x320").split("x"))
COLS = 3

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
objs = list(bpy.data.objects)
mallas = [o for o in objs if o.type == "MESH"]
armaduras = [o for o in objs if o.type == "ARMATURE"]

# ---- clips: acciones + pistas NLA
nombres = set(x.name for x in bpy.data.actions)
for o in objs:
    ad = o.animation_data
    if ad:
        for t in ad.nla_tracks:
            nombres.add(t.name)
            for s in t.strips:
                nombres.add(s.name)
clips_act = sorted(x.name for x in bpy.data.actions)
tex = []
for m in mallas:
    for mat in m.data.materials:
        if mat and mat.node_tree:
            for n in mat.node_tree.nodes:
                if n.type == "TEX_IMAGE" and n.image:
                    lig = any(l.from_node == n for l in mat.node_tree.links)
                    tex.append({"material": mat.name, "imagen": n.image.name, "tam": list(n.image.size), "conectada": lig})
rep = {"acciones": clips_act, "pistas_nla": sorted({t.name for o in objs if o.animation_data for t in o.animation_data.nla_tracks}),
       "armaduras": len(armaduras), "huesos": sum(len(x.data.bones) for x in armaduras),
       "mallas": [(m.name, len(m.data.vertices), len(m.data.polygons)) for m in mallas],
       "objetos_animados": [o.name for o in objs if o.animation_data and (o.animation_data.action or o.animation_data.nla_tracks)],
       "texturas": tex, "textura_ok": bool(tex) and all(t["conectada"] and t["tam"][0] > 0 for t in tex)}


def coincide(clip, nombre):
    return clip == nombre or clip in nombre


candidatos = rep["pistas_nla"] + clips_act
clip = arg("--clip")
if not clip:
    clip = next((c for c in candidatos if c.startswith("Walk")), None) or \
        next((c for c in candidatos if "rod" in c.lower() or "gir" in c.lower()), None) or (candidatos[0] if candidatos else None)
rep["clip_principal"] = clip

# ---- activar solo el clip elegido y obtener su rango
rng = None
if clip:
    for o in objs:
        ad = o.animation_data
        if not ad:
            continue
        if ad.nla_tracks:
            if ad.action:  # accion activa suelta: fuera, manda NLA
                ad.action = None
            for t in ad.nla_tracks:
                if coincide(clip, t.name) or any(coincide(clip, s.name) for s in t.strips):
                    t.mute = False
                    for s in t.strips:
                        rng = (s.frame_start, s.frame_end) if rng is None else (min(rng[0], s.frame_start), max(rng[1], s.frame_end))
                else:
                    t.mute = True
        elif ad.action is not None and not coincide(clip, ad.action.name):
            # la armadura/objeto tiene otra accion: buscar la del clip
            ac = bpy.data.actions.get(clip)
            if ac:
                ad.action = ac
                if ac.slots:
                    ad.action_slot = ac.slots[0]
    for o in objs:
        ad = o.animation_data
        if ad and ad.action and not ad.nla_tracks and coincide(clip, ad.action.name):
            f0, f1 = ad.action.frame_range
            rng = (f0, f1) if rng is None else (min(rng[0], f0), max(rng[1], f1))
if rng is None:
    ac = bpy.data.actions.get(clip) if clip else None
    rng = tuple(ac.frame_range) if ac else (0, 24)
rep["rango_clip"] = [round(rng[0], 1), round(rng[1], 1)]
print("VERIF " + json.dumps(rep))
if arg("--json"):
    json.dump(rep, open(arg("--json"), "w"), indent=1)

# ---- render
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
sc.view_settings.view_transform = "Standard"
sc.world = bpy.data.worlds.new("w")
sc.world.color = (0.55, 0.62, 0.7)
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
bpy.context.collection.objects.link(cam)
cam.data.type = "ORTHO"
cam.data.clip_end = 500
sc.camera = cam

# caja de la malla en el reposo del primer cuadro y a mitad (maxima extension aproximada)
dg = bpy.context.evaluated_depsgraph_get()


def caja(frame):
    sc.frame_set(int(frame))
    dg.update()
    pts = []
    for m in mallas:
        if "Icosphere" in m.name:
            continue
        ev = m.evaluated_get(dg)
        me = ev.to_mesh()
        mw = ev.matrix_world
        pts.extend(mw @ v.co for v in list(me.vertices)[::7])
        ev.to_mesh_clear()
    arr = np.array([[p.x, p.y, p.z] for p in pts])
    return arr.min(0), arr.max(0)


frames = [rng[0] + (rng[1] - rng[0]) * i / N for i in range(N)]
cajas = [caja(f) for f in (frames[0], frames[len(frames) // 2])]
lo = np.min([c[0] for c in cajas], axis=0)
hi = np.max([c[1] for c in cajas], axis=0)
centro = (lo + hi) / 2
radio = float(np.linalg.norm(hi - lo)) / 2
cam.data.ortho_scale = max(radio * 2.1, 0.5)
az, el = math.radians(35), math.radians(20)
d = radio * 4 + 2
# camara por delante-izquierda en 3/4 (el frente de glTF es -Y en Blender tras importar)
cam.location = (centro[0] + d * math.sin(az) * math.cos(el), centro[1] - d * math.cos(az) * math.cos(el), centro[2] + d * math.sin(el))
cam.rotation_euler = (Vector(centro) - Vector(cam.location)).to_track_quat("-Z", "Y").to_euler()

tiles = []
tmp = os.path.join(tempfile.gettempdir(), "_assetgen_verif.png")
for f in frames:
    sc.frame_set(int(math.floor(f)), subframe=f - math.floor(f))
    sc.render.filepath = tmp
    bpy.ops.render.render(write_still=True)
    im = bpy.data.images.load(tmp)
    px = np.zeros(im.size[0] * im.size[1] * 4, dtype=np.float32)
    im.pixels.foreach_get(px)
    tiles.append(px.reshape(im.size[1], im.size[0], 4))
    bpy.data.images.remove(im)
filas = []
for i in range(0, len(tiles), COLS):
    fila = tiles[i:i + COLS]
    while len(fila) < COLS:
        fila.append(np.ones_like(tiles[0]))
    filas.append(np.concatenate(fila, axis=1))
H = np.concatenate(filas[::-1], axis=0)
img = bpy.data.images.new("hoja", H.shape[1], H.shape[0], alpha=True)
img.pixels.foreach_set(H.ravel())
img.filepath_raw = hoja
img.file_format = "PNG"
img.save()
print("HOJA", hoja, H.shape[1], "x", H.shape[0])
