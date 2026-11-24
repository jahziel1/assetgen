"""medir_blender.py -- adaptador FBX (y cualquier formato que Blender importe) de `assetgen remaster analizar`.

    blender -b -P medir_blender.py -- --entrada modelo.fbx --salida modelo.json

Escribe el "modelo normalizado" (ver cli/assetgen_cli/remaster_comun.py) con coordenadas en el espacio canonico Y arriba:
canonico = (bx, bz, -by) respecto a Blender. Vertices = vertices de GPU estimados (vertice + UV distintos), triangulos = suma de
(lados - 2) de cada poligono. Un FBX no declara el tipo de indice (tipo_indice = null). Imprime 'MEDIR {resumen}'."""
import json
import math
import os
import sys

import bpy
import numpy as np

a = sys.argv[sys.argv.index("--") + 1:]


def arg(n, d=None):
    return a[a.index(n) + 1] if n in a else d


entrada, salida = arg("--entrada"), arg("--salida")
if not entrada or not salida:
    sys.exit("uso: --entrada archivo --salida modelo.json")

bpy.ops.wm.read_factory_settings(use_empty=True)
ext = os.path.splitext(entrada)[1].lower()
if ext == ".fbx":
    bpy.ops.import_scene.fbx(filepath=entrada)
elif ext in (".glb", ".gltf"):
    bpy.ops.import_scene.gltf(filepath=entrada)
else:
    sys.exit("formato no soportado por el adaptador Blender: " + ext)
bpy.context.view_layer.update()


def canon(v):
    return [round(float(v[0]), 6), round(float(v[2]), 6), round(float(-v[1]), 6)]


mod = {"formato": "fbx" if ext == ".fbx" else "gltf", "origen": os.path.basename(entrada), "unidades": None, "avisos": [], "piezas": [],
       "esqueleto": None, "materiales": [], "bbox": None, "muestra_puntos": [], "ejes_declarados": None}
if ext == ".fbx":
    mod["avisos"].append("fbx: coordenadas tras importar en Blender y pasadas a Y arriba; las unidades dependen del archivo (suelen ser cm o m)")

# ---- esqueleto
arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
if arms:
    arm = max(arms, key=lambda o: len(o.data.bones))
    huesos = list(arm.data.bones)
    ind = {b.name: i for i, b in enumerate(huesos)}
    mod["esqueleto"] = {"fuente": "armature del archivo", "huesos": [
        {"indice": i, "nombre": b.name, "padre": ind[b.parent.name] if b.parent else -1,
         "reposo": canon(arm.matrix_world @ b.head_local)} for i, b in enumerate(huesos)]}
    if len(arms) > 1:
        mod["avisos"].append(f"el archivo tiene {len(arms)} armatures; se usa la de mas huesos ({len(huesos)})")
    nombre_hueso = ind
else:
    nombre_hueso = {}

# ---- mallas
todos = []
mats_usados = []
for o in bpy.context.scene.objects:
    if o.type != "MESH":
        continue
    me = o.data
    me.calc_loop_triangles()
    nv = len(me.vertices)
    co = np.empty(nv * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    mw = np.array(o.matrix_world)
    w = co @ mw[:3, :3].T + mw[:3, 3]
    can = np.stack([w[:, 0], w[:, 2], -w[:, 1]], axis=1)
    todos.append(can)
    nloops = len(me.loops)
    vi = np.empty(nloops, dtype=np.int64)
    me.loops.foreach_get("vertex_index", vi)
    if me.uv_layers:
        uv = np.empty(nloops * 2)
        me.uv_layers[0].data.foreach_get("uv", uv)
        clave = np.round(uv.reshape(-1, 2), 5)
        gpu = len(np.unique(np.column_stack([vi, clave]), axis=0))
    else:
        gpu = nv
    tri = int(sum(max(len(p.vertices) - 2, 0) for p in me.polygons))
    max_p, smin, smax, usados = 0, None, None, set()
    grupos = {g.index: g.name for g in o.vertex_groups}
    if grupos:
        for v in me.vertices:
            ws = [g for g in v.groups if g.weight > 1e-6]
            suma = sum(g.weight for g in ws)
            max_p = max(max_p, len(ws))
            smin = suma if smin is None else min(smin, suma)
            smax = suma if smax is None else max(smax, suma)
            for g in ws:
                n = grupos.get(g.group)
                if n in nombre_hueso:
                    usados.add(nombre_hueso[n])
    mats = [s.material for s in o.material_slots if s.material]
    for m in mats:
        if m not in mats_usados:
            mats_usados.append(m)
    mod["piezas"].append({
        "nombre": o.name, "vertices": gpu, "triangulos": tri, "tipo_indice": None, "max_pesos": max_p,
        "suma_pesos": [round(smin, 5), round(smax, 5)] if smin is not None else None, "huesos_usados": sorted(usados),
        "uvs": len(me.uv_layers), "material": mats[0].name if mats else None,
        "bbox": [[round(float(v), 6) for v in can.min(0)], [round(float(v), 6) for v in can.max(0)]], "con_pesos": bool(grupos)})
if not mod["piezas"]:
    sys.exit("MEDIR_ERROR sin mallas")
allp = np.concatenate(todos)
mod["bbox"] = [[round(float(v), 6) for v in allp.min(0)], [round(float(v), 6) for v in allp.max(0)]]
paso = max(len(allp) // 20000, 1)
mod["muestra_puntos"] = allp[::paso].tolist()

# ---- materiales
ROLES = {"Base Color": "color", "Normal": "normal", "Metallic": "metal_rugosidad", "Roughness": "metal_rugosidad",
         "Emission Color": "emision", "Emission": "emision"}
for m in mats_usados:
    texs = []
    if m.use_nodes:
        for n in m.node_tree.nodes:
            if n.type != "TEX_IMAGE" or not n.image:
                continue
            rol, vistos = "otra", [n]
            while vistos and rol == "otra":          # sigue los enlaces hasta una entrada conocida del shader
                x = vistos.pop()
                for sal in x.outputs:
                    for l in sal.links:
                        if l.to_socket.name in ROLES:
                            rol = ROLES[l.to_socket.name]
                        else:
                            vistos.append(l.to_node)
            texs.append({"rol": rol, "archivo": os.path.basename(n.image.filepath) or n.image.name,
                         "ancho": int(n.image.size[0]), "alto": int(n.image.size[1]), "uv": 0})
    mod["materiales"].append({"nombre": m.name, "texturas": texs, "doble_cara": not m.use_backface_culling})
json.dump(mod, open(salida, "w"))
print("MEDIR " + json.dumps({"piezas": len(mod["piezas"]), "huesos": len(mod["esqueleto"]["huesos"]) if mod["esqueleto"] else 0}))
