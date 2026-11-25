"""proxy_visor.py -- malla ligera para el visor Godot a partir de un modelo de >1M de triangulos, conservando UV y materiales.

    blender -b -P proxy_visor.py -- --glb modelo.glb --salida DIR [--nombre proxy] [--triangulos 130000] [--textura 2048] [--proteger-borde 0..1]

Metodo (sin bpy.ops sobre la malla grande):
  1. Se importa el glb; las caras se reparten por material en bmeshes separados y en cada uno se sueldan los vertices coincidentes
     (el importador duplica por UV/normal, y el decimador no puede colapsar aristas de vertices sueltos) y se separan las
     aristas con mas de 2 caras (el colapso de Blender se queda en ~12 % de las caras si las deja). Entre materiales NO se
     suelda: la costura cabeza/cuerpo queda abierta como en el modelo.
  2. Decimado por colapso (modificador Decimate evaluado, no se aplica sobre la original) con un reparto de triangulos por
     material proporcional al original; las UV se interpolan en los bucles. Normales suaves recalculadas.
  3. Atlas: los N materiales (color y metal-rugosidad, uno por material) se reducen a `--textura` px y se pegan lado a lado en una
     sola imagen (N*textura x textura); las UV de cada material se remapean a su casilla. El visor solo admite un juego de texturas.
  4. Se escribe DIR/<nombre>.blend (una malla, un material de atlas con color y metal-rugosidad empaquetados) e imprime 'PROXY {resumen}'.
Los grupos de vertices (pesos) los anade despues `transferir_pesos.py`."""
import json
import math
import os
import sys
import time

import bmesh
import bpy
import numpy as np

a = sys.argv[sys.argv.index("--") + 1:]


def arg(n, d=None):
    return a[a.index(n) + 1] if n in a else d


T0 = time.time()


def log(*m):
    print("[%5.1fs]" % (time.time() - T0), *m, flush=True)


def imagen_de(mat, clave):
    """Imagen de color (clave 'color') o de metal-rugosidad ('mat') del material, siguiendo los enlaces del nodo Principled."""
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    entrada = bsdf.inputs["Base Color" if clave == "color" else "Roughness"]
    nodo = entrada.links[0].from_node if entrada.is_linked else None
    for _ in range(4):
        if nodo is None or nodo.type == "TEX_IMAGE":
            break
        ent = next((i for i in nodo.inputs if i.is_linked), None)
        nodo = ent.links[0].from_node if ent else None
    return nodo.image if nodo is not None and nodo.type == "TEX_IMAGE" else None


def pixeles(img, lado):
    """Imagen -> array (lado, lado, 4) float32 (reescalada en una copia)."""
    c = img.copy()
    if tuple(c.size) != (lado, lado):
        c.scale(lado, lado)
    px = np.empty(lado * lado * 4, np.float32)
    c.pixels.foreach_get(px)
    bpy.data.images.remove(c)
    return px.reshape(lado, lado, 4)


def main():
    glb, salida = arg("--glb"), arg("--salida")
    if not glb or not salida:
        raise SystemExit("faltan --glb y --salida")
    nombre, objetivo, lado = arg("--nombre", "proxy"), int(arg("--triangulos", 130000)), int(arg("--textura", 2048))
    proteger_borde = float(arg("--proteger-borde", 0.0))
    os.makedirs(salida, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb)
    ob = next(o for o in bpy.context.scene.objects if o.type == "MESH")
    me = ob.data
    mats = list(me.materials)
    nm = len(mats)
    ntri = len(me.polygons)
    mi = np.empty(ntri, np.int32); me.polygons.foreach_get("material_index", mi)
    cuenta = np.bincount(mi, minlength=nm)
    log("importado: %d vertices, %d triangulos, materiales %s" % (len(me.vertices), ntri, dict(zip([m.name for m in mats], cuenta.tolist()))))

    # --- 1-2. por material: se separan las caras, se sueldan los vertices y se decima
    partes = []
    for k in range(nm):
        bm = bmesh.new(); bm.from_mesh(me)
        borrar = [f for f in bm.faces if f.material_index != k]
        bmesh.ops.delete(bm, geom=borrar, context="FACES")
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
        # aristas con >2 caras (capas solapadas que comparten borde): bloquean el colapso; se separan en bordes abiertos
        bmesh.ops.split_edges(bm, edges=[e for e in bm.edges if len(e.link_faces) > 2])
        borde = [v.index for v in bm.verts if any(len(e.link_faces) == 1 for e in v.link_edges)]
        m2 = bpy.data.meshes.new("parte%d" % k); bm.to_mesh(m2); bm.free()
        o2 = bpy.data.objects.new("parte%d" % k, m2); bpy.context.scene.collection.objects.link(o2)
        if proteger_borde:      # sin esto el colapso se come los bordes abiertos y la malla se llena de agujeros
            vgb = o2.vertex_groups.new(name="borde")      # peso 1 = se decima con normalidad, 0 = se protege
            vgb.add(list(range(len(m2.vertices))), 1.0, "REPLACE"); vgb.add(borde, 1.0 - proteger_borde, "REPLACE")
        log("parte %d soldada: %d vertices, %d triangulos" % (k, len(m2.vertices), len(m2.polygons)))
        mod = o2.modifiers.new("dec", "DECIMATE"); mod.decimate_type = "COLLAPSE"
        mod.use_collapse_triangulate = True
        if proteger_borde:
            mod.vertex_group = "borde"; mod.vertex_group_factor = proteger_borde
        meta = objetivo * cuenta[k] / float(ntri)     # reparto proporcional al original
        mod.ratio = min(1.0, meta / len(m2.polygons))
        for _ in range(3):      # el colapso no alcanza siempre la razon pedida: se corrige con lo medido
            bpy.context.view_layer.update()
            dg = bpy.context.evaluated_depsgraph_get()
            ev = bpy.data.meshes.new_from_object(o2.evaluated_get(dg), depsgraph=dg)
            nt_ = len(ev.polygons)
            log("parte %d decimada (razon %.4f): %d vertices, %d triangulos" % (k, mod.ratio, len(ev.vertices), nt_))
            if abs(nt_ - meta) < 0.06 * meta:
                break
            bpy.data.meshes.remove(ev)
            mod.ratio = min(1.0, mod.ratio * meta / nt_)
            mod.show_viewport = False; mod.show_viewport = True
        else:
            dg = bpy.context.evaluated_depsgraph_get()
            ev = bpy.data.meshes.new_from_object(o2.evaluated_get(dg), depsgraph=dg)
        partes.append(ev)
        bpy.data.objects.remove(o2)

    # --- 3. atlas
    atlas = {}
    for clave in ("color", "mat"):
        lienzo = np.zeros((lado, lado * nm, 4), np.float32); lienzo[..., 3] = 1.0
        for k, mat in enumerate(mats):
            img = imagen_de(mat, clave)
            if img is not None:
                lienzo[:, k * lado:(k + 1) * lado] = pixeles(img, lado)
        im = bpy.data.images.new("atlas_" + clave, lado * nm, lado, alpha=False)
        im.colorspace_settings.name = "sRGB" if clave == "color" else "Non-Color"
        im.pixels.foreach_set(lienzo.ravel()); im.pack(); atlas[clave] = im
        log("atlas %s %dx%d" % (clave, lado * nm, lado))

    # --- 4. malla final: partes unidas con UV remapeadas
    bm = bmesh.new()
    for k, ev in enumerate(partes):
        b2 = bmesh.new(); b2.from_mesh(ev)
        uvl = b2.loops.layers.uv.active
        for f in b2.faces:
            for lp in f.loops:
                u, v = lp[uvl].uv
                lp[uvl].uv = ((min(max(u, 0.0), 1.0) + k) / nm, v)
            f.material_index = 0
        tmp = bpy.data.meshes.new("t"); b2.to_mesh(tmp); b2.free()
        bm.from_mesh(tmp); bpy.data.meshes.remove(tmp)
    bm.normal_update()
    final = bpy.data.meshes.new(nombre); bm.to_mesh(final); bm.free()
    final.polygons.foreach_set("use_smooth", [True] * len(final.polygons))
    final.update()
    # indice de la parte (material de origen) por vertice, guardado como atributo para `transferir_pesos.py`
    parte = np.concatenate([np.full(len(ev.vertices), k, np.int32) for k, ev in enumerate(partes)])
    att = final.attributes.new("parte", "INT", "POINT"); att.data.foreach_set("value", parte)

    mat = bpy.data.materials.new("atlas"); mat.use_nodes = True
    nt = mat.node_tree; bsdf = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexImage"); tc.image = atlas["color"]
    tm = nt.nodes.new("ShaderNodeTexImage"); tm.image = atlas["mat"]
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    nt.links.new(tc.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(tm.outputs["Color"], sep.inputs["Color"])
    nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
    nt.links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
    final.materials.append(mat)
    po = bpy.data.objects.new(nombre, final); bpy.context.scene.collection.objects.link(po)
    for o in list(bpy.context.scene.objects):
        if o is not po:
            bpy.data.objects.remove(o)
    bpy.data.meshes.remove(me)
    ruta = os.path.join(salida, nombre + ".blend")
    bpy.ops.wm.save_as_mainfile(filepath=ruta)
    res = {"vertices": len(final.vertices), "triangulos": len(final.polygons), "atlas": [lado * nm, lado],
           "partes_triangulos": [len(e.polygons) for e in partes], "segundos": round(time.time() - T0, 1)}
    json.dump(res, open(os.path.join(salida, nombre + "_proxy.json"), "w"))
    print("PROXY " + json.dumps(res))


main()
