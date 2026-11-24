"""retarget_ual.py -- retarget de clips de la Universal Animation Library (Quaternius, CC0) a un humanoide riggeado por Puppeteer.

    blender -b -P retarget_ual.py -- --target soldado_rigged.glb --out soldado_ual.glb
        [--source UAL1_Standard.glb] [--clips "Idle_Loop=Idle_Loop,Walk_Loop=Walk_Loop,..."] [--mapa-json mapa.json]

Metodo (el de oldgames, sin constraints): por hueso mapeado, el DELTA DE ROTACION EN ESPACIO MUNDO de la fuente respecto a su reposo
se aplica al reposo del destino. Como el destino esta en pose A y la fuente en T, el reposo del destino se alinea antes con el de la
fuente por rotacion de arco minimo, pero calculada con los VECTORES ENTRE ARTICULACIONES (hueso -> siguiente articulacion semantica),
NO con el eje del hueso: los huesos de Puppeteer tienen ejes arbitrarios (p.ej. la cadera apunta en diagonal).
La traslacion de la cadera se escala por (altura de cadera destino / altura de cadera fuente). Despues se ajusta el suelo
(desplazamiento Z de la cadera) midiendo la malla evaluada: constante para bucles, por fotograma (suavizado) para Death.
Antes de animar, el destino se baja/sube para que los pies de la malla toquen Z=0 (la cadera de Puppeteer queda en el origen).
Los dedos quedan sin animar (heredan de la mano). Mapeo por posicion/topologia en mapeo_auto.py (no usa nombres).
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from bpy_extras import anim_utils
from mathutils import Matrix, Quaternion, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mapeo_auto import mapear, pares  # noqa: E402

# Ruta del pack Quaternius UAL1 (no se redistribuye): --source, o variable ASSETGEN_UAL1, o `quaternius.ual1` en ~/.config/assetgen/config.yaml
UAL1 = os.environ.get("ASSETGEN_UAL1", "")
# nombre de salida = accion de UAL (sufijo _Loop => bucle en Godot al importar)
CLIPS = [("Idle_Loop", "Idle_Loop"), ("Walk_Loop", "Walk_Loop"), ("Run_Loop", "Jog_Fwd_Loop"),
         ("Sword_Attack", "Sword_Attack"), ("Death", "Death01")]
POR_CUADRO = ("Death",)  # clips cuyo contacto con el suelo cambia (pie -> cuerpo): suelo por fotograma
FPS_FUENTE = 30.0  # UAL: claves cada 0.8 fotogramas a 24 fps = 30 fps


def rotq(m):
    return m.to_3x3().normalized().to_quaternion()


def cargar(src_path, tgt_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=src_path)
    for o in bpy.data.objects:
        o.name = "SRC_" + o.name
    ya = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=tgt_path)
    nuevos = [o.name for o in bpy.data.objects if o.name not in ya]
    for o in list(bpy.data.objects):  # esferas auxiliares de Puppeteer / UAL ("Icosphere")
        if o.type == "MESH" and "Icosphere" in o.name:
            bpy.data.objects.remove(o, do_unlink=True)
    src = [o for o in bpy.data.objects if o.type == "ARMATURE" and o.name.startswith("SRC_")][0]
    tgt = [bpy.data.objects[n] for n in nuevos if n in bpy.data.objects and bpy.data.objects[n].type == "ARMATURE"][0]
    meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.parent == tgt]
    return src, tgt, meshes


def bajar_al_suelo(tgt, meshes):
    """Mueve huesos y vertices en Z para que el punto mas bajo de la malla quede en Z=0. Devuelve dz (m)."""
    zs = []
    for m in meshes:
        n = len(m.data.vertices)
        co = np.zeros(n * 3)
        m.data.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        zs.append(co[:, 2].min())
    dz = -min(zs)
    bpy.context.view_layer.objects.active = tgt
    bpy.ops.object.mode_set(mode="EDIT")
    for eb in tgt.data.edit_bones:
        eb.head.z += dz
        eb.tail.z += dz
    bpy.ops.object.mode_set(mode="OBJECT")
    for m in meshes:
        n = len(m.data.vertices)
        co = np.zeros(n * 3)
        m.data.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        co[:, 2] += dz
        m.data.vertices.foreach_set("co", co.ravel())
        m.data.update()
    bpy.context.view_layer.update()
    return dz


def vec_siguiente(arm, m, clave):
    """Vector (mundo) hueso -> siguiente articulacion semantica, para alinear el reposo. None = sin alineacion (cadera)."""
    M = arm.matrix_world
    B = arm.data.bones
    P = pares(m)

    def H(n):
        return M @ B[n].head_local

    def T(n):
        return M @ B[n].tail_local

    if clave.startswith("spine_"):
        idx = int(clave.split("_")[1])
        sp = m["spine"]
        if idx < len(sp):
            return H(sp[idx]) - H(sp[idx - 1])
        nxt = m["neck"] or m["head"]
        return (H(nxt) - H(sp[-1])) if nxt else (T(sp[-1]) - H(sp[-1]))
    if clave == "hips":
        return None
    if clave == "neck":
        return H(m["head"]) - H(m["neck"]) if m["head"] else None
    if clave == "head":
        return T(m["head"]) - H(m["head"])
    s = clave[-1]
    k = clave[:-2]
    d = m[s]
    cadena = {"shoulder": "upper_arm", "upper_arm": "forearm", "forearm": "hand", "thigh": "shin", "shin": "foot"}
    if k in cadena and d.get(cadena[k]):
        return H(d[cadena[k]]) - H(d[k])
    if k == "foot":
        if d.get("toe"):
            return H(d["toe"]) - H(d["foot"])
        return T(d["foot"]) - H(d["foot"])
    if k == "hand":
        hijos = [c.name for c in B[d["hand"]].children]
        if hijos:
            c = sum((H(x) for x in hijos), Vector()) / len(hijos)
            return c - H(d["hand"])
        return T(d["hand"]) - H(d["hand"])
    if k == "toe":
        return T(d["toe"]) - H(d["toe"])
    return None


def emparejar(ms, mt):
    """{hueso_destino: hueso_fuente}. La columna se reparte proporcionalmente si tienen distinta longitud."""
    fs, ft = pares(ms), pares(mt)
    out = {}
    for k, tb in ft.items():
        if k.startswith("spine_"):
            continue
        if k in fs:
            out[tb] = fs[k]
    n, mm = len(mt["spine"]), len(ms["spine"])
    for i, tb in enumerate(mt["spine"]):
        out[tb] = ms["spine"][round(i * (mm - 1) / max(1, n - 1))] if n > 1 else ms["spine"][-1]
    return out


def muestrear(src, tgt, ms, mt, bmap, accion, escala_pos, ang_corr):
    """Devuelve (n_cuadros, {hueso: [Quaternion local por cuadro]}, [Vector loc cadera por cuadro])."""
    sc = bpy.context.scene
    src.animation_data_create()
    src.animation_data.action = accion
    if accion.slots:
        src.animation_data.action_slot = accion.slots[0]
    f0, f1 = accion.frame_range
    n = int(round((f1 - f0) / 0.8)) + 1
    B = tgt.data.bones
    orden = sorted(B, key=lambda b: len(b.parent_recursive))
    rest_rel = {b.name: (b.parent.matrix_local.inverted() @ b.matrix_local if b.parent else b.matrix_local.copy()) for b in orden}
    tw = rotq(tgt.matrix_world)
    t_rest = {b: rotq(tgt.matrix_world @ B[b].matrix_local) for b in bmap}
    sc.frame_set(0)
    s_rest = {s: rotq(src.matrix_world @ src.data.bones[s].matrix_local) for s in bmap.values()}
    t_rest_v = {b: ang_corr[b] @ t_rest[b] for b in bmap}  # reposo virtual (T) del destino
    hips_t, hips_s = mt["hips"], ms["hips"]
    s_hips_rest = src.matrix_world @ src.data.bones[hips_s].head_local
    t_hips_rest = tgt.matrix_world @ B[hips_t].head_local
    keys = {b: [] for b in bmap}
    locs = []
    for i in range(n):
        f = f0 + i * 0.8
        sc.frame_set(int(math.floor(f)), subframe=f - math.floor(f))
        pose = {}
        for b in orden:
            Mp = pose[b.parent.name] if b.parent else Matrix.Identity(4)
            base = Mp @ rest_rel[b.name]
            if b.name not in bmap:
                pose[b.name] = base
                continue
            s = bmap[b.name]
            sw = rotq(src.matrix_world @ src.pose.bones[s].matrix)
            dw = sw @ s_rest[s].inverted() @ t_rest_v[b.name]
            da = tw.inverted() @ dw
            lq = rotq(base).inverted() @ da
            prev = keys[b.name][-1] if keys[b.name] else None
            if prev is not None and lq.dot(prev) < 0:
                lq.negate()
            loc = Vector()
            if b.name == hips_t:
                hs = src.matrix_world @ src.pose.bones[s].head
                want_w = t_hips_rest + (hs - s_hips_rest) * escala_pos
                want_a = tgt.matrix_world.inverted() @ want_w
                loc = base.to_3x3().inverted() @ (want_a - base.translation)
                locs.append(loc.copy())
            pose[b.name] = base @ Matrix.Translation(loc) @ lq.to_matrix().to_4x4()
            keys[b.name].append(lq)
    src.animation_data.action = None
    return n, keys, locs


def zmin_malla(meshes):
    dg = bpy.context.evaluated_depsgraph_get()
    z = 1e9
    for m in meshes:
        ev = m.evaluated_get(dg)
        me = ev.to_mesh()
        co = np.zeros(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        z = min(z, float((np.c_[co, np.ones(len(co))] @ np.array(m.matrix_world).T)[:, 2].min()))
        ev.to_mesh_clear()
    return z


def escribir_accion(tgt, nombre, n, keys, locs, dz, hips):
    """Crea la accion `nombre` en tgt. dz = lista (por cuadro) de desplazamiento Z mundo de la cadera."""
    if nombre in bpy.data.actions:
        bpy.data.actions.remove(bpy.data.actions[nombre])
    B = tgt.data.bones
    base = B[hips].matrix_local
    Rinv = base.to_3x3().inverted()  # parent = None => base = matrix_local
    act = bpy.data.actions.new(nombre)
    act.use_fake_user = True
    slot = act.slots.new(id_type="OBJECT", name=tgt.name)
    tgt.animation_data_create()
    tgt.animation_data.action = act
    tgt.animation_data.action_slot = slot
    cb = anim_utils.action_ensure_channelbag_for_slot(act, slot)
    lin = bpy.types.Keyframe.bl_rna.properties["interpolation"].enum_items["LINEAR"].value

    def write(path, idx, grupo, vals):
        fc = cb.fcurves.ensure(path, index=idx, group_name=grupo)
        fc.keyframe_points.add(len(vals))
        fc.keyframe_points.foreach_set("co", [v for i, x in enumerate(vals) for v in (float(i), float(x))])
        fc.keyframe_points.foreach_set("interpolation", [lin] * len(vals))
        fc.update()

    for b, qs in keys.items():
        tgt.pose.bones[b].rotation_mode = "QUATERNION"
        for c in range(4):
            write(f'pose.bones["{b}"].rotation_quaternion', c, b, [q[c] for q in qs])
    L = [locs[i] + Rinv @ Vector((0, 0, dz[i])) for i in range(n)]
    for c in range(3):
        write(f'pose.bones["{hips}"].location', c, hips, [v[c] for v in L])
    return act


def retarget(src_path, tgt_path, clips, tol=0.004):
    src, tgt, meshes = cargar(src_path, tgt_path)
    for pb in tgt.pose.bones:
        pb.rotation_mode = "QUATERNION"
    ms, mt = mapear(src), mapear(tgt)
    bmap = emparejar(ms, mt)
    dz0 = bajar_al_suelo(tgt, meshes)
    # alineacion A -> T por vectores entre articulaciones
    ang = {}
    ps, pt = pares(ms), pares(mt)
    for k, tb in pt.items():
        vt = vec_siguiente(tgt, mt, k)
        vs = vec_siguiente(src, ms, k) if k in ps else None
        ang[tb] = vt.rotation_difference(vs) if (vt is not None and vs is not None and vt.length > 1e-6 and vs.length > 1e-6) else Quaternion()
    for tb in bmap:
        ang.setdefault(tb, Quaternion())
    # escala de la traslacion: altura de cadera (articulacion del muslo) destino / fuente
    ht = np.mean([(tgt.matrix_world @ tgt.data.bones[mt[s]["thigh"]].head_local).z for s in "LR"])
    hs = np.mean([(src.matrix_world @ src.data.bones[ms[s]["thigh"]].head_local).z for s in "LR"])
    escala = float(ht / hs)
    info = {"dz_suelo_inicial": round(dz0, 4), "escala_cadera": round(escala, 4), "mapa_hueso_destino_a_fuente": bmap,
            "correccion_reposo_deg": {b: round(math.degrees(q.angle), 1) for b, q in ang.items() if q.angle > 0.01}, "clips": {}}
    sc = bpy.context.scene
    sc.render.fps = 30
    sc.render.fps_base = 1.0
    hips = mt["hips"]
    acts = {}
    for nombre, an in clips:
        accion = bpy.data.actions[an]
        n, keys, locs = muestrear(src, tgt, ms, mt, bmap, accion, escala, ang)
        sc.frame_end = max(sc.frame_end, n - 1)
        # 1a pasada sin ajuste de suelo; medir malla
        act = escribir_accion(tgt, nombre, n, keys, locs, [0.0] * n, hips)
        por_cuadro = nombre in POR_CUADRO
        idx = list(range(n)) if por_cuadro else sorted(set(list(range(0, n, max(1, n // 16))) + [n - 1]))
        zs = {}
        for i in idx:
            sc.frame_set(i)
            zs[i] = zmin_malla(meshes)
        if por_cuadro:
            raw = [(-zs[i] if (zs[i] > tol or zs[i] < -0.01) else 0.0) for i in range(n)]
            dz = [sum(raw[max(0, i - 2):i + 3]) / len(raw[max(0, i - 2):i + 3]) for i in range(n)]
        else:
            zm = min(zs.values())
            dz = [(-zm if (zm < -0.01 or zm > tol) else 0.0)] * n
        act = escribir_accion(tgt, nombre, n, keys, locs, dz, hips)
        # tras corregir, medir de nuevo (informe)
        z2 = []
        for i in idx:
            sc.frame_set(i)
            z2.append(zmin_malla(meshes))
        info["clips"][nombre] = {"fuente": an, "cuadros": n, "zmin_antes_cm": round(min(zs.values()) * 100, 1),
                                 "zmin_despues_cm": [round(min(z2) * 100, 1), round(max(z2) * 100, 1)], "ajuste_z_cm": round(float(np.mean(dz)) * 100, 1)}
        acts[nombre] = act
        # costura de bucle (informativa)
        if nombre.endswith("_Loop"):
            seam = max(math.degrees(2 * math.acos(min(1.0, abs(keys[b][0].dot(keys[b][-1]))))) for b in keys)
            info["clips"][nombre]["costura_deg"] = round(seam, 2)
    tgt.animation_data.action = None
    # limpiar: fuera la fuente y todas las acciones de UAL (el exportador volcaria todas las acciones)
    for o in [o for o in bpy.data.objects if o.name.startswith("SRC_")]:
        bpy.data.objects.remove(o, do_unlink=True)
    for ac in list(bpy.data.actions):
        if ac.name not in acts:
            bpy.data.actions.remove(ac)
    sc.frame_set(0)
    return None, tgt, meshes, acts, info, ms, mt


def quitar_formas_hueso():
    """El importador de glTF crea objetos 'Icosphere' (forma de visualizacion de huesos, ~42 vertices) en una coleccion
    glTF_not_exported. Fuera: cualquier malla suelta (sin armadura ni piel) que no sea la del personaje; los nombres
    pueden estar truncados/renombrados, asi que se identifican por el uso como custom_shape y por no tener piel."""
    formas = set()
    for o in bpy.data.objects:
        if o.type == "ARMATURE":
            for b in o.pose.bones:
                if b.custom_shape:
                    formas.add(b.custom_shape.name); b.custom_shape = None
    for o in list(bpy.data.objects):
        if o.type != "MESH":
            continue
        sin_piel = not any(m.type == "ARMATURE" for m in o.modifiers)
        if o.name in formas or "Icosphere" in o.name or "Icosphere" in o.data.name or (sin_piel and o.parent is None):
            bpy.data.objects.remove(o, do_unlink=True)


def exportar(tgt, meshes, out):
    quitar_formas_hueso()
    meshes = [m for m in meshes if m.name in bpy.data.objects]
    bpy.ops.object.select_all(action="DESELECT")
    for o in [tgt] + meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = tgt
    bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=True, export_animations=True,
                              export_animation_mode="ACTIONS", export_skins=True, export_image_format="AUTO",
                              export_force_sampling=True, export_optimize_animation_size=False, export_yup=True)


if __name__ == "__main__":
    a = sys.argv[sys.argv.index("--") + 1:]

    def arg(n, d=None):
        return a[a.index(n) + 1] if n in a else d
    clips = CLIPS
    if arg("--clips"):
        clips = [tuple(x.split("=")) for x in arg("--clips").split(",")]
    _, tgt, meshes, acts, info, ms, mt = retarget(arg("--source", UAL1), arg("--target"), clips)
    out = arg("--out")
    if arg("--mapa-json"):
        json.dump({"fuente": ms, "destino": mt, "destino_a_fuente": info["mapa_hueso_destino_a_fuente"]}, open(arg("--mapa-json"), "w"), indent=1)
    exportar(tgt, meshes, out)
    if arg("--save-blend"):
        bpy.ops.wm.save_as_mainfile(filepath=arg("--save-blend"), copy=True)
    resumen = {k: v for k, v in info.items() if k != "mapa_hueso_destino_a_fuente"}
    json.dump(resumen, open(os.path.splitext(out)[0] + "_info.json", "w"), indent=1)
    print("@@RESUMEN@@" + json.dumps(resumen))
