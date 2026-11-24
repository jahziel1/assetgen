"""Remesh a quads con QuadriFlow (con alternativas) + UVs nuevos.

Uso:
  Blender -b -P remesh_quads.py -- entrada.glb salida.glb --caras 20000 [--margen 0.004]
Salidas: salida.glb (triangulado, glTF no guarda quads) y salida.blend (quads reales).
Metodos (se prueban en orden hasta que uno da una malla valida):
  1) quadriflow directo sobre la malla limpia
  2) voxel remesh (hermetica/manifold) + quadriflow
  3) voxel remesh + decimate + tris-a-quads
"""
import bpy, bmesh, sys, time, os, argparse
from mathutils import Vector


def parse(argv):
    a = argv[argv.index('--') + 1:] if '--' in argv else argv
    p = argparse.ArgumentParser()
    p.add_argument('entrada'); p.add_argument('salida')
    p.add_argument('--caras', type=int, default=20000)
    p.add_argument('--margen', type=float, default=0.004)
    p.add_argument('--metodo', default='auto', choices=['auto', 'quadriflow', 'voxel_qf', 'voxel_dec'])
    return p.parse_args(a)


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def load_join(path):
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    bpy.ops.object.select_all(action='DESELECT')
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    for o in meshes:   # aplanar jerarquia
        o.parent = None
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if len(meshes) > 1:
        bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    return ob


def strip(ob):
    me = ob.data
    while me.uv_layers:
        me.uv_layers.remove(me.uv_layers[0])
    me.materials.clear()
    for a in list(me.color_attributes):
        me.color_attributes.remove(a)
    me.polygons.foreach_set('use_smooth', [True] * len(me.polygons))


def diag(ob):
    bb = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
    mn = Vector((min(v[i] for v in bb) for i in range(3)))
    mx = Vector((max(v[i] for v in bb) for i in range(3)))
    return (mx - mn).length


def clean(ob, log):
    """Une vertices cercanos, quita islas diminutas, repara."""
    d = diag(ob)
    bm = bmesh.new(); bm.from_mesh(ob.data)
    n0 = len(bm.faces)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=d * 2e-5)
    bmesh.ops.dissolve_degenerate(bm, dist=d * 1e-6, edges=bm.edges)
    bm.faces.ensure_lookup_table()
    # islas
    seen = set(); islands = []
    for f in bm.faces:
        if f.index in seen: continue
        st = [f]; seen.add(f.index); isl = []
        while st:
            x = st.pop(); isl.append(x)
            for e in x.edges:
                for g in e.link_faces:
                    if g.index not in seen:
                        seen.add(g.index); st.append(g)
        islands.append(isl)
    tot_area = sum(f.calc_area() for f in bm.faces)
    big = max(islands, key=len)
    kill = []
    for isl in islands:
        if isl is big: continue
        ar = sum(f.calc_area() for f in isl)
        if ar < 0.002 * tot_area and len(isl) < 0.01 * n0:
            kill.extend(isl)
    if kill:
        bmesh.ops.delete(bm, geom=kill, context='FACES')
    # vertices sueltos / aristas sueltas
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    nm = sum(1 for e in bm.edges if not e.is_manifold)
    bm.to_mesh(ob.data); bm.free()
    log['limpieza'] = dict(caras_antes=n0, caras_despues=len(ob.data.polygons), islas=len(islands),
                           caras_borradas_islas=len(kill), aristas_no_manifold=nm)
    print('[limpieza]', log['limpieza'])


def stats(ob):
    me = ob.data
    q = sum(1 for p in me.polygons if len(p.vertices) == 4)
    t = sum(1 for p in me.polygons if len(p.vertices) == 3)
    o = len(me.polygons) - q - t
    return dict(caras=len(me.polygons), quads=q, tris=t, ngons=o,
                pct_quads=round(100 * q / max(1, len(me.polygons)), 1))


def activate(ob):
    bpy.ops.object.select_all(action='DESELECT')
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob


def quadriflow(ob, faces):
    """QuadriFlow apunta al numero de caras pero suele quedarse ~10-15% corto:
    se reintenta una vez corrigiendo el objetivo."""
    activate(ob)
    base = ob.data.copy()
    tgt = faces; pts = []; best = None
    for i in range(3):
        ob.data = base.copy()
        r = bpy.ops.object.quadriflow_remesh(
            mode='FACES', target_faces=int(tgt), use_mesh_symmetry=False,
            use_preserve_sharp=False, use_preserve_boundary=False,
            smooth_normals=True, seed=0)
        if r != {'FINISHED'}:
            return False
        n = len(ob.data.polygons); pts.append((tgt, n))
        print('[quadriflow] intento', i, 'objetivo', int(tgt), '->', n)
        if best is None or abs(n - faces) < abs(best[0] - faces):
            best = (n, ob.data.copy())
        if abs(n - faces) / faces < 0.08:
            break
        if len(pts) >= 2 and pts[-1][1] != pts[-2][1]:   # secante
            (t1, n1), (t2, n2) = pts[-2], pts[-1]
            tgt = t2 + (faces - n2) * (t2 - t1) / (n2 - n1)
        else:
            tgt = tgt * faces / n * 1.05
        tgt = max(faces * 0.5, min(faces * 2.5, tgt))
    ob.data = best[1]
    return True


def es_manifold(ob):
    bm = bmesh.new(); bm.from_mesh(ob.data)
    nm = sum(1 for e in bm.edges if not e.is_manifold or len(e.link_faces) != 2)
    bm.free()
    return nm == 0


def voxel(ob, d, faces):
    # tamano de voxel: lo bastante fino para conservar detalle sin cerrar espacios finos
    vs = d / 350.0
    ob.data.remesh_voxel_size = vs
    ob.data.remesh_voxel_adaptivity = 0.0
    activate(ob)
    bpy.ops.object.voxel_remesh()
    return len(ob.data.polygons)


def decimate_to(ob, tris_target):
    activate(ob)
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris(); bpy.ops.object.mode_set(mode='OBJECT')
    n = len(ob.data.polygons)
    m = ob.modifiers.new('dec', 'DECIMATE'); m.ratio = min(1.0, tris_target / n)
    bpy.ops.object.modifier_apply(modifier=m.name)
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.tris_convert_to_quads(face_threshold=1.0, shape_threshold=1.0)
    bpy.ops.object.mode_set(mode='OBJECT')


def valid(ob, faces):
    n = len(ob.data.polygons)
    return 0.5 * faces <= n <= 1.6 * faces


def count_islands(ob):
    """Numero de islas UV (por conectividad de caras que comparten UV en sus aristas)."""
    me = ob.data; uv = me.uv_layers.active.data
    parent = list(range(len(me.polygons)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    edge_loops = {}
    for p in me.polygons:
        for li in p.loop_indices:
            l = me.loops[li]; e = l.edge_index
            a = tuple(round(c, 5) for c in uv[li].uv)
            nxt = p.loop_indices[(li - p.loop_start + 1) % p.loop_total]
            b = tuple(round(c, 5) for c in uv[nxt].uv)
            edge_loops.setdefault(e, []).append((p.index, frozenset((a, b))))
    for e, lst in edge_loops.items():
        if len(lst) == 2 and lst[0][1] == lst[1][1]:
            parent[find(lst[0][0])] = find(lst[1][0])
    return len({find(i) for i in range(len(me.polygons))})


def smart_uv(ob, margen, angulo=1.15, suavizado=10):
    """Smart UV Project calculado sobre una copia SUAVIZADA de la malla (las
    normales ruidosas de una malla de IA remallada la rompen en miles de islas) y
    trasladado a la malla real: la topologia es identica."""
    me0 = ob.data
    smooth = me0.copy()
    ob.data = smooth
    activate(ob)
    bpy.ops.object.mode_set(mode='EDIT'); bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.vertices_smooth(factor=1.0, repeat=suavizado)
    bpy.ops.uv.smart_project(angle_limit=angulo, island_margin=margen, area_weight=0.6,
                             correct_aspect=True, scale_to_bounds=False)
    bpy.ops.mesh.select_all(action='SELECT'); bpy.ops.uv.select_all(action='SELECT')
    bpy.ops.uv.pack_islands(margin=margen, rotate=True)
    bpy.ops.object.mode_set(mode='OBJECT')
    # copiar UVs a la malla original (mismo orden de loops)
    import numpy as np
    n = len(smooth.loops)
    arr = np.empty(n * 2, np.float32)
    smooth.uv_layers.active.data.foreach_get('uv', arr)
    while me0.uv_layers: me0.uv_layers.remove(me0.uv_layers[0])
    nl = me0.uv_layers.new(name='UVMap')
    nl.data.foreach_set('uv', arr)
    ob.data = me0
    bpy.data.meshes.remove(smooth)


def remesh(ob, faces, metodo='auto', margen=0.004):
    """Devuelve (ob, log). ob queda con quads y UVs nuevos."""
    log = {}
    t0 = time.time()
    strip(ob); clean(ob, log)
    # QuadriFlow falla ("mesh needs to be manifold") con aristas < ~1e-4 aunque la malla
    # sea manifold: se normaliza el tamano a diag=200 durante el remesh y se revierte al final.
    d0 = diag(ob); k = 200.0 / d0
    ob.scale = (k, k, k); activate(ob); bpy.ops.object.transform_apply(scale=True)
    d = diag(ob)
    base_me = ob.data.copy()          # copia limpia (ya escalada) para reintentos
    log['manifold_tras_limpieza'] = es_manifold(ob)
    orden = ['quadriflow', 'voxel_qf', 'voxel_dec'] if metodo == 'auto' else [metodo]
    if metodo == 'auto' and not log['manifold_tras_limpieza']:
        orden = orden[1:]   # QuadriFlow directo no puede (aristas no manifold)
    usado = None; intentos = []
    for m in orden:
        ob.data = base_me.copy()
        ts = time.time()
        try:
            if m == 'quadriflow':
                ok = quadriflow(ob, faces)
            elif m == 'voxel_qf':
                voxel(ob, d, faces); ok = quadriflow(ob, faces)
            else:
                voxel(ob, d, faces); decimate_to(ob, faces * 2); ok = True
            ok = ok and valid(ob, faces)
            if ok and m != 'voxel_dec': ok = stats(ob)['pct_quads'] > 80
        except Exception as e:
            print('[remesh] fallo', m, e); ok = False
        intentos.append(dict(metodo=m, ok=bool(ok), seg=round(time.time() - ts, 1),
                             caras=len(ob.data.polygons)))
        print('[remesh]', intentos[-1])
        if ok:
            usado = m; break
    if not usado:
        raise RuntimeError('ningun metodo de remesh funciono: %s' % intentos)
    log['metodo'] = usado; log['intentos'] = intentos
    ob.data.polygons.foreach_set('use_smooth', [True] * len(ob.data.polygons))
    ob.data.update()
    t1 = time.time()
    smart_uv(ob, margen)
    ob.scale = (1 / k, 1 / k, 1 / k); activate(ob); bpy.ops.object.transform_apply(scale=True)
    log['seg_remesh'] = round(t1 - t0, 1)
    log['seg_uv'] = round(time.time() - t1, 1)
    log['islas_uv'] = count_islands(ob)
    log['stats'] = stats(ob)
    print('[remesh] resultado', log)
    return ob, log


def export_glb(ob, path):
    activate(ob)
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', use_selection=True,
                              export_apply=False, export_materials='NONE', export_yup=True)


def main():
    a = parse(sys.argv)
    reset()
    ob = load_join(a.entrada)
    ob, log = remesh(ob, a.caras, a.metodo, a.margen)
    ob.name = 'low'
    base = os.path.splitext(a.salida)[0]
    bpy.ops.wm.save_as_mainfile(filepath=base + '.blend', copy=True)
    export_glb(ob, a.salida)
    print('[remesh] OK', a.salida, log['stats'])


if __name__ == '__main__':
    main()
