"""Horneado alta -> baja: color base, normal map (tangent, OpenGL = convencion glTF),
AO, roughness y metallic. Empaca ORM (R=AO, G=roughness, B=metallic) y exporta GLB.

Uso:
  Blender -b -P bake.py -- alta.glb baja.(glb|blend) salida.glb --tex 2048
Si `baja` es .blend se toma el objeto 'low' (conserva quads) ; si es .glb se importa.
"""
import bpy, sys, os, time, argparse
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import remesh_quads as R


def parse(argv):
    a = argv[argv.index('--') + 1:] if '--' in argv else argv
    p = argparse.ArgumentParser()
    p.add_argument('alta'); p.add_argument('baja'); p.add_argument('salida')
    p.add_argument('--tex', type=int, default=2048)
    p.add_argument('--extrusion', type=float, default=0.012, help='fraccion de la diagonal')
    p.add_argument('--dist', type=float, default=0.03, help='max ray distance, fraccion de la diagonal')
    p.add_argument('--muestras_ao', type=int, default=48)
    return p.parse_args(a)


def setup_cycles(samples):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    dev = 'CPU'
    try:
        pr = bpy.context.preferences.addons['cycles'].preferences
        pr.compute_device_type = 'METAL'
        pr.get_devices()
        gpus = [d for d in pr.devices if d.type == 'METAL']
        for d in pr.devices:
            d.use = d.type == 'METAL'
        if gpus:
            sc.cycles.device = 'GPU'; dev = 'METAL'
    except Exception as e:
        print('[bake] sin Metal:', e)
    sc.cycles.samples = samples
    sc.cycles.use_denoising = False
    sc.cycles.use_adaptive_sampling = False
    print('[bake] dispositivo', dev)
    return dev


def load_high(path):
    return R.load_join(path)


def load_low(path):
    if path.lower().endswith('.blend'):
        bpy.ops.wm.append(directory=path + '/Object/', filename='low')
        return bpy.data.objects['low']
    ob = R.load_join(path)
    ob.name = 'low'
    return ob


def principled(mat):
    for n in mat.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            return n
    return None


class EmitSwap:
    """Cambia temporalmente los materiales de la alta para hornear una entrada del
    Principled (Base Color/Metallic/Roughness) como Emission: evita que el
    pase DIFFUSE oscurezca el color de las partes metalicas."""
    def __init__(self, high, inp):
        self.saved = []; self.high = high; self.inp = inp

    def __enter__(self):
        for mat in self.high.data.materials:
            nt = mat.node_tree; pb = principled(mat)
            out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL' and n.is_active_output) \
                if any(n.type == 'OUTPUT_MATERIAL' and n.is_active_output for n in nt.nodes) \
                else next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
            old = out.inputs['Surface'].links[0].from_socket if out.inputs['Surface'].links else None
            em = nt.nodes.new('ShaderNodeEmission')
            sock = pb.inputs[self.inp]
            if sock.is_linked:
                nt.links.new(sock.links[0].from_socket, em.inputs['Color'])
            else:
                v = sock.default_value
                em.inputs['Color'].default_value = tuple(v) if hasattr(v, '__len__') else (v, v, v, 1.0)
            nt.links.new(em.outputs['Emission'], out.inputs['Surface'])
            self.saved.append((nt, out, old, em))
        return self

    def __exit__(self, *a):
        for nt, out, old, em in self.saved:
            if old: nt.links.new(old, out.inputs['Surface'])
            nt.nodes.remove(em)


def new_image(name, size, color, nc, alpha=False):
    img = bpy.data.images.new(name, size, size, alpha=alpha, float_buffer=False)
    img.generated_color = color
    img.colorspace_settings.name = 'Non-Color' if nc else 'sRGB'
    return img


def set_target(low, img):
    mat = low.data.materials[0]
    nt = mat.node_tree
    node = nt.nodes.new('ShaderNodeTexImage'); node.image = img
    for n in nt.nodes: n.select = False
    node.select = True
    nt.nodes.active = node
    print('[bake] nodo activo', nt.nodes.active.name, nt.nodes.active.type, node.image.name, 'mats', [m.name for m in low.data.materials], 'slot', low.active_material_index, 'uv', [u.name for u in low.data.uv_layers])
    return node


def do_bake(high, low, btype, img, a, ext, dist, margin, **kw):
    sc = bpy.context.scene
    bpy.ops.object.select_all(action='DESELECT')
    high.select_set(True); low.select_set(True)
    bpy.context.view_layer.objects.active = low
    node = set_target(low, img)
    b = sc.render.bake
    b.use_selected_to_active = True; b.use_cage = False
    b.cage_extrusion = ext; b.max_ray_distance = dist
    b.margin = margin; b.margin_type = 'EXTEND'
    b.use_clear = True
    b.target = 'IMAGE_TEXTURES'
    t = time.time()
    bpy.ops.object.bake(type=btype, **kw)
    return time.time() - t


def bake_all(high, low, tex, extrusion=0.012, dist=0.03, muestras_ao=48):
    """Devuelve (imagenes, tiempos). `low` debe tener UVs; se le crea un material."""
    log = {}
    d = R.diag(low)
    ext, mrd = extrusion * d, dist * d
    dev = setup_cycles(1)
    log['dispositivo'] = dev
    mat = bpy.data.materials.new('baked'); mat.use_nodes = True
    low.data.materials.clear(); low.data.materials.append(mat)
    mar = max(8, tex // 128)
    sc = bpy.context.scene
    imgs = {}; tm = {}
    # color base
    imgs['base'] = new_image('base_color', tex, (0, 0, 0, 1), False)
    sc.cycles.samples = 1
    with EmitSwap(high, 'Base Color'):
        tm['base'] = do_bake(high, low, 'EMIT', imgs['base'], None, ext, mrd, mar)
    # normal
    imgs['normal'] = new_image('normal', tex, (0.5, 0.5, 1, 1), True)
    sc.render.bake.normal_space = 'TANGENT'
    sc.render.bake.normal_r = 'POS_X'; sc.render.bake.normal_g = 'POS_Y'; sc.render.bake.normal_b = 'POS_Z'
    tm['normal'] = do_bake(high, low, 'NORMAL', imgs['normal'], None, ext, mrd, mar)
    # roughness / metallic
    imgs['rough'] = new_image('rough', tex, (0.5, 0.5, 0.5, 1), True)
    with EmitSwap(high, 'Roughness'):
        tm['rough'] = do_bake(high, low, 'EMIT', imgs['rough'], None, ext, mrd, mar)
    imgs['metal'] = new_image('metal', tex, (0, 0, 0, 1), True)
    with EmitSwap(high, 'Metallic'):
        tm['metal'] = do_bake(high, low, 'EMIT', imgs['metal'], None, ext, mrd, mar)
    # AO
    imgs['ao'] = new_image('ao', tex, (1, 1, 1, 1), True)
    sc.cycles.samples = muestras_ao
    w = bpy.data.worlds.new('w') if not sc.world else sc.world
    sc.world = w
    w.light_settings.distance = 0.15 * d
    tm['ao'] = do_bake(high, low, 'AO', imgs['ao'], None, ext, mrd, mar)
    log['seg_por_pase'] = {k: round(v, 1) for k, v in tm.items()}
    log['seg_bake'] = round(sum(tm.values()), 1)
    return imgs, log


def pack_orm(imgs, tex):
    def px(img):
        a = np.empty(tex * tex * 4, np.float32); img.pixels.foreach_get(a); return a.reshape(-1, 4)
    ao, ro, me = px(imgs['ao']), px(imgs['rough']), px(imgs['metal'])
    out = np.ones((tex * tex, 4), np.float32)
    out[:, 0] = ao[:, 0]; out[:, 1] = ro[:, 0]; out[:, 2] = me[:, 0]
    img = new_image('orm', tex, (1, 1, 1, 1), True)
    img.pixels.foreach_set(out.ravel()); img.update()
    return img


def build_final_material(low, imgs, orm):
    mat = low.data.materials[0]
    nt = mat.node_tree
    for n in list(nt.nodes):
        if n.type != 'OUTPUT_MATERIAL': nt.nodes.remove(n)
    out = nt.nodes['Material Output']
    pb = nt.nodes.new('ShaderNodeBsdfPrincipled')
    nt.links.new(pb.outputs['BSDF'], out.inputs['Surface'])
    def tx(img):
        n = nt.nodes.new('ShaderNodeTexImage'); n.image = img; return n
    nt.links.new(tx(imgs['base']).outputs['Color'], pb.inputs['Base Color'])
    nm = nt.nodes.new('ShaderNodeNormalMap'); nm.space = 'TANGENT'
    nm.uv_map = low.data.uv_layers[0].name
    nt.links.new(tx(imgs['normal']).outputs['Color'], nm.inputs['Color'])
    nt.links.new(nm.outputs['Normal'], pb.inputs['Normal'])
    t = tx(orm)
    sep = nt.nodes.new('ShaderNodeSeparateColor')
    nt.links.new(t.outputs['Color'], sep.inputs['Color'])
    nt.links.new(sep.outputs['Green'], pb.inputs['Roughness'])
    nt.links.new(sep.outputs['Blue'], pb.inputs['Metallic'])
    # AO: el exportador glTF lee el grupo "glTF Material Output"
    ng = bpy.data.node_groups.new('glTF Material Output', 'ShaderNodeTree')
    ng.interface.new_socket('Occlusion', in_out='INPUT', socket_type='NodeSocketFloat')
    g = nt.nodes.new('ShaderNodeGroup'); g.node_tree = ng
    nt.links.new(sep.outputs['Red'], g.inputs['Occlusion'])


def export_glb(low, path):
    bpy.ops.object.select_all(action='DESELECT')
    low.select_set(True); bpy.context.view_layer.objects.active = low
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', use_selection=True,
                              export_materials='EXPORT', export_image_format='AUTO',
                              export_yup=True)


def run(high, low, salida, tex, **kw):
    imgs, log = bake_all(high, low, tex, **kw)
    orm = pack_orm(imgs, tex)
    bpy.data.objects.remove(high, do_unlink=True)
    build_final_material(low, imgs, orm)
    for im in list(imgs.values()) + [orm]: im.pack()
    export_glb(low, salida)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.splitext(salida)[0] + '_quads.blend', copy=True)
    log['glb_MB'] = round(os.path.getsize(salida) / 1e6, 1)
    return log


def main():
    a = parse(sys.argv)
    R.reset()
    high = load_high(a.alta)
    low = load_low(a.baja)
    log = run(high, low, a.salida, a.tex, extrusion=a.extrusion, dist=a.dist, muestras_ao=a.muestras_ao)
    print('[bake] OK', log)


if __name__ == '__main__':
    main()
