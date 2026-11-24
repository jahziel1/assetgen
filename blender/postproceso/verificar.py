"""Reimporta el GLB final y comprueba normal map, ORM/AO, caras y quads (del .blend)."""
import bpy, sys, os, json
glb = sys.argv[-1]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
ob = [o for o in bpy.data.objects if o.type == 'MESH'][0]
mat = ob.data.materials[0]; nt = mat.node_tree
nm = [n for n in nt.nodes if n.type == 'NORMAL_MAP']
pb = [n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'][0]
res = dict(glb=os.path.basename(glb), caras_tris_glb=len(ob.data.polygons),
           normal_map_nodo=len(nm) == 1, normal_conectado=pb.inputs['Normal'].is_linked,
           normal_img=nm[0].inputs['Color'].links[0].from_node.image.size[:] if nm and nm[0].inputs['Color'].is_linked else None,
           base_color=pb.inputs['Base Color'].is_linked,
           rough_metal_desde_ORM=pb.inputs['Roughness'].is_linked and pb.inputs['Metallic'].is_linked,
           occlusion_en_glb=any(n.type == 'GROUP' for n in nt.nodes))
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.open_mainfile(filepath=os.path.splitext(glb)[0] + '_quads.blend')
low = bpy.data.objects['low']; me = low.data
q = sum(1 for p in me.polygons if len(p.vertices) == 4)
res.update(caras_quads_blend=len(me.polygons), pct_quads=round(100 * q / len(me.polygons), 1))
print('VERIF', json.dumps(res))
