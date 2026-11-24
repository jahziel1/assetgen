# glb texturizado -> carpeta con mesh.obj (+mtl+png) conservando orden de vertices
import sys, os, trimesh, numpy as np
glb, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
sc = trimesh.load(glb, process=False)
geoms = list(sc.geometry.values()) if isinstance(sc, trimesh.Scene) else [sc]
print("geoms", len(geoms), [ (g.vertices.shape, g.faces.shape, type(g.visual).__name__) for g in geoms])
if isinstance(sc, trimesh.Scene):
    m = sc.dump(concatenate=True) if len(geoms)>1 else geoms[0]
    if len(geoms)==1:
        m = geoms[0]; 
        # aplicar transformaciones del grafo
        m = sc.dump()[0]
else: m = sc
print(m.vertices.shape, m.faces.shape, m.bounds)
m.export(os.path.join(outdir, "mesh.obj"), include_texture=True, write_texture=True)
print(os.listdir(outdir))
