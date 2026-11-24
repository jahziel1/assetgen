#!/usr/bin/env python
"""Uso: python run_trellis2.py [carpeta_in=/root/in] [carpeta_out=/root/out/trellis2]
Envolver con: flock /root/gpu.lock python /root/run_trellis2.py
Env opcionales: DINOV3_MODEL, REMBG_MODEL, PIPELINE_TYPE (1024_cascade|1536_cascade), TEX_SIZE, DECIM."""
import os, sys, time, glob, json
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.chdir('/root/trellis2'); sys.path.insert(0, '/root/trellis2')
import torch, trimesh
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

IN = sys.argv[1] if len(sys.argv) > 1 else '/root/in'
OUT = sys.argv[2] if len(sys.argv) > 2 else '/root/out/trellis2'
PT = os.environ.get('PIPELINE_TYPE', '1024_cascade')
TEX = int(os.environ.get('TEX_SIZE', 4096)); DECIM = int(os.environ.get('DECIM', 1000000))
os.makedirs(OUT, exist_ok=True)

# DINOv3 oficial (facebook/...) es gated: usamos la conversion local de los pesos publicos de timm
# (ver /root/convert_dinov3_timm_to_hf.py) y BiRefNet publico en vez de briaai/RMBG-2.0 (gated).
DINO = os.environ.get('DINOV3_MODEL', '/root/dinov3-vitl16-hf'); REMBG = os.environ.get('REMBG_MODEL', 'ZhengPeng7/BiRefNet')
from trellis2.modules import image_feature_extractor as _ife
from trellis2.pipelines import rembg as _rb
_o1 = _ife.DinoV3FeatureExtractor.__init__
_ife.DinoV3FeatureExtractor.__init__ = lambda self, model_name, **kw: _o1(self, DINO, **kw)
_o2 = _rb.BiRefNet.__init__
_rb.BiRefNet.__init__ = lambda self, model_name=None, **kw: _o2(self, REMBG, **kw)
MAXMB = float(os.environ.get('MAX_GLB_MB', 50)); DECIM_FALLBACK = int(os.environ.get('DECIM_FALLBACK', 200000))

t0 = time.time()
pipe = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipe.cuda()
print(f"carga: {time.time()-t0:.1f}s", flush=True)

stats = []
files = sorted(f for f in glob.glob(os.path.join(IN, '*')) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
_ord=[x for x in os.environ.get("ORDER","").split(",") if x]
if _ord: files.sort(key=lambda f: _ord.index(os.path.splitext(os.path.basename(f))[0]) if os.path.splitext(os.path.basename(f))[0] in _ord else 99)
for f in files:
    name = os.path.splitext(os.path.basename(f))[0]
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
    img = Image.open(f)
    mesh = pipe.run(img, pipeline_type=PT)[0]
    mesh.simplify(16777216)
    p = os.path.join(OUT, name + '.glb'); dec = DECIM
    for attempt in range(2):
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
            attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=dec, texture_size=TEX, remesh=True, remesh_band=1, remesh_project=0, verbose=False)
        glb.export(p, extension_webp=True)
        if os.path.getsize(p) / 2**20 <= MAXMB or attempt == 1: break
        print(f'{name}: glb {os.path.getsize(p)/2**20:.0f} MB > {MAXMB}, redecimando a {DECIM_FALLBACK}', flush=True); dec = DECIM_FALLBACK
    torch.cuda.synchronize(); dt = time.time() - t
    sc = trimesh.load(p, force='scene'); nf = sum(len(g.faces) for g in sc.geometry.values())
    s = dict(name=name, seconds=round(dt, 1), vram_peak_gb=round(torch.cuda.max_memory_allocated()/2**30, 2),
             faces=nf, decimation_target=dec, textures=sum(1 for g in sc.geometry.values() if getattr(g.visual,'material',None) is not None and getattr(g.visual.material,'baseColorTexture',None) is not None), size_mb=round(os.path.getsize(p)/2**20, 1))
    print(json.dumps(s), flush=True); stats.append(s)
    if os.environ.get("WRITE_READY"): open(os.path.join(OUT, name + ".ready"), "w").write("ok\n")
    json.dump(stats, open(os.environ.get("STATS_PATH", os.path.join(OUT, "stats.json")), "w"), indent=1)
    del mesh, glb; torch.cuda.empty_cache()
json.dump(stats, open(os.environ.get('STATS_PATH', os.path.join(OUT, 'stats.json')), 'w'), indent=1)
