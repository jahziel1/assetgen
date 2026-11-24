#!/usr/bin/env python
"""Uso: python run_trellis2.py [carpeta_in=/root/in] [carpeta_out=/root/out/trellis2]
Envolver con: flock /root/gpu.lock python /root/run_trellis2.py
Env opcionales: DINOV3_MODEL, REMBG_MODEL, PIPELINE_TYPE (512|1024_cascade|1536_cascade), TEX_SIZE, DECIM,
SEEDS ("42,7": una salida <nombre>_s<semilla>.glb por semilla; sin SEEDS: <nombre>.glb con semilla 42),
MULTIVISTA=1 (EXPERIMENTAL; TRELLIS.2 NO trae multi-imagen oficial: para <n>.png con <n>_lado.png y <n>_espalda.png se concatenan los
tokens DINOv3 de las 3 vistas como condicion de la atencion cruzada y se escribe <n>_multi.glb)."""
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
VISTAS = ('_lado', '_espalda')
SEEDS = [int(x) for x in os.environ.get('SEEDS', '').split(',') if x.strip()]
MULTI = os.environ.get('MULTIVISTA', '') not in ('', '0')
todas = sorted(f for f in glob.glob(os.path.join(IN, '*')) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
_stem = lambda f: os.path.splitext(os.path.basename(f))[0]
files = [f for f in todas if not (MULTI and _stem(f).endswith(VISTAS))]
_ord=[x for x in os.environ.get("ORDER","").split(",") if x]
if _ord: files.sort(key=lambda f: _ord.index(_stem(f)) if _stem(f) in _ord else 99)
if os.environ.get('LOWVRAM') == '0': pipe.low_vram = False   # todo en GPU, sin offload entre etapas (48 GB)

_get_cond0 = pipe.get_cond
_vistas = None   # lista de imagenes ya preprocesadas (frente, lado, espalda) mientras se genera el multi
def _get_cond_multi(image, resolution, include_neg_cond=True):
    if _vistas is None:
        return _get_cond0(image, resolution, include_neg_cond)
    c = _get_cond0(_vistas, resolution, include_neg_cond=False)['cond']   # (V, N, D) -> (1, V*N, D)
    c = c.reshape(1, -1, c.shape[-1])
    return {'cond': c, 'neg_cond': torch.zeros_like(c)}
pipe.get_cond = _get_cond_multi

def generar(name, img, seed, vistas=None):
    global _vistas
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
    if vistas:
        _vistas = [pipe.preprocess_image(v) for v in vistas]
        mesh = pipe.run(_vistas[0], seed=seed, pipeline_type=PT, preprocess_image=False)[0]
    else:
        mesh = pipe.run(img, seed=seed, pipeline_type=PT)[0]
    _vistas = None
    res = round(1 / mesh.voxel_size)
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
    sc = trimesh.load(p, force='scene'); nf = sum(len(g.faces) for g in sc.geometry.values()); nv = sum(len(g.vertices) for g in sc.geometry.values())
    s = dict(name=name, seed=seed, seconds=round(dt, 1), vram_peak_gb=round(torch.cuda.max_memory_allocated()/2**30, 2),
             faces=nf, vertices=nv, decimation_target=dec, pipeline_type=PT, resolucion_voxel=res, texture_size=TEX, multivista=bool(vistas),
             textures=sum(1 for g in sc.geometry.values() if getattr(g.visual,'material',None) is not None and getattr(g.visual.material,'baseColorTexture',None) is not None), size_mb=round(os.path.getsize(p)/2**20, 1))
    print(json.dumps(s), flush=True); stats.append(s)
    if os.environ.get("WRITE_READY"): open(os.path.join(OUT, name + ".ready"), "w").write("ok\n")
    json.dump(stats, open(os.environ.get("STATS_PATH", os.path.join(OUT, "stats.json")), "w"), indent=1)
    del mesh, glb; torch.cuda.empty_cache()

for f in files:
    name = _stem(f); img = Image.open(f)
    for seed in (SEEDS or [42]):
        try:
            generar(f"{name}_s{seed}" if SEEDS else name, img, seed)
        except torch.cuda.OutOfMemoryError:   # 1536 en GPU de 24 GB: baja a 1024_cascade en vez de perder la corrida
            print(f'{name}: OOM con {PT}; reintento con 1024_cascade', flush=True)
            torch.cuda.empty_cache(); PT = '1024_cascade'
            generar(f"{name}_s{seed}" if SEEDS else name, img, seed)
    if MULTI:
        ex = [os.path.join(IN, name + v + e) for v in VISTAS for e in ('.png', '.jpg', '.jpeg', '.webp') if os.path.exists(os.path.join(IN, name + v + e))]
        if len(ex) == len(VISTAS):
            try:
                generar(f"{name}_multi", img, (SEEDS or [42])[0], vistas=[img] + [Image.open(x) for x in ex])
            except Exception as e:   # experimental: que un fallo no tire las salidas ya hechas
                import traceback; traceback.print_exc(); _vistas = None; torch.cuda.empty_cache()
                print(f'{name}: multivista FALLO: {e!r}', flush=True)
        else:
            print(f'{name}: multivista omitida (faltan {name}_lado / {name}_espalda)', flush=True)
json.dump(stats, open(os.environ.get('STATS_PATH', os.path.join(OUT, 'stats.json')), 'w'), indent=1)
