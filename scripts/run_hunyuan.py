#!/usr/bin/env python
"""Hunyuan3D-2.1: carpeta de imagenes -> .glb texturizado PBR. Uso: python run_hunyuan.py [in_dir] [out_dir]"""
import os, sys, time, json, glob, gc
ROOT = "/root/hunyuan"
os.chdir(ROOT)
sys.path.insert(0, f"{ROOT}/hy3dshape"); sys.path.insert(0, f"{ROOT}/hy3dpaint")
sys.path.insert(0, f"{ROOT}/hy3dpaint/DifferentiableRenderer"); sys.path.insert(0, ROOT)
import torch
from PIL import Image
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
try:
    from torchvision_fix import apply_fix; apply_fix()
except Exception as e:
    print("torchvision_fix:", e)
import trimesh

IN = sys.argv[1] if len(sys.argv) > 1 else "/root/in"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/root/out/hunyuan"
os.makedirs(OUT, exist_ok=True)
STEPS, OCTREE, VIEWS, RES = 50, 512, 9, 768

shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
rembg = BackgroundRemover()
conf = Hunyuan3DPaintConfig(VIEWS, RES)
conf.realesrgan_ckpt_path = f"{ROOT}/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
conf.multiview_cfg_path = f"{ROOT}/hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
conf.custom_pipeline = f"{ROOT}/hy3dpaint/hunyuanpaintpbr"
paint = Hunyuan3DPaintPipeline(conf)

stats = []
for p in sorted(glob.glob(f"{IN}/*.png") + glob.glob(f"{IN}/*.jpg") + glob.glob(f"{IN}/*.jpeg") + glob.glob(f"{IN}/*.webp")):
    name = os.path.splitext(os.path.basename(p))[0]
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    img = Image.open(p).convert("RGBA")
    # quitar fondo solo si no hay transparencia real
    if img.getchannel("A").getextrema()[0] == 255:
        img = rembg(img.convert("RGB"))
    mesh = shape(image=img, num_inference_steps=STEPS, octree_resolution=OCTREE, num_chunks=200000,
                 generator=torch.manual_seed(1234))[0]
    raw = f"{OUT}/{name}_shape.glb"; mesh.export(raw)
    t1 = time.time(); vram_shape = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    out = paint(mesh_path=raw, image_path=p if False else img, output_mesh_path=f"{OUT}/{name}.glb")
    t2 = time.time(); vram_paint = torch.cuda.max_memory_allocated() / 2**30
    sc = trimesh.load(f"{OUT}/{name}.glb")
    faces = sum(g.faces.shape[0] for g in sc.geometry.values())
    tex = any(getattr(g.visual, "material", None) is not None and getattr(g.visual.material, "baseColorTexture", None) is not None for g in sc.geometry.values())
    s = dict(name=name, sec_shape=round(t1 - t0, 1), sec_paint=round(t2 - t1, 1), sec_total=round(t2 - t0, 1),
             vram_shape_gb=round(vram_shape, 1), vram_paint_gb=round(vram_paint, 1), faces=faces,
             size_mb=round(os.path.getsize(f"{OUT}/{name}.glb") / 2**20, 1), textured=tex)
    print(json.dumps(s), flush=True); stats.append(s)
    gc.collect(); torch.cuda.empty_cache()
json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=1)
