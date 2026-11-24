#!/usr/bin/env python
"""Editor de imagenes Qwen-Image-Edit-2511 (multi-imagen) + LoRA Lightning 8 pasos, para `assetgen remaster concepto`.

Uso:  /root/venv-imagen/bin/python gen_concepto.py plan.json --in DIR --out DIR [--force]
plan.json: {"model": "Qwen/Qwen-Image-Edit-2511", "steps": 8, "width": 768, "height": 1328,
            "jobs": [{"id": "v1_frente", "seed": 11, "prompt": "...", "imagenes": [{"ref": "vistas/frente.png"}, {"gen": "v0"}]}]}
  {"ref": ruta relativa a --in}  imagen de entrada;  {"gen": id}  salida de otro trabajo (OUT/<id>.png, o IN/gen/<id>.png si ya existia).
Los trabajos se ejecutan por rondas: primero los que no dependen de ninguna salida generada, luego los que si.
Limite de RAM del cgroup (~50 GB) y de VRAM (48 GB): text encoder (16 GB) y transformer (41 GB) no caben juntos, asi que cada ronda
codifica sus prompts (con las imagenes de condicion) con el text encoder en GPU, lo libera, y carga el transformer directo a GPU.
Salida: OUT/<id>.png y OUT/stats.json (carga, segundos por imagen, VRAM pico)."""
import argparse, gc, json, math, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("plan")
ap.add_argument("--in", dest="indir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

plan = json.load(open(args.plan, encoding="utf-8"))
REPO = plan.get("model", "Qwen/Qwen-Image-Edit-2511")
STEPS = int(plan.get("steps", 8))
W, H = int(plan.get("width", 768)), int(plan.get("height", 1328))
assert W % 16 == 0 and H % 16 == 0
os.makedirs(args.out, exist_ok=True)
jobs = [j for j in plan["jobs"] if args.force or not os.path.exists(f"{args.out}/{j['id']}.png")]
if not jobs:
    print("nada que generar"); sys.exit(0)

import torch
from PIL import Image
from diffusers import DiffusionPipeline, QwenImageEditPlusPipeline, QwenImageTransformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions, CONDITION_IMAGE_SIZE

def cargar(im):
    if "ref" in im:
        return Image.open(os.path.join(args.indir, im["ref"])).convert("RGB")
    for d in (args.out, os.path.join(args.indir, "gen")):
        f = os.path.join(d, im["gen"] + ".png")
        if os.path.exists(f):
            return Image.open(f).convert("RGB")
    raise FileNotFoundError(im)

def listo(j, hechos):
    return all("ref" in im or im["gen"] in hechos or os.path.exists(os.path.join(args.indir, "gen", im["gen"] + ".png")) for im in j["imagenes"])

def rondas():
    pend, hechos, res = list(jobs), set(), []
    while pend:
        r = [j for j in pend if listo(j, hechos)]
        if not r:
            raise SystemExit("dependencias circulares o salidas que faltan: " + ", ".join(j["id"] for j in pend))
        res.append(r); hechos |= {j["id"] for j in r}; pend = [j for j in pend if j not in r]
    return res

tocpu = lambda v: tuple(None if x is None else x.cpu() for x in v)
stats, t_all = [], time.time()
tr = None
for n_ronda, ronda in enumerate(rondas(), 1):
    # fase 1: embeddings de texto+imagenes de condicion (text encoder en GPU; luego se libera)
    t1 = time.time()
    p1 = DiffusionPipeline.from_pretrained(REPO, transformer=None, vae=None, torch_dtype=torch.bfloat16)
    p1.text_encoder.to("cuda")
    emb = {}
    with torch.no_grad():
        for j in ronda:
            conds = []
            for im in map(cargar, j["imagenes"]):
                cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, im.width / im.height)
                conds.append(p1.image_processor.resize(im, ch, cw))
            emb[j["id"]] = tocpu(p1.encode_prompt(prompt=j["prompt"], image=conds, device="cuda"))
    del p1; gc.collect(); torch.cuda.empty_cache()
    print(f"ronda {n_ronda}: embeddings {time.time()-t1:.1f}s", flush=True)
    # fase 2: transformer directo a GPU (41 GB bf16), sin text encoder
    t2 = time.time()
    tr = QwenImageTransformer2DModel.from_pretrained(REPO, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cuda")
    sched = FlowMatchEulerDiscreteScheduler.from_config({
        "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False, "max_image_seq_len": 8192,
        "max_shift": math.log(3), "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": None,
        "stochastic_sampling": False, "time_shift_type": "exponential", "use_beta_sigmas": False,
        "use_dynamic_shifting": True, "use_exponential_sigmas": False, "use_karras_sigmas": False})
    pipe = QwenImageEditPlusPipeline.from_pretrained(REPO, transformer=tr, text_encoder=None, tokenizer=None, processor=None,
                                                     scheduler=sched, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights("lightx2v/Qwen-Image-Edit-2511-Lightning", weight_name="Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors")
    pipe.fuse_lora(); pipe.unload_lora_weights()
    pipe.vae.to("cuda"); pipe.vae.enable_tiling()
    torch.cuda.synchronize()
    print(f"ronda {n_ronda}: transformer+LoRA+VAE {time.time()-t2:.1f}s", flush=True)
    for j in ronda:
        pe, pm = emb[j["id"]]
        imgs = [cargar(im) for im in j["imagenes"]]
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
        img = pipe(image=imgs, prompt_embeds=pe.cuda(), prompt_embeds_mask=pm.cuda() if pm is not None else None, width=W, height=H,
                   num_inference_steps=STEPS, true_cfg_scale=1.0, generator=torch.Generator("cuda").manual_seed(int(j["seed"]))).images[0]
        torch.cuda.synchronize(); dt = time.time() - t
        img.save(f"{args.out}/{j['id']}.png")
        r = dict(id=j["id"], seed=j["seed"], steps=STEPS, width=img.width, height=img.height, entradas=len(imgs), seconds=round(dt, 2),
                 vram_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        print(json.dumps(r), flush=True); stats.append(r)
        json.dump(dict(model=REPO, steps=STEPS, total_seconds=round(time.time() - t_all, 1), imagenes=stats),
                  open(f"{args.out}/stats.json", "w"), indent=1)
    del pipe, tr, emb; tr = None; gc.collect(); torch.cuda.empty_cache()
