#!/usr/bin/env python
"""Generador unificado Qwen-Image (texto -> imagen). Por defecto Lightning de 8 pasos; --base usa 40 pasos sin LoRA.

Uso:  /root/venv-qwen/bin/python /opt/assetgen/gen_images.py prompts.yaml [--out DIR] [--base] [--only a,b] [--force]
Entrada (JSON o YAML):
  style_suffix: ", single object, full body, centered, plain white background, game asset"   # se anade a cada prompt
  size: 1024            # o "1024x768"; tambien width/height (multiplos de 16)
  variants: 2           # variantes por prompt (se usan las primeras N seeds)
  seeds: [11, 222]      # si faltan seeds para las variantes, se completan con seed+1...
  prompts:              # mapa nombre -> texto, o lista de {name, prompt, variants, seeds, size, style_suffix}
    soldado: "a roman legionary soldier ..."
    escorpion: {prompt: "a giant desert scorpion", variants: 4, size: "1024x1024"}
Salida: DIR/<nombre>_v<i>.png y DIR/stats.json (carga, segundos por imagen, VRAM pico).

Limite de RAM del cgroup (~50 GB): enable_model_cpu_offload (57 GB en RAM) muere por OOM. Carga por etapas:
fase 1 codifica los prompts con el text encoder en GPU y lo libera; fase 2 carga el transformer (41 GB bf16) directo a GPU
(device_map=cuda) sin text encoder. Pico ~41 GB de VRAM (L40S: ~4 s/imagen con Lightning 8 pasos)."""
import argparse, gc, json, math, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("config", help="JSON o YAML con los prompts")
ap.add_argument("--out", default=os.environ.get("OUT", "/root/shared/img"))
ap.add_argument("--base", action="store_true", help="Qwen-Image base: 40 pasos, true_cfg 4.0, sin LoRA (mas lento, mas fiel)")
ap.add_argument("--steps", type=int, help="sobrescribe los pasos (8 Lightning / 40 base)")
ap.add_argument("--only", help="nombres separados por coma")
ap.add_argument("--force", action="store_true", help="regenerar aunque el PNG exista")
args = ap.parse_args()

def load_cfg(path):
    txt = open(path, encoding="utf-8").read()
    if path.lower().endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(txt)
    return json.loads(txt)

def parse_size(v, default=(1024, 1024)):
    if v is None: return default
    if isinstance(v, int): return (v, v)
    if isinstance(v, str) and "x" in v.lower():
        w, h = v.lower().split("x"); return (int(w), int(h))
    return (int(v), int(v))

cfg = load_cfg(args.config)
g_suffix = cfg.get("style_suffix", "")
g_size = parse_size(cfg.get("size")) if "size" in cfg else (int(cfg.get("width", 1024)), int(cfg.get("height", 1024)))
g_var = int(cfg.get("variants", 1))
g_seeds = [int(s) for s in cfg.get("seeds", [11, 222, 3333, 44444])]
raw = cfg["prompts"]
if isinstance(raw, dict):
    raw = [{"name": k, **(v if isinstance(v, dict) else {"prompt": v})} for k, v in raw.items()]
only = set(args.only.split(",")) if args.only else None
jobs = []  # (name, variante, seed, ancho, alto, texto)
for it in raw:
    name = it["name"]
    if only and name not in only: continue
    nvar = int(it.get("variants", g_var))
    seeds = [int(s) for s in it.get("seeds", g_seeds)]
    while len(seeds) < nvar: seeds.append(seeds[-1] + 1)
    if "size" in it: w, h = parse_size(it["size"])
    elif "width" in it or "height" in it: w, h = int(it.get("width", g_size[0])), int(it.get("height", g_size[1]))
    else: w, h = g_size
    assert w % 16 == 0 and h % 16 == 0, f"{name}: la resolucion debe ser multiplo de 16 ({w}x{h})"
    text = it["prompt"] + it.get("style_suffix", g_suffix)
    for i in range(nvar):
        jobs.append((name, i + 1, seeds[i], w, h, text))
os.makedirs(args.out, exist_ok=True)
todo = [j for j in jobs if args.force or not os.path.exists(f"{args.out}/{j[0]}_v{j[1]}.png")]
if not todo:
    print("nada que generar (PNG ya existentes; usa --force)"); sys.exit(0)

import torch
from diffusers import DiffusionPipeline, QwenImagePipeline, QwenImageTransformer2DModel, FlowMatchEulerDiscreteScheduler
MODE = "base" if args.base else "lightning8"
STEPS = args.steps or (40 if args.base else 8)
CFG = 4.0 if args.base else 1.0   # Lightning se destila sin CFG: sin prompt negativo
REPO = "Qwen/Qwen-Image"
t0 = time.time()
# fase 1: embeddings de texto (un text encoder en GPU y se libera)
p1 = DiffusionPipeline.from_pretrained(REPO, transformer=None, vae=None, torch_dtype=torch.bfloat16)
p1.text_encoder.to("cuda")
emb = {}
with torch.no_grad():
    for n, text in {(j[0], j[5]): None for j in todo}:
        emb[(n, text)] = p1.encode_prompt(prompt=text, device="cuda")
    neg = p1.encode_prompt(prompt=" ", device="cuda") if CFG > 1 else None
tocpu = lambda v: tuple(None if x is None else x.cpu() for x in v)
emb = {k: tocpu(v) for k, v in emb.items()}
neg = tocpu(neg) if neg else None
del p1; gc.collect(); torch.cuda.empty_cache()
print(f"fase1 {time.time()-t0:.1f}s", flush=True)
# fase 2: transformer directo a GPU, sin text encoder
t1 = time.time()
tr = QwenImageTransformer2DModel.from_pretrained(REPO, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cuda")
kw = {}
if not args.base:
    kw["scheduler"] = FlowMatchEulerDiscreteScheduler.from_config({
        "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False, "max_image_seq_len": 8192,
        "max_shift": math.log(3), "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": None,
        "stochastic_sampling": False, "time_shift_type": "exponential", "use_beta_sigmas": False,
        "use_dynamic_shifting": True, "use_exponential_sigmas": False, "use_karras_sigmas": False})
pipe = QwenImagePipeline.from_pretrained(REPO, transformer=tr, text_encoder=None, tokenizer=None, torch_dtype=torch.bfloat16, **kw)
if not args.base:
    pipe.load_lora_weights("lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors")
    pipe.fuse_lora(); pipe.unload_lora_weights()
pipe.vae.to("cuda"); pipe.vae.enable_tiling()
torch.cuda.synchronize(); load = time.time() - t0
print(f"fase2 {time.time()-t1:.1f}s (carga total {load:.1f}s) modo={MODE} pasos={STEPS}", flush=True)

stats = []
for name, v, seed, w, h, text in todo:
    pe, pm = emb[(name, text)]
    a = dict(prompt_embeds=pe.cuda(), prompt_embeds_mask=pm.cuda() if pm is not None else None)
    if neg:
        a.update(negative_prompt_embeds=neg[0].cuda(), negative_prompt_embeds_mask=neg[1].cuda() if neg[1] is not None else None)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
    img = pipe(**a, width=w, height=h, num_inference_steps=STEPS, true_cfg_scale=CFG,
               generator=torch.Generator("cuda").manual_seed(seed)).images[0]
    torch.cuda.synchronize(); dt = time.time() - t
    f = f"{args.out}/{name}_v{v}.png"; img.save(f)
    r = dict(name=f"{name}_v{v}", seed=seed, steps=STEPS, width=w, height=h, seconds=round(dt, 2),
             vram_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), file=os.path.basename(f))
    print(json.dumps(r), flush=True); stats.append(r)
    json.dump(dict(mode=MODE, steps=STEPS, true_cfg_scale=CFG, load_seconds=round(load, 1), images=stats),
              open(f"{args.out}/stats.json", "w"), indent=1)
