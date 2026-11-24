#!/usr/bin/env python
"""Generador Z-Image Turbo (texto -> imagen) con ControlNet de pose opcional. Misma interfaz que gen_images.py.

Uso:  /root/venv-zimage/bin/python /opt/assetgen/gen_images_zimage.py prompts.yaml [--out DIR] [--only a,b] [--force]
Entrada (JSON o YAML), igual que gen_images.py (style_suffix, size, variants, seeds, prompts) mas, por prompt:
  pose: a        # genera un esqueleto OpenPose en pose A y lo usa con ControlNet (Fun-Controlnet-Union-2.1, 8 pasos); "libre" = sin ControlNet
  cn_scale: 0.8  # control_context_scale (0.65-1.0)
  arm_angle: 38  # apertura de los brazos de la pose A en grados
Sin `size`: 1536. Salida: DIR/<nombre>_v<i>.png y DIR/stats.json (carga, segundos por imagen, VRAM pico).
Z-Image-Turbo: 9 pasos (8 forwards), guidance 0. Con ControlNet: 8 pasos. En A40: ~18 s a 1536; con ControlNet ~29 s; VRAM 30-39 GB."""
import argparse, gc, json, math, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("config", help="JSON o YAML con los prompts")
ap.add_argument("--out", default=os.environ.get("OUT", "/root/shared/img"))
ap.add_argument("--only", help="nombres separados por coma")
ap.add_argument("--force", action="store_true", help="regenerar aunque el PNG exista")
ap.add_argument("--cn-scale", type=float, default=float(os.environ.get("CN_SCALE", 0.8)))
args = ap.parse_args()

REPO = "Tongyi-MAI/Z-Image-Turbo"
CN_REPO = "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1"
CN_FILE = os.environ.get("CN_FILE", "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors")


def load_cfg(path):
    txt = open(path, encoding="utf-8").read()
    if path.lower().endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(txt)
    return json.loads(txt)


def parse_size(v, default=(1536, 1536)):
    if v is None: return default
    if isinstance(v, int): return (v, v)
    if isinstance(v, str) and "x" in v.lower():
        w, h = v.lower().split("x"); return (int(w), int(h))
    return (int(v), int(v))


def pose_a(S, grados=38.0):
    """Esqueleto OpenPose (18 puntos COCO, colores estandar) en pose A sobre fondo negro, cuerpo completo centrado, SxS.
    0 nariz,1 cuello,2/5 hombros,3/6 codos,4/7 munecas,8/11 caderas,9/12 rodillas,10/13 tobillos,14/15 ojos,16/17 orejas."""
    from PIL import Image, ImageDraw
    ang = math.radians(grados)
    cx = S / 2; top = S * 0.06; H = S * 0.88
    u = H / 7.5
    P = lambda dx, y: (cx + dx * u, top + y * u)
    sh = 0.95; hipx = 0.45; arm1 = 1.55; arm2 = 1.45; leg1 = 1.9; leg2 = 1.9
    K = {}
    K[0] = P(0, 0.55); K[1] = P(0, 1.35)
    K[2] = P(-sh, 1.5); K[5] = P(sh, 1.5)
    for s, sg in ((3, -1), (6, 1)):
        sx = K[2 if sg < 0 else 5]
        K[s] = (sx[0] + sg * arm1 * u * math.sin(ang), sx[1] + arm1 * u * math.cos(ang))
        K[s + 1] = (K[s][0] + sg * arm2 * u * math.sin(ang), K[s][1] + arm2 * u * math.cos(ang))
    K[8] = P(-hipx, 3.6); K[11] = P(hipx, 3.6)
    K[9] = P(-hipx - 0.15, 3.6 + leg1); K[12] = P(hipx + 0.15, 3.6 + leg1)
    K[10] = P(-hipx - 0.25, 3.6 + leg1 + leg2); K[13] = P(hipx + 0.25, 3.6 + leg1 + leg2)
    K[14] = P(-0.18, 0.42); K[15] = P(0.18, 0.42); K[16] = P(-0.38, 0.55); K[17] = P(0.38, 0.55)
    limbs = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
    cols = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170),
            (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]
    A = 4  # supersampling
    img = Image.new("RGB", (S * A, S * A), (0, 0, 0)); d = ImageDraw.Draw(img)
    stick = max(4, S / 180) * A
    for (a, b), c in zip(limbs, cols):
        (x1, y1), (x2, y2) = K[a], K[b]; x1, y1, x2, y2 = [v * A for v in (x1, y1, x2, y2)]
        L = math.hypot(x2 - x1, y2 - y1); th = math.atan2(y2 - y1, x2 - x1); nx, ny = -math.sin(th), math.cos(th)
        poly = []
        for t in (2 * math.pi * k / 39 for k in range(40)):   # elipse OpenPose: eje largo = limb, eje corto = stick
            ex = math.cos(t) * L / 2; ey = math.sin(t) * stick
            poly.append(((x1 + x2) / 2 + ex * math.cos(th) + ey * nx, (y1 + y2) / 2 + ex * math.sin(th) + ey * ny))
        d.polygon(poly, fill=tuple(int(v * 0.6) for v in c))
    for i, (x, y) in K.items():
        r = stick * 0.9; d.ellipse([x * A - r, y * A - r, x * A + r, y * A + r], fill=cols[i])
    return img.resize((S, S), Image.LANCZOS)


cfg = load_cfg(args.config)
g_suffix = cfg.get("style_suffix", "")
g_size = parse_size(cfg.get("size")) if "size" in cfg else (int(cfg.get("width", 1536)), int(cfg.get("height", 1536)))
g_var = int(cfg.get("variants", 1))
g_seeds = [int(s) for s in cfg.get("seeds", [11, 222, 3333, 44444])]
raw = cfg["prompts"]
if isinstance(raw, dict):
    raw = [{"name": k, **(v if isinstance(v, dict) else {"prompt": v})} for k, v in raw.items()]
only = set(args.only.split(",")) if args.only else None
jobs = []  # (name, variante, seed, ancho, alto, texto, pose, cn_scale, arm_angle)
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
    pose = str(it.get("pose") or cfg.get("pose") or "libre").lower()
    assert pose in ("a", "libre"), f"{name}: pose debe ser a|libre"
    text = it["prompt"] + it.get("style_suffix", g_suffix)
    for i in range(nvar):
        jobs.append((name, i + 1, seeds[i], w, h, text, pose, float(it.get("cn_scale", args.cn_scale)), float(it.get("arm_angle", 38))))
os.makedirs(args.out, exist_ok=True)
todo = [j for j in jobs if args.force or not os.path.exists(f"{args.out}/{j[0]}_v{j[1]}.png")]
if not todo:
    print("nada que generar (PNG ya existentes; usa --force)"); sys.exit(0)

import torch
t0 = time.time()
stats = []; load = 0.0


def correr(lista, pipe, steps, cn):
    for name, v, seed, w, h, text, pose, sc, ang in lista:
        kw = dict(prompt=text, height=h, width=w, num_inference_steps=steps, guidance_scale=0.0,
                  generator=torch.Generator("cuda").manual_seed(seed))
        if cn:
            kw.update(control_image=pose_a(max(w, h), ang).resize((w, h)), controlnet_conditioning_scale=sc)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
        img = pipe(**kw).images[0]
        torch.cuda.synchronize(); dt = time.time() - t
        f = f"{args.out}/{name}_v{v}.png"; img.save(f)
        r = dict(name=f"{name}_v{v}", seed=seed, steps=steps, width=w, height=h, pose=pose if cn else "libre", seconds=round(dt, 2),
                 vram_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), file=os.path.basename(f))
        print(json.dumps(r), flush=True); stats.append(r)
        json.dump(dict(mode="zimage-turbo", load_seconds=round(load, 1), images=stats), open(f"{args.out}/stats.json", "w"), indent=1)


libres = [j for j in todo if j[6] == "libre"]
poses = [j for j in todo if j[6] == "a"]
if libres:
    from diffusers import ZImagePipeline
    tl = time.time()
    pipe = ZImagePipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16).to("cuda")
    torch.cuda.synchronize(); load += time.time() - tl
    print(f"carga Z-Image {time.time()-tl:.1f}s", flush=True)
    correr(libres, pipe, 9, False)
    del pipe; gc.collect(); torch.cuda.empty_cache()
if poses:
    from diffusers import ZImageControlNetPipeline, ZImageControlNetModel
    from huggingface_hub import hf_hub_download
    tl = time.time()
    cn = ZImageControlNetModel.from_single_file(hf_hub_download(CN_REPO, CN_FILE), torch_dtype=torch.bfloat16)
    pipe = ZImageControlNetPipeline.from_pretrained(REPO, controlnet=cn, torch_dtype=torch.bfloat16).to("cuda")
    torch.cuda.synchronize(); load += time.time() - tl
    print(f"carga Z-Image + ControlNet {time.time()-tl:.1f}s", flush=True)
    correr(poses, pipe, 8, True)
json.dump(dict(mode="zimage-turbo", load_seconds=round(load, 1), images=stats), open(f"{args.out}/stats.json", "w"), indent=1)
