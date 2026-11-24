import json, glob, os, sys
d = {}
for f in sorted(glob.glob('/tmp/scr/stats/*.json')):
    j = json.load(open(f)); d[j['stage']] = {k: v for k, v in j.items() if k != 'stage'}
out = {
 "pod": "A40 48GB (GPU compartida, vram_base ~0 porque los runs iban bajo flock)",
 "nota_vram": "vram_peak_mib = pico de nvidia-smi memory.used durante la etapa (incluye cache de PyTorch)",
 "etapas": d,
 "oso": {"huesos": 22, "caras": 40000, "vertices": 33125, "textura_conservada": True, "frames_animacion": 30, "fps_optim": 10,
         "iteraciones": 200, "video_wan": "704x704, 97 cuadros@24fps (4 s), recortado a 3 s (72 cuadros) para la optimizacion"},
 "soldado": {"huesos": 52, "caras": 145434, "vertices": 132505, "textura_conservada": True, "frames_animacion": 25, "iteraciones": 100,
             "video_wan": "704x704, 61 cuadros@24fps, 30 pasos"},
 "escorpion": {"huesos": 65, "caras": 142525, "vertices": 102620, "textura_conservada": True, "frames_animacion": 25, "iteraciones": 50,
             "video_wan": "704x704, 61 cuadros@24fps, 30 pasos, vista lateral izquierda"},
 "instalacion": "Puppeteer ~12 min (varias fallas de compilacion), Wan ~5 min + descarga 32 GB; no incluido en etapas",
 "espera_lock_gpu": "~25 min esperando el lock de GPU (agente A generaba imagenes/3D); no incluido en las etapas"
}
json.dump(out, open(sys.argv[1], 'w'), indent=1, ensure_ascii=False)
print(json.dumps(d, indent=0)[:1500])
