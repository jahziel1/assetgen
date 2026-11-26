#!/usr/bin/env python
"""Uso: python run_pixal3d.py [carpeta_in=/root/in] [carpeta_out=/root/out/pixal3d]
Envolver con: flock /root/gpu.lock /root/venv-pixal3d/bin/python /root/run_pixal3d.py
Pixal3D (TencentARC/Pixal3D, imagen -> GLB con PBR) sobre el entorno de TRELLIS.2. Carga el pipeline UNA vez y procesa todas las
imagenes de la carpeta de entrada.
Env opcionales: RESOLUTION (1024|1536; por defecto 1536, o 1024 con LOW_VRAM=1), LOW_VRAM (0|1: modelos en CPU y a GPU por etapa, ~10-12 GB),
TEX_SIZE (4096), DECIM (caras objetivo, 1000000), SEEDS ("42,7": una salida <n>_s<semilla>.glb por semilla; sin SEEDS: <n>.glb con semilla 42),
MAX_GLB_MB (50; si el GLB pesa mas se redecima a DECIM_FALLBACK=200000), STATS_PATH, FOV (radianes; si no, MoGe-2 estima la camara),
WRITE_READY (escribe <salida>.ready por GLB), ORDER ("a,b": orden de procesado), REMBG_MODEL (ZhengPeng7/BiRefNet).
Salidas por imagen <n>: <n>.glb (o <n>_s<semilla>.glb), <n>_entrada.png (imagen ya recortada y sin fondo que vio el modelo),
<n>.camara.json (camara de Pixal3D en el sistema de coordenadas del GLB; ver "convencion" dentro del propio JSON) y stats.json (lista)."""
import os, sys, time, glob, json, math
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
try:
    import flash_attn  # noqa: F401
    os.environ.setdefault("ATTN_BACKEND", "flash_attn")
except Exception:
    os.environ.setdefault("ATTN_BACKEND", "sdpa")

IN = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else '/root/in')
OUT = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else '/root/out/pixal3d')
os.makedirs(OUT, exist_ok=True)
os.chdir('/root/pixal3d'); sys.path.insert(0, '/root/pixal3d')

import numpy as np
import torch, trimesh
from PIL import Image
import PIL.Image as _PILImage

LOW_VRAM = os.environ.get('LOW_VRAM', '') not in ('', '0')
RES = int(os.environ.get('RESOLUTION', 1024 if LOW_VRAM else 1536))
TEX = int(os.environ.get('TEX_SIZE', 4096)); DECIM = int(os.environ.get('DECIM', 1000000))
MAXMB = float(os.environ.get('MAX_GLB_MB', 50)); DECIM_FALLBACK = int(os.environ.get('DECIM_FALLBACK', 200000))
FOV = float(os.environ.get('FOV', 0) or 0)
SEEDS = [int(x) for x in os.environ.get('SEEDS', '').split(',') if x.strip()]
STATS_PATH = os.environ.get('STATS_PATH', os.path.join(OUT, 'stats.json'))
MESH_SCALE = 1.0; EXTEND_PIXEL = 0; IMG_RES = 512   # los mismos valores que inference.py (run_inference)
MODELO = 'pixal3d'

# briaai/RMBG-2.0 (el que pide pipeline.json de TencentARC/Pixal3D) es gated y de uso no comercial: se usa BiRefNet publico (MIT), como run_trellis2.py.
# DINOv3: inference.py ya usa el espejo publico camenduru/dinov3-vitl16-pretrain-lvd1689m (no gated), no hace falta sustituirlo.
REMBG = os.environ.get('REMBG_MODEL', 'ZhengPeng7/BiRefNet')
from pixal3d.pipelines import rembg as _rb
_o_rb = _rb.BiRefNet.__init__
_rb.BiRefNet.__init__ = lambda self, model_name=None, **kw: _o_rb(self, REMBG, **kw)

import importlib.util   # /root/pixal3d/inference.py por ruta (nombre unico): init_pipeline, load_moge_model, get_camera_params_wild_moge, distance_from_fov
_spec = importlib.util.spec_from_file_location('pixal3d_inference', '/root/pixal3d/inference.py')
P3D = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(P3D)
import o_voxel

# ---------------------------------------------------------------------------------------------------------------------
# Camara. Derivada de pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py (ProjGrid, project_points_to_image_batch),
# inference.py (rotacion final antes de exportar) y o_voxel/postprocess.py (to_glb, cambio de ejes al final).
#  - Mundo "Blender" (Z arriba) de la proyeccion: la rejilla/malla v (cubo [-0.5,0.5]^3) se coloca en p_B = R v / mesh_scale con
#    R = [[1,0,0],[0,0,-1],[0,1,0]] (ProjGrid.__init__, "rotation_matrix"; grid_points / mesh_scale / 2 en forward).
#  - Camara en ese mundo: c2w_B = front_view_transform_matrix con [1,3] = -distance:
#    [[1,0,0,0],[0,0,-1,-d],[0,1,0,0],[0,0,0,1]]  -> posicion (0,-d,0), mira a +Y, arriba +Z, derecha +X (OpenGL/Blender: mira a -Z local).
#  - Vertices del GLB: to_glb hace (x,y,z)->(x,z,-y) (ultimas lineas de postprocess.py) e inference.py aplica despues
#    rot=[[-1,0,0],[0,0,-1],[0,-1,0]]. Compuesto: p_G = (-x, y, -z) con v=(x,y,z); en terminos de p_B: p_G = mesh_scale * M p_B,
#    M = [[-1,0,0],[0,0,1],[0,1,0]] (det +1, rotacion propia, sin espejo).
# ---------------------------------------------------------------------------------------------------------------------
R_GRID = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
M_B2G = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float64)
ROT_GLB = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)   # el mismo que inference.py

CONVENCION = (
    "Camara en el sistema del GLB exportado por run_pixal3d.py (el de inference.py de Pixal3D: to_glb + giro final): Y arriba, unidades del GLB "
    "(el objeto cabe en el cubo [-0.5,0.5]^3 antes del escalado posterior del CLI; mesh_scale=1). 'camara_a_mundo' es una matriz 4x4 camara->mundo "
    "en convencion OpenGL: la camara MIRA HACIA -Z LOCAL, +X local = derecha de la imagen, +Y local = arriba de la imagen. Un punto P del GLB se "
    "proyecta asi: Pc = inv(camara_a_mundo) @ [P,1]; u = fx*Pc.x/(-Pc.z) + cx ; v = -fy*Pc.y/(-Pc.z) + cy, con (u,v) en pixeles de "
    "<n>_entrada.png (origen arriba-izquierda, v crece hacia abajo; cx=ancho/2, cy=alto/2 con pixel = cuadro de 1 unidad, es decir centro geometrico de la imagen). "
    "Con esta cuenta el frente del objeto mira a -Z del GLB (la camara esta en -Z mirando hacia +Z). Si al renderizar con esta camara el calco sale "
    "girado 180 grados sobre Y (camara en +Z), usar 'hipotesis_alternativa.camara_a_mundo'. "
    "Nota: el codigo de Pixal3D muestrea los rasgos con (x_pixel+0.5) y align_corners=False, o sea punto principal a medio pixel del centro (irrelevante a 512 px)."
)


def _lista(m):
    return [[float(x) for x in fila] for fila in np.asarray(m, dtype=np.float64)]


def camara_glb(camera_params, ancho, alto):
    ax, d, ms = camera_params['camera_angle_x'], camera_params['distance'], camera_params.get('mesh_scale', 1.0)
    # camara->mundo en el mundo Blender de la proyeccion (ProjGrid.front_view_transform_matrix con [1,3] = -distance)
    c2w_b = np.array([[1, 0, 0, 0], [0, 0, -1, -d], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
    # mundo Blender -> GLB: rotacion M y escala mesh_scale (p_G = mesh_scale * M p_B)
    c2w_g = np.eye(4); c2w_g[:3, :3] = M_B2G @ c2w_b[:3, :3]; c2w_g[:3, 3] = ms * (M_B2G @ c2w_b[:3, 3])
    # hipotesis alternativa: el GLB NO llevaria el giro final de inference.py (equivale a girar 180 grados sobre Y todo el conjunto camara+mundo)
    gy = np.diag([-1.0, 1.0, -1.0, 1.0]); c2w_alt = gy @ c2w_g
    fx = (ancho / 2.0) / math.tan(ax / 2.0)            # = focal_length_pixels de project_points_to_image_batch, escalada de 512 a ancho
    fov_y = 2.0 * math.atan((alto / ancho) * math.tan(ax / 2.0))
    fy = fx
    return {
        'camara_a_mundo': _lista(c2w_g),
        'posicion': [float(x) for x in c2w_g[:3, 3]],
        'objetivo': [0.0, 0.0, 0.0],
        'arriba': [0.0, 1.0, 0.0],
        'fov_horizontal_rad': float(ax), 'fov_vertical_rad': float(fov_y),
        'fov_horizontal_grados': math.degrees(ax), 'fov_vertical_grados': math.degrees(fov_y),
        'intrinsecos_px': {'fx': float(fx), 'fy': float(fy), 'cx': ancho / 2.0, 'cy': alto / 2.0},
        'distancia_al_origen': float(ms * d),
    }, {
        'descripcion': 'Si el calco sale girado 180 grados sobre Y, la camara correcta es esta (camara en +Z mirando a -Z, sin el giro final de inference.py).',
        'camara_a_mundo': _lista(c2w_alt),
        'posicion': [float(x) for x in c2w_alt[:3, 3]],
    }, _lista(c2w_b)


def preparar(pipe, img):
    """preprocess_image de Pixal3D + registro del recorte. Devuelve (entrada, info_recorte)."""
    registro = []
    crop0 = _PILImage.Image.crop

    def crop_reg(self, box=None):
        registro.append((self.size, box))
        return crop0(self, box)
    _PILImage.Image.crop = crop_reg
    try:
        entrada = pipe.preprocess_image(img)
    finally:
        _PILImage.Image.crop = crop0
    ow, oh = img.size
    info = {'original_px': [ow, oh], 'entrada_px': list(entrada.size)}
    if registro:
        (rw, rh), box = registro[-1]   # el unico crop de preprocess_image: sobre la imagen ya reducida a <=1024
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        sx, sy = rw / ow, rh / oh      # preprocess_image reduce (LANCZOS) si max(lado) > 1024; despues NO reescala: solo recorta
        info.update({
            'reducida_px': [rw, rh], 'escala_reduccion': [sx, sy],
            'caja_en_reducida_px': [x0, y0, x1, y1],
            'caja_en_original_px': [x0 / sx, y0 / sy, x1 / sx, y1 / sy],
            'nota': 'entrada(u,v) = reducida(u+x0, v+y0) = original((u+x0)/sx, (v+y0)/sy). La caja puede salirse de la imagen: lo que cae fuera es fondo negro. '
                    'El recorte es cuadrado (lado = max(ancho,alto) del objeto * 1.1) y no se reescala despues.',
        })
        if (x1 - x0, y1 - y0) != tuple(entrada.size):
            info['aviso'] = f'tamano de entrada {entrada.size} distinto de la caja {(x1 - x0, y1 - y0)}'
    else:
        info['aviso'] = 'no se registro recorte'
    return entrada, info


def camara_crudo(entrada_path, moge, name):
    if FOV > 0:
        gp = torch.tensor([-1.0, 0.0, 0.0])
        d = P3D.distance_from_fov(FOV, gp, torch.tensor([0 - EXTEND_PIXEL, IMG_RES - 1 + EXTEND_PIXEL]), MESH_SCALE, IMG_RES)['distance_from_x']
        cp = {'camera_angle_x': float(FOV), 'distance': float(d), 'mesh_scale': MESH_SCALE}
        print(f'{name}: FOV manual {math.degrees(FOV):.2f} grados, distancia {d:.4f}', flush=True)
        return cp, 'manual'
    moge.cuda()
    try:
        cp = P3D.get_camera_params_wild_moge(entrada_path, moge, device='cuda', mesh_scale=MESH_SCALE, extend_pixel=EXTEND_PIXEL, image_resolution=IMG_RES)
    finally:
        moge.cpu(); torch.cuda.empty_cache()
    print(f"{name}: MoGe-2 camera_angle_x={cp['camera_angle_x']:.4f} ({math.degrees(cp['camera_angle_x']):.1f} grados) distancia={cp['distance']:.4f}", flush=True)
    return cp, 'moge-2'


t0 = time.time()
pipe = P3D.init_pipeline(P3D.MODEL_PATH, device='cuda', low_vram=LOW_VRAM)
moge = None
if FOV <= 0:
    moge = P3D.load_moge_model(device='cpu'); torch.cuda.empty_cache()   # en CPU; solo sube a GPU mientras estima la camara
print(f"carga: {time.time()-t0:.1f}s (low_vram={LOW_VRAM}, resolucion {RES})", flush=True)

_stem = lambda f: os.path.splitext(os.path.basename(f))[0]
files = sorted(f for f in glob.glob(os.path.join(IN, '*')) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
_ord = [x for x in os.environ.get('ORDER', '').split(',') if x]
if _ord: files.sort(key=lambda f: _ord.index(_stem(f)) if _stem(f) in _ord else 99)

stats = []; fallos = []
SAMPLERS = dict(
    sparse_structure_sampler_params={'steps': 12, 'guidance_strength': 7.5, 'guidance_rescale': 0.7, 'rescale_t': 5.0},
    shape_slat_sampler_params={'steps': 12, 'guidance_strength': 7.5, 'guidance_rescale': 0.5, 'rescale_t': 3.0},
    tex_slat_sampler_params={'steps': 12, 'guidance_strength': 1.0, 'guidance_rescale': 0.0, 'rescale_t': 3.0},
)   # los mismos overrides que run_inference de inference.py


def guardar_stats():
    json.dump(stats, open(STATS_PATH, 'w'), indent=1)


def generar(name, entrada, cam, seed, prep_s):
    global RES
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
    mesh_list, (_shape, _tex, res) = pipe.run(
        entrada, camera_params=cam, seed=seed, preprocess_image=False, return_latent=True,
        pipeline_type=f'{RES}_cascade', max_num_tokens=49152, **SAMPLERS)
    mesh = mesh_list[0]; del _shape, _tex
    p = os.path.join(OUT, name + '.glb'); dec = DECIM
    for attempt in range(2):
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
            attr_layout=pipe.pbr_attr_layout, grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=dec, texture_size=TEX, remesh=True, remesh_band=1, remesh_project=0, verbose=False)
        glb.apply_transform(ROT_GLB)   # el mismo giro final que inference.py
        glb.export(p, extension_webp=True)
        if os.path.getsize(p) / 2**20 <= MAXMB or attempt == 1: break
        print(f'{name}: glb {os.path.getsize(p)/2**20:.0f} MB > {MAXMB}, redecimando a {DECIM_FALLBACK}', flush=True); dec = DECIM_FALLBACK
    torch.cuda.synchronize(); dt = time.time() - t
    sc = trimesh.load(p, force='scene'); nf = sum(len(g.faces) for g in sc.geometry.values()); nv = sum(len(g.vertices) for g in sc.geometry.values())
    s = dict(name=name, seed=seed, seconds=round(dt, 1), seconds_prep=round(prep_s, 1), vram_peak_gb=round(torch.cuda.max_memory_allocated()/2**30, 2),
             faces=nf, vertices=nv, decimation_target=dec, resolution=int(res), resolution_pedida=RES, texture_size=TEX,
             size_mb=round(os.path.getsize(p)/2**20, 1), modelo=MODELO)
    print(json.dumps(s), flush=True); stats.append(s)
    if os.environ.get('WRITE_READY'): open(os.path.join(OUT, name + '.ready'), 'w').write('ok\n')
    guardar_stats()
    del mesh, mesh_list, glb; torch.cuda.empty_cache()
    return os.path.basename(p)


for f in files:
    base = _stem(f); tp = time.time()
    try:
        img = Image.open(f); img.load()
        entrada, recorte = preparar(pipe, img)
        entrada_path = os.path.join(OUT, base + '_entrada.png'); entrada.save(entrada_path)
        cam, origen_cam = camara_crudo(entrada_path, moge, base)
        torch.cuda.empty_cache()
    except Exception as e:
        import traceback; traceback.print_exc(); torch.cuda.empty_cache()
        print(f'{base}: FALLO en preproceso/camara: {e!r}', flush=True); fallos.append(base); continue
    prep_s = time.time() - tp
    c_glb, c_alt, c2w_b = camara_glb(cam, entrada.size[0], entrada.size[1])
    cam_json = {
        'modelo': MODELO, 'convencion': CONVENCION,
        'imagen_entrada': base + '_entrada.png', 'ancho_px': entrada.size[0], 'alto_px': entrada.size[1],
        'parametros_crudos': {
            'camera_angle_x': cam['camera_angle_x'], 'distance': cam['distance'], 'mesh_scale': cam['mesh_scale'],
            'resolucion_imagen_nocional': IMG_RES, 'extend_pixel': EXTEND_PIXEL, 'origen': origen_cam,
            'mundo_blender_camara_a_mundo': c2w_b,
            'nota': 'mundo Blender (Z arriba) de ProjGrid: camara en (0,-distance,0) mirando a +Y; la malla v se coloca en p_B = R v / mesh_scale.',
        },
        'camara_glb': c_glb,
        'hipotesis_alternativa': c_alt,
        'recorte': recorte,
        'glb': [], 'resolucion_pipeline': RES,
    }
    for seed in (SEEDS or [42]):
        nombre = f'{base}_s{seed}' if SEEDS else base
        try:
            try:
                salida = generar(nombre, entrada, cam, seed, prep_s)
            except torch.cuda.OutOfMemoryError:   # 1536 en GPU de 24 GB: baja a 1024 en vez de perder la corrida
                if RES == 1024: raise
                print(f'{nombre}: OOM con {RES}; reintento con 1024', flush=True)
                torch.cuda.empty_cache(); RES = 1024; cam_json['resolucion_pipeline'] = RES
                salida = generar(nombre, entrada, cam, seed, prep_s)
            cam_json['glb'].append(salida)
        except Exception as e:
            import traceback; traceback.print_exc(); torch.cuda.empty_cache()
            print(f'{nombre}: FALLO: {e!r}', flush=True); fallos.append(nombre)
    if cam_json['glb']:
        json.dump(cam_json, open(os.path.join(OUT, base + '.camara.json'), 'w'), indent=1)

guardar_stats()
if fallos:
    print('FALLARON: ' + ', '.join(fallos), flush=True)
    sys.exit(1)
