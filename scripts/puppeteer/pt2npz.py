# raw/*.pt de Puppeteer + malla -> anim.npz (para export_glb.py).  Uso: pt2npz.py <raw_dir> <mesh.obj> <out.npz>
import sys, torch, numpy as np
raw, obj, out = sys.argv[1:4]
V = np.array([[float(x) for x in l.split()[1:4]] for l in open(obj) if l.startswith('v ')])
vmin, vmax = V.min(0), V.max(0)
center = (vmax+vmin)/2; scale = (vmax-vmin).max()   # igual que normalize_vertices del cargador de Puppeteer
ld = lambda n: torch.load(f'{raw}/{n}.pt', map_location='cpu').float().numpy()
np.savez(out, local_quats=ld('local_quats'), root_quats=ld('root_quats'), root_pos=ld('root_pos'), center=center, scale=scale)
print({k: v.shape for k, v in np.load(out).items()})
