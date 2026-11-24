"""cara_comun.py -- utilidades compartidas (numpy, sin bpy) de los scripts de cara: lectura de la malla base MakeHuman (hm08)
y de los ojos MakeHuman, que se colocan sobre la base con el mhclo (coordenadas baricentricas + desplazamiento).

Los datos MakeHuman estan en Y arriba, +Z frente, unidades de decimetro; `a_blender` los pasa a metros con Z arriba y el
frente en -Y (como queda un glTF importado en Blender)."""
import numpy as np

# contorno del ovalo facial en MediaPipe (478 puntos), en orden alrededor de la cara
OVALO = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172,
         58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def leer_obj(ruta):
    """-> dict(v=(n,3) float, caras=[[idx..]], grupo=[nombre por cara], vt=(m,2), caras_vt=[[idx..]])."""
    v, vt, caras, cvt, grp = [], [], [], [], []
    g = ""
    with open(ruta) as f:
        for l in f:
            if l.startswith("v "):
                v.append([float(x) for x in l.split()[1:4]])
            elif l.startswith("vt "):
                vt.append([float(x) for x in l.split()[1:3]])
            elif l.startswith("g "):
                g = l.split()[1]
            elif l.startswith("f "):
                p = [t.split("/") for t in l.split()[1:]]
                caras.append([int(t[0]) - 1 for t in p])
                cvt.append([int(t[1]) - 1 if len(t) > 1 and t[1] else -1 for t in p])
                grp.append(g)
    return dict(v=np.array(v, dtype=np.float64), caras=caras, grupo=grp, vt=np.array(vt, dtype=np.float64), caras_vt=cvt)


def a_blender(p, escala=0.1):
    """MakeHuman (x, y, z) dm -> Blender (x, -z, y) m."""
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1) * escala


def leer_mhclo(ruta):
    """-> dict(escala=[(v1, v2, den)]*3 para x,y,z, filas=[(v1,v2,v3,w1,w2,w3,ox,oy,oz)])."""
    esc = {}
    filas = []
    en_verts = False
    with open(ruta) as f:
        for l in f:
            t = l.split()
            if not t or l.startswith("#"):
                continue
            if t[0] in ("x_scale", "y_scale", "z_scale"):
                esc[t[0][0]] = (int(t[1]), int(t[2]), float(t[3]))
            elif t[0] == "verts":
                en_verts = True
            elif en_verts and len(t) == 9:
                filas.append(tuple(int(x) for x in t[:3]) + tuple(float(x) for x in t[3:]))
            elif en_verts and t[0][0].isalpha():
                en_verts = False
    return dict(escala=[esc["x"], esc["y"], esc["z"]], filas=filas)


def ojos_sobre_base(mhclo, base_v):
    """Posiciones (n,3) de los vertices del proxy mhclo sobre la malla base `base_v` (mismas unidades que base_v)."""
    s = np.array([abs(base_v[a][i] - base_v[b][i]) / den for i, (a, b, den) in enumerate(mhclo["escala"])])
    f = mhclo["filas"]
    idx = np.array([r[:3] for r in f])
    w = np.array([r[3:6] for r in f])
    off = np.array([r[6:9] for r in f])
    return (base_v[idx] * w[:, :, None]).sum(axis=1) + off * s
