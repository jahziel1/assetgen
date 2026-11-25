"""tono_piel.py -- lleva la piel de la textura de color base de un glb hacia uno o varios tonos objetivo (sRGB 0-255).

    blender -b -P tono_piel.py -- --entrada m.glb --salida DIR --nombre n --tono 172,128,100 [--tono r,g,b ...]
        [--zona cabeza | --caja xmin,xmax,zmin,zmax ...] [--vmin 0.33] [--json stats.json]

Escribe DIR/<nombre>_t<i>.glb y <nombre>_t<i>_color.png (uno por tono) y <nombre>_mascara.npy (mascara de piel a 1/4).
Con --json guarda la piel media antes (y del primer tono despues), en sRGB 0-255, y el numero de texeles.

La mascara de piel = zona geometrica (donde mirar) x color de piel (donde hay piel) -> aclara sin tocar cuero, cabello ni dorados:
  * Zona geometrica. `--zona cabeza`: cabeza y cuello centrales de un modelo cuerpo entero normalizado (el 32 % superior;
    rampa suave; sin el chaleco y sin la banda dorada de la frente). `--caja`: cajas en coordenadas normalizadas del modelo
    (xn = |x - cx| / alto, zn = (z - zmin) / alto, con z arriba como en un glTF importado en Blender; se pueden repetir; la
    zona es la union); sirve para manos, cuello, brazos. Sin ninguna de las dos, todo el modelo.
  * Color. Triangulos cuyo color en la textura tiene matiz de piel (hsv), con voto de vecinos; se excluyen los dorados (y 2 anillos
    alrededor) en cualquier posicion. `--vmin` (0-1) exige un brillo minimo: separa el cuero oscuro de la piel clara.
  * Recoloreado: obj = tono + (rgb - media) * c, con c = sqrt(media(tono / media)) (contraste amortiguado: la razon pura
    multiplica el ruido 2-3 veces). Pasa a sRGB tal cual esta en la textura (no lineal), como el resto del pipeline.

Se puede importar (funciones `aclarar_glb`) o llamar por linea de comandos. Las utilidades sin bpy (ΔE, color) estan en
`assetgen_cli/piel_color.py`."""
import json
import os
import sys

import bpy
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "..", "cli"))
from assetgen_cli import piel_color as PC  # noqa: E402


def ss(e0, e1, t):
    return np.clip((t - e0) / (e1 - e0), 0, 1)


def hsv(c):
    mx_ = c.max(-1)
    mn_ = c.min(-1)
    d_ = np.maximum(mx_ - mn_, 1e-6)
    r_, g2, b_ = c[..., 0], c[..., 1], c[..., 2]
    h_ = np.where(mx_ == r_, ((g2 - b_) / d_) % 6, np.where(mx_ == g2, (b_ - r_) / d_ + 2, (r_ - g2) / d_ + 4)) / 6
    return h_, np.where(mx_ > 0, (mx_ - mn_) / np.maximum(mx_, 1e-6), 0), mx_


def _imagen_base(me):
    mat = me.materials[0]
    nodos = [n for n in mat.node_tree.nodes if n.type == "TEX_IMAGE" and any(l.to_socket.name == "Base Color" for l in n.outputs[0].links)]
    return nodos[0]


def _malla(ob):
    me = ob.data
    me.calc_loop_triangles()
    nt = len(me.loop_triangles)
    lt = np.empty(nt * 3, np.int32)
    me.loop_triangles.foreach_get("loops", lt)
    vt = np.empty(nt * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", vt)
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    M = np.array(ob.matrix_world)
    co = co @ M[:3, :3].T + M[:3, 3]
    uv = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers.active.data.foreach_get("uv", uv)
    return co, uv.reshape(-1, 2), lt.reshape(nt, 3), vt.reshape(nt, 3)


def peso_zona(co, vt, zona, cajas):
    """Peso 0-1 por vertice de la zona geometrica; devuelve (wv, zn, xn, zb_banda_dorada_o_None callback)."""
    z = co[:, 2]
    zmin, zmax = z.min(), z.max()
    Hh = zmax - zmin
    zn = (z - zmin) / Hh
    cx = np.median(co[zn > 0.6, 0])
    xn = np.abs(co[:, 0] - cx) / Hh
    if zona == "cabeza":
        wv = np.maximum(ss(0.30, 0.36, zn), ss(0.20, 0.27, zn) * (1 - ss(0.07, 0.11, xn)))   # cabeza + cuello central, rampa suave
    elif cajas:
        wv = np.zeros(len(co), np.float32)
        r = 0.012
        for x0, x1, z0, z1 in cajas:
            w = ss(x0 - r, x0, xn) * (1 - ss(x1, x1 + r, xn)) * ss(z0 - r, z0, zn) * (1 - ss(z1, z1 + r, zn))
            wv = np.maximum(wv, w)
    else:
        wv = np.ones(len(co), np.float32)
    return wv, zn, xn


def mascara_piel(px, W, H, co, uv, lt, vt, zona=None, cajas=(), vmin=0.0):
    """-> (piel (H,W) float 0-1, media sRGB 0-1 de los texeles de piel fuerte, texeles)."""
    wv, zn, xn = peso_zona(co, vt, zona, cajas)
    wt = wv[vt].mean(1)
    sel = wt > 0.02
    uc = uv[lt].mean(1)
    cxi = np.clip((uc[:, 0] * W).astype(int), 0, W - 1)
    cyi = np.clip((uc[:, 1] * H).astype(int), 0, H - 1)
    th, ts, tv = hsv(px[cyi, cxi, :3])
    if zona == "cabeza":
        # banda dorada de la frente: triangulos dorados brillantes al frente y centrados en la mitad alta -> la piel no pasa de su borde inferior
        zt = zn[vt].mean(1)
        xt = xn[vt].mean(1)
        yt = co[vt, 1].mean(1)
        oro = (th > 0.085) & (th < 0.16) & (ts > 0.3) & (tv > 0.45) & (zt > 0.6) & (xt < 0.06) & (yt < np.median(co[:, 1]))
        zb = np.percentile(zt[oro], 3) if oro.sum() > 50 else 2.0
        print("banda: triangulos", int(oro.sum()), "borde inferior zn", round(float(zb), 3))
        wt = wt * (1 - ss(zb - 0.015, zb - 0.003, zt))
    tri_piel = (((th < 0.092) | (th > 0.97)) & (ts > 0.2) & (tv > 0.12) & (tv < 0.9)).astype(np.float32) * wt
    if vmin > 0:
        tri_piel = tri_piel * ss(vmin - 0.04, vmin + 0.04, tv)
    for _ in range(3):        # voto de vecinos por vertices compartidos
        acc = np.zeros(len(co))
        cnt = np.zeros(len(co))
        np.add.at(acc, vt.ravel(), np.repeat(tri_piel, 3))
        np.add.at(cnt, vt.ravel(), 1)
        tri_piel = (acc / np.maximum(cnt, 1))[vt].mean(1)
    tri_piel = np.clip((tri_piel - 0.35) / 0.3, 0, 1)
    # dorado (banda y ribetes) en cualquier posicion: triangulos dorados + 2 anillos de vecinos fuera de la piel
    oro2 = ((th > 0.088) & (th < 0.17) & (ts > 0.28) & (tv > 0.33)).astype(np.float32)
    for _ in range(2):
        acc = np.zeros(len(co))
        np.maximum.at(acc, vt.ravel(), np.repeat(oro2, 3))
        oro2 = acc[vt].max(1)
    tri_piel = tri_piel * (1 - oro2)
    sel = sel & (tri_piel > 0.02)
    u = uv[lt[sel]]
    pts = np.concatenate([u, u.mean(1, keepdims=True), (u + np.roll(u, 1, 1)) / 2], 1).reshape(-1, 2)
    xi = np.clip((pts[:, 0] * W).astype(int), 0, W - 1)
    yi = np.clip((pts[:, 1] * H).astype(int), 0, H - 1)
    wp = np.repeat(tri_piel[sel], 7)
    geo = np.zeros((H, W), np.float32)
    np.maximum.at(geo, (yi, xi), wp)
    for _ in range(2):
        g = geo.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            g = np.maximum(g, np.roll(np.roll(geo, dy, 0), dx, 1))
        geo = g
    rgb = px[..., :3]
    h, s, v = hsv(rgb)
    piel_h = np.clip((0.13 - h) / 0.03, 0, 1) * (h < 0.5) + (h > 0.97)
    piel = geo * piel_h * np.clip((s - 0.15) / 0.1, 0, 1) * np.clip((v - 0.12) / 0.1, 0, 1) * np.clip((0.95 - v) / 0.1, 0, 1)
    if vmin > 0:
        piel = piel * ss(vmin - 0.04, vmin + 0.04, v)
    fuerte = piel > 0.5
    return piel, rgb[fuerte].mean(0), int(fuerte.sum())


def recolorear(px, piel, media, tono):
    """px (H,W,4) 0-1; piel (H,W); media y tono sRGB 0-1 -> copia de px con la piel llevada al tono."""
    c = PC.contraste_amortiguado(tono, media)
    rgb = px[..., :3]
    obj = np.asarray(tono) + (rgb - media) * c
    nuevo = rgb + piel[..., None] * (obj - rgb)
    p2 = px.copy()
    p2[..., :3] = np.clip(nuevo, 0, 1)
    return p2, c


def aclarar_glb(glb, out, nom, tonos, zona=None, cajas=(), vmin=0.0, json_ruta=None, exportar=True):
    """tonos: lista de (r,g,b) 0-255. Escribe los glb/png y devuelve dict(media_antes, media_despues, texeles)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=glb)
    ob = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
    nodo = _imagen_base(ob.data)
    img = nodo.image
    W, H = img.size
    print("imagen", img.name, W, H, img.colorspace_settings.name)
    px = np.empty(W * H * 4, np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(H, W, 4)
    co, uv, lt, vt = _malla(ob)
    piel, media, n = mascara_piel(px, W, H, co, uv, lt, vt, zona, cajas, vmin)
    print("piel media actual (0-255):", np.round(media * 255), "texeles:", n)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, nom + "_mascara.npy"), (piel * 255).astype(np.uint8)[::4, ::4])
    res = {"media_antes": [round(float(x) * 255, 1) for x in media], "texeles": n, "tonos": []}
    for i, t in enumerate(tonos, 1):
        t01 = np.array(t, np.float64) / 255
        p2, c = recolorear(px, piel, media, t01)
        despues = p2[piel > 0.5, :3].mean(0)
        res["tonos"].append({"objetivo": list(t), "media_despues": [round(float(x) * 255, 1) for x in despues], "contraste": round(float(c), 3)})
        if exportar:
            im2 = bpy.data.images.new(f"{nom}_t{i}", W, H, alpha=True)
            im2.pixels.foreach_set(p2.ravel())
            ruta = os.path.join(out, f"{nom}_t{i}_color.png")
            im2.filepath_raw = ruta
            im2.file_format = "PNG"
            im2.save()
            nodo.image = im2
            bpy.ops.export_scene.gltf(filepath=os.path.join(out, f"{nom}_t{i}.glb"), export_format="GLB")
        print("OK tono", i, t, "media despues", res["tonos"][-1]["media_despues"])
    if json_ruta:
        with open(json_ruta, "w") as f:
            json.dump(res, f, indent=1)
    return res


def main():
    a = sys.argv[sys.argv.index("--") + 1:]

    def todos(n):
        return [a[i + 1] for i, x in enumerate(a) if x == n]

    def uno(n, d=None):
        v = todos(n)
        return v[0] if v else d

    if not uno("--entrada") or not uno("--salida") or not todos("--tono"):
        sys.exit("uso: --entrada m.glb --salida DIR [--nombre n] --tono r,g,b [--tono ...] [--zona cabeza | --caja x0,x1,z0,z1 ...] [--vmin v] [--json f]")
    tonos = [tuple(PC.parse_rgb(t)) for t in todos("--tono")]
    cajas = [tuple(float(x) for x in c.split(",")) for c in todos("--caja")]
    aclarar_glb(uno("--entrada"), uno("--salida"), uno("--nombre", "piel"), tonos, uno("--zona"), cajas, float(uno("--vmin", "0")), uno("--json"))


if __name__ == "__main__":
    main()
