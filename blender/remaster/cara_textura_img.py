"""cara_textura_img.py -- etapa 3, lado imagen (Python 3 con numpy, opencv y pillow: el venv de cara_pipeline.py). Hornea el retrato en la UV de la cara.

    python cara_textura_img.py --dir DIR --retrato-color color.png --lm-color lm_color.json --retrato-detalle nitido.png --lm-detalle lm_nitido.json
                               [--salida DIR/cara_tex.png] [--sigma-detalle 1.6] [--ganancia-detalle 1.0]

Lee DIR/_trabajo/{texeles.npz, cabeza_tex.png, puntos3d.json} (los deja `cara_textura.py --fase extraer`) y los landmarks MediaPipe 2D de cada
retrato (cara_landmarks2d.py). Pasos: (1) correspondencias landmark 2D <-> 3D (puntos de la base ajustada + contornos de los parpados);
(2) camara afin por minimos cuadrados y TPS 2D (en el plano de la imagen) para cada retrato; (3) color del retrato original, llevado en Lab
(media y desviacion de la piel) a la piel de la cabeza alrededor del borde de la cara; (4) alta frecuencia del retrato nitido (luminancia menos
su gaussiano) sumada a la luminancia; (5) fundido hacia el color de la textura de la cabeza en el borde (faldon) y en las zonas de perfil;
(6) relleno de los huecos de la isla UV. Escribe cara_tex.png (RGB, sRGB) y DIR/_trabajo/textura_info.json con los errores y la diferencia de color."""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "..", "cli"))
sys.path.insert(0, AQUI)
from assetgen_cli import cara_imagen as I  # noqa: E402
import cara_comun as C  # noqa: E402

# contornos MediaPipe (478 puntos) de cada ojo: el "izq" del pipeline es el de la izquierda de la imagen (indices 33/133)
OJO_MP = {"izq": dict(ext=33, int=133, sup=[246, 161, 160, 159, 158, 157, 173], inf=[7, 163, 144, 145, 153, 154, 155]),
          "dch": dict(ext=263, int=362, sup=[466, 388, 387, 386, 385, 384, 398], inf=[249, 390, 373, 374, 380, 381, 382])}
EXCLUIR_BASE = {33, 133, 263, 362, 468, 473}                   # comisuras (salen del borde de los parpados) e iris (no hay malla ahi)
POLI_OJOS = [[33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
             [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]]
POLI_CEJAS = [[70, 63, 105, 66, 107, 55, 65, 52, 53, 46], [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]]
POLI_LABIOS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]


def leer_rgb(ruta):
    return np.asarray(Image.open(ruta).convert("RGB"), dtype=np.float64) / 255.0


def correspondencias(lm, pj):
    """-> (P3 (n,3), px (n,2), nombres) de un retrato: puntos de la base + contornos de los parpados (misma fraccion entre comisuras)."""
    pts = np.array(lm["puntos"])[:, :2]
    P3, px, nom = [], [], []
    for k, d in pj["puntos"].items():
        if int(k) in EXCLUIR_BASE:
            continue
        P3.append(d["p"])
        px.append(pts[int(k)])
        nom.append("mp%s" % k)
    for ojo, ix in OJO_MP.items():
        sup3, inf3 = np.array(pj["ojos"][ojo]["sup"]), np.array(pj["ojos"][ojo]["inf"])
        x_ext, x_int = (min(sup3[:, 0].min(), inf3[:, 0].min()), max(sup3[:, 0].max(), inf3[:, 0].max()))
        if ojo == "dch":
            x_ext, x_int = x_int, x_ext
        a2, b2 = pts[ix["ext"]], pts[ix["int"]]
        eje = b2 - a2
        for cara, lista, poli3 in (("sup", ix["sup"], sup3), ("inf", ix["inf"], inf3)):
            t3 = (poli3[:, 0] - x_ext) / (x_int - x_ext)
            o = np.argsort(t3)
            for i in ([ix["ext"]] + lista + [ix["int"]] if cara == "sup" else lista):      # las comisuras una sola vez
                t = float(np.dot(pts[i] - a2, eje) / np.dot(eje, eje))
                t = min(max(t, 0.0), 1.0)
                P3.append([np.interp(t, t3[o], poli3[o, c]) for c in range(3)])
                px.append(pts[i])
                nom.append("%s_%s_%d" % (ojo, cara, i))
    return np.array(P3), np.array(px), nom


def alinear(P3, px, lams=(0.0, 30.0, 300.0, 3000.0)):
    """Camara afin + TPS 2D; elige la regularizacion de la TPS por validacion dejando uno fuera. -> dict con los errores en pixeles."""
    A = I.ajustar_camara_afin(P3, px)
    q = I.proyectar_afin(A, P3)
    _, ix = np.unique(np.round(q / 0.5), axis=0, return_index=True)          # puntos que caen en el mismo sitio dejan la TPS singular
    ix = np.sort(ix)
    P3, px = P3[ix], px[ix]
    A = I.ajustar_camara_afin(P3, px)
    q = I.proyectar_afin(A, P3)
    e_cam = np.linalg.norm(q - px, axis=1)
    loo = {lam: I.tps_loo(q, px, lam) for lam in lams}
    mejor = min(lams, key=lambda l: np.sqrt((loo[l] ** 2).mean()))
    m = I.tps_ajustar(q, px, mejor)
    e_tps = np.linalg.norm(I.tps_evaluar(m, q) - px, axis=1)
    return dict(A=A, tps=m, lam=mejor, n=len(P3), e_cam=e_cam, e_tps=e_tps, e_loo=loo[mejor], loo_todos={str(l): float(np.sqrt((v ** 2).mean())) for l, v in loo.items()})


def mapa_a_retrato(al, P, cub, paso=6.0):
    """Para cada texel cubierto, el pixel del retrato (x, y): camara afin y luego TPS evaluada en una rejilla y muestreada bilineal."""
    q = I.proyectar_afin(al["A"], P[cub])
    lo, hi = q.min(axis=0) - 10, q.max(axis=0) + 10
    gx = np.arange(lo[0], hi[0] + paso, paso)
    gy = np.arange(lo[1], hi[1] + paso, paso)
    G = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)
    R = I.tps_evaluar(al["tps"], G).reshape(len(gy), len(gx), 2)
    fx, fy = (q[:, 0] - gx[0]) / paso, (q[:, 1] - gy[0]) / paso
    return I.muestrear_bilineal(R, fx, fy)


def remuestrear(img, xy, cub):
    """Muestrea img (h, w, c) en los pixeles xy (n, 2) de los texeles cubiertos -> (n_tex, n_tex, c) con 0 fuera."""
    n = cub.shape[0]
    mx = np.full((n, n), -1.0, np.float32)
    my = np.full((n, n), -1.0, np.float32)
    mx[cub], my[cub] = xy[:, 0], xy[:, 1]
    out = cv2.remap(img.astype(np.float32), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE).astype(np.float64)
    return out if out.ndim == 3 else out[..., None]


def mascara_piel(lm, forma):
    """Mascara de piel del retrato: ovalo facial sin ojos, cejas ni labios (dilatados), entre la frente y el labio superior (sin barba)."""
    h, w = forma
    pts = np.array(lm["puntos"])[:, :2]
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [pts[C.OVALO].astype(np.int32)], 1)
    ancho = pts[454, 0] - pts[234, 0]
    m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=max(1, int(0.04 * ancho)))
    k = np.ones((3, 3), np.uint8)
    for poli, dil in [(p, 0.03) for p in POLI_OJOS] + [(p, 0.03) for p in POLI_CEJAS] + [(POLI_LABIOS, 0.04)]:
        x = np.zeros((h, w), np.uint8)
        cv2.fillPoly(x, [pts[poli].astype(np.int32)], 1)
        m[cv2.dilate(x, k, iterations=max(1, int(dil * ancho))) > 0] = 0
    m[int(pts[0, 1]):] = 0
    return m > 0


def _convolucion_normalizada(valor, mascara, sigma_px, reduccion=8):
    """Suaviza `valor` (n, n, 3) solo donde `mascara` es cierta y extiende el resultado a todo el plano (gaussiano sobre valor*mascara
    dividido por el gaussiano de la mascara), en una version reducida del plano para que sea rapido."""
    n = valor.shape[0]
    m = n // reduccion
    num = cv2.resize((valor * mascara[..., None]).astype(np.float32), (m, m), interpolation=cv2.INTER_AREA)
    den = cv2.resize(mascara.astype(np.float32), (m, m), interpolation=cv2.INTER_AREA)
    sg = max(1.0, sigma_px / reduccion)
    num, den = cv2.GaussianBlur(num, (0, 0), sg), cv2.GaussianBlur(den, (0, 0), sg)
    out = num / np.maximum(den[..., None], 1e-6)
    return cv2.resize(out, (n, n), interpolation=cv2.INTER_LINEAR).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--retrato-color", required=True)
    ap.add_argument("--lm-color", required=True)
    ap.add_argument("--retrato-detalle")
    ap.add_argument("--lm-detalle")
    ap.add_argument("--salida")
    ap.add_argument("--sigma-detalle", type=float, default=1.6, help="gaussiano (px del retrato nitido a 1024 de ancho) que se resta para quedarse con el detalle")
    ap.add_argument("--ganancia-detalle", type=float, default=1.0)
    ap.add_argument("--franja-ini", type=float, default=0.02, help="fraccion del ancho de la cara: el faldon queda 100%% de la cabeza hasta aqui")
    ap.add_argument("--franja-ancho", type=float, default=0.07, help="fraccion del ancho de la cara que dura el fundido cabeza -> retrato")
    a = ap.parse_args()
    W = os.path.join(a.dir, "_trabajo")
    salida = a.salida or os.path.join(a.dir, "cara_tex.png")
    T = np.load(os.path.join(W, "texeles.npz"))
    cub, P, N, D, uvh, dh = T["cubierto"], T["P"].astype(np.float64), T["N"].astype(np.float64), T["D"].astype(np.float64), T["uvh"], T["dh"]
    pj = json.load(open(os.path.join(W, "puntos3d.json")))
    A_ = pj["ancho"]
    n = cub.shape[0]
    info = dict(tex=n, ancho_cara_m=A_)

    # ---- (1)(2) alineacion --------------------------------------------------------------------------------------------------------
    retratos = {"color": (a.retrato_color, a.lm_color)}
    if a.retrato_detalle:
        retratos["detalle"] = (a.retrato_detalle, a.lm_detalle)
    imgs, mapas = {}, {}
    for k, (ruta, ljson) in retratos.items():
        lm = json.load(open(ljson))
        imgs[k] = leer_rgb(ruta)
        P3, px, nom = correspondencias(lm, pj)
        al = alinear(P3, px)
        esc = 1024.0 / lm["w"]
        info["alineacion_" + k] = dict(
            puntos=al["n"], lambda_tps=al["lam"], retrato_px=lm["w"],
            error_camara_px=dict(media=float(al["e_cam"].mean()), max=float(al["e_cam"].max())),
            error_tps_px=dict(media=float(al["e_tps"].mean()), max=float(al["e_tps"].max())),
            error_loo_tps_px=dict(media=float(al["e_loo"].mean()), max=float(al["e_loo"].max()), todos_lambda_rms=al["loo_todos"]),
            error_camara_px_a1024=float(al["e_cam"].mean() * esc), error_tps_px_a1024=float(al["e_tps"].mean() * esc),
            error_loo_px_a1024=float(al["e_loo"].mean() * esc),
            peores_camara=[(nom[i], round(float(al["e_cam"][i]), 1)) for i in np.argsort(-al["e_cam"])[:5]])
        mapas[k] = mapa_a_retrato(al, P, cub)
        info["alineacion_" + k]["texeles_fuera_del_retrato"] = int(((mapas[k][:, 0] < 0) | (mapas[k][:, 1] < 0) | (mapas[k][:, 0] > lm["w"] - 1) | (mapas[k][:, 1] > lm["h"] - 1)).sum())
        print("alineacion %s: camara %.2f px, TPS %.2f px (lambda %g), LOO %.2f px (a 1024: %.2f / %.2f / %.2f)" % (
            k, al["e_cam"].mean(), al["e_tps"].mean(), al["lam"], al["e_loo"].mean(), al["e_cam"].mean() * esc, al["e_tps"].mean() * esc, al["e_loo"].mean() * esc))
        if k == "color":
            lm_c = lm

    # ---- (3) color: retrato original -> piel de la cabeza -----------------------------------------------------------------------------
    col = remuestrear(imgs["color"], mapas["color"], cub)
    cabeza = leer_rgb(os.path.join(W, "cabeza_tex.png"))
    Hh, Wh = cabeza.shape[:2]
    cu = cub & (D < 0.045) & (dh < 0.02)
    H = np.zeros((n, n, 3))
    H[cu] = I.muestrear_bilineal(cabeza, uvh[cu, 0] * Wh - 0.5, (1 - uvh[cu, 1]) * Hh - 0.5)
    lab_h = I.rgb_a_lab(H)
    anillo = cu & (D > 0.02 * A_) & (D < 0.14 * A_) & I.es_piel_lab(lab_h)           # solo la piel de la cabeza (no pelo, barba, tela ni dorados)
    dst = I.estadistica_lab(lab_h, anillo, tol_de=14)
    lab_p = I.rgb_a_lab(imgs["color"])
    msk = mascara_piel(lm_c, imgs["color"].shape[:2]) & I.es_piel_lab(lab_p)
    src = I.estadistica_lab(lab_p, msk, tol_de=14)
    lab_c0 = I.rgb_a_lab(col)
    lab_t = I.transferir_lab(lab_c0, src, dst, 0.7, 1.2)
    # lo oscuro (barba, pestanas, cejas) conserva su color: la transferencia pesa segun cuanto se parece el pixel a piel
    w_t = I.suavizado((lab_c0[..., 0] - 26.0) / 16.0)[..., None]
    lab_col = lab_c0 + w_t * (lab_t - lab_c0)
    # correccion local de baja frecuencia: la piel de la cabeza no es uniforme (p. ej. la frente es mucho mas clara que las mejillas), asi que
    # la diferencia entre la piel de la cabeza y el retrato transferido, medida en el anillo y extendida por convolucion normalizada, se resta
    # en el borde y se desvanece hacia el interior (en el centro de la cara manda el retrato)
    m_res = anillo & (D < 0.10 * A_)
    base_suave = cv2.GaussianBlur(lab_col, (0, 0), 4.0)
    res = _convolucion_normalizada(lab_h - base_suave, m_res, sigma_px=0.06 * A_ / (A_ / n * 0.94))
    peso_res = (1.0 - I.suavizado((D - 0.04 * A_) / (0.20 * A_)))[..., None] * w_t
    lab_col = lab_col + peso_res * res
    info["correccion_local"] = dict(texeles_referencia=int(m_res.sum()), media_abs_lab=np.abs(res[m_res]).mean(axis=0).tolist())
    info["color"] = dict(piel_retrato_lab=src, piel_cabeza_borde_lab=dst, texeles_anillo=int(anillo.sum()),
                         piel_retrato_rgb=(I.lab_a_rgb(src["media"]) * 255).round().tolist(), piel_cabeza_rgb=(I.lab_a_rgb(dst["media"]) * 255).round().tolist())
    print("piel retrato Lab", np.round(src["media"], 1), "-> piel cabeza Lab", np.round(dst["media"], 1), "(n anillo %d)" % anillo.sum())

    # ---- (4) detalle: alta frecuencia del retrato nitido ------------------------------------------------------------------------------
    L = lab_col[..., 0].copy()
    if "detalle" in imgs:
        im_d = imgs["detalle"]
        sig = a.sigma_detalle * im_d.shape[1] / 1024.0
        Ls = I.rgb_a_lab(im_d)[..., 0]
        hf = I.alta_frecuencia(Ls, sig)
        hf_t = remuestrear(hf[..., None], mapas["detalle"], cub)[..., 0]
        # el detalle no debe cambiar el tono: sin sesgo local, y acotado para no crear halos en los bordes fuertes
        hf_t = np.clip(hf_t * a.ganancia_detalle, -25.0, 25.0)
        info["detalle"] = dict(sigma_px=sig, ganancia=a.ganancia_detalle, rms_hf=float(np.sqrt((hf_t[cub] ** 2).mean())))
        L = L + hf_t
    lab_fin = lab_col.copy()
    lab_fin[..., 0] = np.clip(L, 0, 100)
    c_retrato = I.lab_a_rgb(lab_fin)

    # ---- (5) fundido con la cabeza (borde, faldon, perfil) ----------------------------------------------------------------------------------
    w_borde = I.suavizado((D - a.franja_ini * A_) / (a.franja_ancho * A_))
    cos = np.clip(-N[..., 1], 0, 1)
    w_n = I.suavizado((cos - 0.15) / 0.40)
    interior = I.suavizado((D - 0.10 * A_) / (0.12 * A_))            # lejos del borde no hay perfil: la normal deja de contar
    w = w_borde * (1 - (1 - interior) * (1 - w_n))
    media_rgb = I.lab_a_rgb(dst["media"])
    mezcla_h = I.suavizado((D - 0.10 * A_) / (0.06 * A_))[..., None]
    F = H * (1 - mezcla_h) + media_rgb * mezcla_h
    F[~cu] = media_rgb
    F = np.where((cu | (D > 0.045))[..., None], F, media_rgb)
    final = c_retrato * w[..., None] + F * (1 - w[..., None])
    final[~cub] = 0
    final = I.rellenar_vacios(final, cub, 48)

    # ---- diferencia de color en la costura ----------------------------------------------------------------------------------------------
    banda_vis = cu & (D > 0.02 * A_) & (D < 0.03 * A_)               # justo donde el borde de la cabeza tapa el faldon
    banda_in = cu & (D > 0.10 * A_) & (D < 0.14 * A_)                  # donde el retrato ya manda: cuanto se parece su color al de la cabeza
    de_costura = float(I.delta_e76(I.rgb_a_lab(final)[banda_vis], lab_h[banda_vis]).mean())
    banda_in = banda_in & I.es_piel_lab(lab_h) & I.es_piel_lab(lab_c0)            # solo donde ambos son piel (sin pelo ni barba)
    de_sin = float(I.delta_e76(lab_c0[banda_in], lab_h[banda_in]).mean())
    de_con = float(I.delta_e76(lab_col[banda_in], lab_h[banda_in]).mean())
    info["costura"] = dict(de76_borde_visible_final_vs_cabeza=de_costura, de76_banda_interior_retrato_sin_transferir=de_sin,
                           de76_banda_interior_retrato_transferido=de_con, texeles_borde=int(banda_vis.sum()), texeles_banda_interior=int(banda_in.sum()))
    # tono medio de la cara final frente al de la cabeza (mismo recuadro del anillo)
    info["costura"]["de76_media_cara_final_vs_piel_cabeza"] = float(I.delta_e76(I.rgb_a_lab(final)[cub & (w > 0.95)].mean(axis=0), dst["media"]))
    print("costura: dE76 borde visible %.2f; banda interior sin transferir %.2f -> transferido %.2f" % (de_costura, de_sin, de_con))
    img8 = (np.clip(final, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(np.flipud(img8), "RGB").save(salida)
    # pesos para depurar (fila 0 arriba)
    dbg = np.stack([w, cub.astype(float), np.clip(D / (0.2 * A_), 0, 1)], axis=-1)
    Image.fromarray(np.flipud((dbg * 255).astype(np.uint8)), "RGB").resize((512, 512)).save(os.path.join(W, "pesos_debug.png"))
    json.dump(info, open(os.path.join(W, "textura_info.json"), "w"), indent=1, default=float)
    print("TEXTURA", salida)


if __name__ == "__main__":
    main()
