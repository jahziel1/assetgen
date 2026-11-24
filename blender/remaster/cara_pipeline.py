"""cara_pipeline.py -- corre de principio a fin el ajuste de la cabeza base MakeHuman a una cabeza generada (Python 3 estandar).

    python3 cara_pipeline.py --glb cabeza.glb --retrato frente.png --salida DIR --centro-gen x,y,z --escala-gen m
        [--blender /ruta/Blender] [--venv-python /ruta/venv/bin/python] [--cache DIR] [extra de cara_base.py ...]
        [--unir [--retrato-ap frente_nitido.png] [--margen-borde 0.04]]

Pasos: (1) asegura los assets CC0 de MakeHuman en la cache (assets_makehuman.py); (2) render ortografico de frente de la cabeza generada
(centrado en --centro-gen, ancho visible --escala-gen, metros, Z arriba, frente en -Y) y de la base MakeHuman (cara_render.py);
(3) landmarks 2D con MediaPipe en un venv aparte (cara_landmarks2d.py; el retrato da el color del iris); (4) cara_base.py (ajuste,
ojos, glb, blend, landmarks.json, hoja_cara.jpg). Los intermedios quedan en DIR/_trabajo. El venv necesita: mediapipe numpy opencv-python-headless.

Etapa 2 opcional (`--unir`, cara_unir.py): (5) limpia la cara, calibra la apertura de los parpados (mide con MediaPipe la apertura de cada ojo en
el retrato --retrato-ap, o en --retrato si falta, y en un render de la cara limpia; ajusta el factor de cada ojo por secante hasta +-3 %, max.
4 pasadas) y (6) une la cara a la cabeza (UV propias, cara_tex, cabeza_unida.blend/.glb, hoja_unida.jpg)."""
import argparse
import os
import subprocess
import json
import math
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "..", "cli"))
from assetgen_cli import cara_geom as G  # noqa: E402
BLENDER_DEF = "/Applications/Blender.app/Contents/MacOS/Blender"
OJOS_MP = {"izq": (33, 133, 159, 145), "dch": (263, 362, 386, 374)}      # (ext, int, sup, inf) de cada ojo en MediaPipe


def correr(cmd):
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


def aperturas_mp(ruta_json):
    """Apertura relativa (alto entre parpados / ancho entre comisuras) de cada ojo a partir de un json de cara_landmarks2d.py."""
    P = json.load(open(ruta_json))["puntos"]
    return {k: G.apertura_relativa(P[s][:2], P[i][:2], P[e][:2], P[n][:2]) for k, (e, n, s, i) in OJOS_MP.items()}


def calibrar_apertura(a, w, det, render, objetivo, extra_cara, max_iter=4, tol=0.03):
    """Corre la fase `cara` de cara_unir.py y mide la apertura (MediaPipe sobre un render frontal de la cara limpia) hasta igualar `objetivo`
    por ojo (+-tol). -> (factores por ojo, aperturas medidas, historial)."""
    hist = {"izq": [], "dch": []}
    f = {"izq": 1.0, "dch": 1.0}
    med = {}
    for it in range(max_iter):
        correr([a.blender, "-b", "-P", os.path.join(AQUI, "cara_unir.py"), "--", "--fase", "cara", "--dir", a.salida,
                "--ap-factor-izq", "%.4f" % f["izq"], "--ap-factor-dch", "%.4f" % f["dch"]] + extra_cara)
        png = os.path.join(w, "limpia_frente.png")
        correr(render + ["--blend", os.path.join(a.salida, "cara_limpia.blend"), "--salida", png, "--centro", a.centro_gen, "--escala", a.escala_gen])
        correr(det + ["--img", png, "--salida", os.path.join(w, "lm_limpia.json")])
        med = aperturas_mp(os.path.join(w, "lm_limpia.json"))
        for k in f:
            hist[k].append((f[k], med[k]))
        print("apertura (iteracion %d): medida %s objetivo %s factores %s" % (it + 1, {k: round(v, 3) for k, v in med.items()},
              {k: round(v, 3) for k, v in objetivo.items()}, {k: round(v, 3) for k, v in f.items()}), file=sys.stderr)
        if all(abs(med[k] - objetivo[k]) <= tol * objetivo[k] for k in f):
            break
        if it < max_iter - 1:
            f = {k: G.siguiente_factor(hist[k], objetivo[k]) for k in f}
    return f, med, hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", required=True)
    ap.add_argument("--retrato", help="retrato frontal: color del iris")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--centro-gen", required=True, help="x,y,z (m) del centro de la cara generada para el render")
    ap.add_argument("--escala-gen", required=True, help="ancho visible (m) del render de la cabeza generada")
    ap.add_argument("--centro-base", default="0,0,0.73")
    ap.add_argument("--escala-base", default="0.3")
    ap.add_argument("--blender", default=os.environ.get("BLENDER", BLENDER_DEF))
    ap.add_argument("--venv-python", required=True)
    ap.add_argument("--cache")
    ap.add_argument("--unir", action="store_true", help="etapa 2: limpiar, calibrar parpados, unir a la cabeza (cara_unir.py)")
    ap.add_argument("--retrato-ap", help="retrato nitido de frente para medir la apertura de los ojos (por defecto, --retrato)")
    ap.add_argument("--margen-borde", default="0.04", help="el contorno de la cara queda esta fraccion del ancho dentro del ovalo MediaPipe")
    ap.add_argument("--sin-hoja", action="store_true")
    a, extra = ap.parse_known_args()
    w = os.path.join(a.salida, "_trabajo")
    os.makedirs(w, exist_ok=True)
    cache = ["--cache", a.cache] if a.cache else []
    correr([sys.executable, os.path.join(AQUI, "assets_makehuman.py")] + cache)
    render = [a.blender, "-b", "-P", os.path.join(AQUI, "cara_render.py"), "--"]
    correr(render + ["--glb", a.glb, "--salida", os.path.join(w, "gen_frente.png"), "--centro", a.centro_gen, "--escala", a.escala_gen])
    correr(render + ["--base", "--salida", os.path.join(w, "base_frente.png"), "--centro", a.centro_base, "--escala", a.escala_base] + cache)
    det = [a.venv_python, os.path.join(AQUI, "cara_landmarks2d.py")]
    correr(det + ["--img", os.path.join(w, "gen_frente.png"), "--salida", os.path.join(w, "lm_gen.json")])
    correr(det + ["--img", os.path.join(w, "base_frente.png"), "--salida", os.path.join(w, "lm_base.json")])
    ajuste = [a.blender, "-b", "-P", os.path.join(AQUI, "cara_base.py"), "--", "--glb", a.glb, "--salida", a.salida,
              "--lm-gen", os.path.join(w, "lm_gen.json"), "--render-gen", os.path.join(w, "gen_frente.json"),
              "--lm-base", os.path.join(w, "lm_base.json"), "--render-base", os.path.join(w, "base_frente.json")] + cache
    if a.retrato:
        correr(det + ["--img", a.retrato, "--salida", os.path.join(w, "lm_retrato.json"), "--iris-rgb"])
        ajuste += ["--retrato-lm", os.path.join(w, "lm_retrato.json")]
    correr(ajuste + extra + (["--sin-hoja"] if a.sin_hoja else []))
    if a.unir:
        ret = a.retrato_ap or a.retrato
        if not ret:
            sys.exit("--unir necesita --retrato (o --retrato-ap) para medir la apertura de los ojos")
        correr(det + ["--img", ret, "--salida", os.path.join(w, "lm_retrato_ap.json")])
        objetivo = aperturas_mp(os.path.join(w, "lm_retrato_ap.json"))
        f, med, hist = calibrar_apertura(a, w, det, render, objetivo, ["--margen-borde", a.margen_borde])
        correr([a.blender, "-b", "-P", os.path.join(AQUI, "cara_unir.py"), "--", "--fase", "unir", "--dir", a.salida, "--glb", a.glb] +
               (["--sin-hoja"] if a.sin_hoja else []))
        json.dump(dict(objetivo=objetivo, medida_cara_limpia=med, factores=f, historial=hist), open(os.path.join(a.salida, "apertura_ojos.json"), "w"), indent=1)
