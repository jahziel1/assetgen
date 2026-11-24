"""cara_pipeline.py -- corre de principio a fin el ajuste de la cabeza base MakeHuman a una cabeza generada (Python 3 estandar).

    python3 cara_pipeline.py --glb cabeza.glb --retrato frente.png --salida DIR --centro-gen x,y,z --escala-gen m
        [--blender /ruta/Blender] [--venv-python /ruta/venv/bin/python] [--cache DIR] [extra de cara_base.py ...]

Pasos: (1) asegura los assets CC0 de MakeHuman en la cache (assets_makehuman.py); (2) render ortografico de frente de la cabeza generada
(centrado en --centro-gen, ancho visible --escala-gen, metros, Z arriba, frente en -Y) y de la base MakeHuman (cara_render.py);
(3) landmarks 2D con MediaPipe en un venv aparte (cara_landmarks2d.py; el retrato da el color del iris); (4) cara_base.py (ajuste,
ojos, glb, blend, landmarks.json, hoja_cara.jpg). Los intermedios quedan en DIR/_trabajo. El venv necesita: mediapipe numpy opencv-python-headless."""
import argparse
import os
import subprocess
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
BLENDER_DEF = "/Applications/Blender.app/Contents/MacOS/Blender"


def correr(cmd):
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)


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
    correr(ajuste + extra)
