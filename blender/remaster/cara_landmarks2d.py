"""cara_landmarks2d.py -- detecta landmarks faciales 2D (MediaPipe Face Landmarker, Apache 2.0) en una imagen.

    python cara_landmarks2d.py --img frente.png --salida lm.json [--iris-rgb]   (necesita un venv con: mediapipe numpy opencv-python-headless)

Escribe {w, h, puntos: [[x_px, y_px, z]...478], iris: {"izq"/"der": {centro, radio_px}}} y, con --iris-rgb, el color medio del iris
(muestreo del disco de la pupila, sin pupila ni brillos: ver cara_geom.color_iris) en `iris_rgb`.
El modelo (~3.7 MB, Apache 2.0) se descarga una vez a ~/.cache/assetgen/mediapipe/. Imagenes: render ortografico de frente de una cabeza
(cara_render.py) o un retrato frontal. Falla con codigo 2 si no detecta cara."""
import argparse
import json
import os
import sys
import urllib.request

MODELO_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def modelo(cache=None):
    cache = cache or os.path.join(os.path.expanduser("~"), ".cache", "assetgen", "mediapipe")
    os.makedirs(cache, exist_ok=True)
    ruta = os.path.join(cache, "face_landmarker.task")
    if not os.path.exists(ruta) or os.path.getsize(ruta) < 1000000:
        with urllib.request.urlopen(MODELO_URL, timeout=120) as r, open(ruta + ".tmp", "wb") as f:
            f.write(r.read())
        os.replace(ruta + ".tmp", ruta)
    return ruta


def detectar(img_path):
    import mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    opt = vision.FaceLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=modelo()), num_faces=1,
                                       min_face_detection_confidence=0.1, min_face_presence_confidence=0.1)
    im = mp.Image.create_from_file(img_path)
    r = vision.FaceLandmarker.create_from_options(opt).detect(im)
    if not r.face_landmarks:
        return None
    w, h = im.width, im.height
    return w, h, [[p.x * w, p.y * h, p.z * w] for p in r.face_landmarks[0]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--iris-rgb", action="store_true")
    a = ap.parse_args()
    res = detectar(a.img)
    if res is None:
        sys.exit(2)
    w, h, pts = res
    out = dict(w=w, h=h, puntos=pts)
    # iris: 468 centro der + 469..472 borde; 473 centro izq + 474..477 (indices MediaPipe, "der/izq" = del sujeto)
    iris = {}
    for nombre, c in (("der", 468), ("izq", 473)):
        cx, cy = pts[c][0], pts[c][1]
        rad = sum(((pts[c + k][0] - cx) ** 2 + (pts[c + k][1] - cy) ** 2) ** 0.5 for k in range(1, 5)) / 4
        iris[nombre] = dict(centro=[cx, cy], radio_px=rad)
    out["iris"] = iris
    if a.iris_rgb:
        import cv2
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cli"))
        from assetgen_cli import cara_geom
        im = cv2.cvtColor(cv2.imread(a.img), cv2.COLOR_BGR2RGB)
        pix = []
        for d in iris.values():
            cx, cy, r = d["centro"][0], d["centro"][1], d["radio_px"] * 0.85
            for y in range(int(cy - r), int(cy + r) + 1):
                for x in range(int(cx - r), int(cx + r) + 1):
                    if 0 <= y < h and 0 <= x < w and (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                        pix.append(tuple(int(c) for c in im[y, x]))
        out["iris_rgb"] = list(cara_geom.color_iris(pix))
    json.dump(out, open(a.salida, "w"))
    print("LANDMARKS", a.salida, "iris_rgb=", out.get("iris_rgb"))
