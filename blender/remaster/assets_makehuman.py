"""assets_makehuman.py -- descarga (si faltan) los assets CC0 de MakeHuman usados como base de cara/cuerpo.

    python3 assets_makehuman.py [--cache DIR] [--forzar]

Cache por defecto: ~/.cache/assetgen/makehuman/ (fuera del repo; NO subir estos archivos al repositorio).
Fuente: https://github.com/makehumancommunity/makehuman (carpeta makehuman/data). Los assets (malla base hm08,
ojos, texturas) estan declarados CC0 1.0 en LICENSE.md y en la cabecera de cada archivo ("This asset was explicitly
released as CC0 in september 2020"). Tambien se guardan LICENSE.md y LICENSE.ASSETS.md (texto CC0).
Solo biblioteca estandar. Importable desde Blender o Python normal: `from assets_makehuman import asegurar`."""
import argparse
import os
import sys
import urllib.request

RAIZ = "https://raw.githubusercontent.com/makehumancommunity/makehuman/master/"
ARCHIVOS = {
    "base.obj": "makehuman/data/3dobjs/base.obj",
    "base.mhclo": "makehuman/data/3dobjs/base.mhclo",
    "eyes_high-poly.obj": "makehuman/data/eyes/high-poly/high-poly.obj",
    "eyes_high-poly.mhclo": "makehuman/data/eyes/high-poly/high-poly.mhclo",
    "eyes_brown.mhmat": "makehuman/data/eyes/materials/brown.mhmat",
    "eyes_brown_eye.png": "makehuman/data/eyes/materials/brown_eye.png",
    "LICENSE.md": "LICENSE.md",
    "LICENSE.ASSETS.md": "LICENSE.ASSETS.md",
}


def cache_por_defecto():
    return os.path.join(os.path.expanduser("~"), ".cache", "assetgen", "makehuman")


def asegurar(cache=None, forzar=False):
    """Devuelve {nombre: ruta_local}; descarga los que falten (o todos con forzar=True)."""
    cache = cache or cache_por_defecto()
    os.makedirs(cache, exist_ok=True)
    rutas = {}
    for nombre, rel in ARCHIVOS.items():
        dst = os.path.join(cache, nombre)
        if forzar or not os.path.exists(dst) or os.path.getsize(dst) == 0:
            req = urllib.request.Request(RAIZ + rel, headers={"User-Agent": "assetgen"})
            with urllib.request.urlopen(req, timeout=60) as r:
                datos = r.read()
            if not datos:
                raise RuntimeError("descarga vacia: " + rel)
            with open(dst + ".tmp", "wb") as f:
                f.write(datos)
            os.replace(dst + ".tmp", dst)
            print("descargado", nombre, len(datos), "bytes", file=sys.stderr)
        rutas[nombre] = dst
    return rutas


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache")
    ap.add_argument("--forzar", action="store_true")
    a = ap.parse_args()
    for k, v in asegurar(a.cache, a.forzar).items():
        print(k, v)
