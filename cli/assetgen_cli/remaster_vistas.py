"""Renders de referencia (frente, lado, espalda) y la hoja `vistas.jpg` (<= 1000 px), todo con Blender (sin Pillow)."""
import json, os, subprocess

from . import globalcfg
from .remaster_comun import a_canonico, vector_eje

RAIZ_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT = os.path.join(RAIZ_REPO, "blender", "remaster", "renderizar.py")
VISTAS = ("frente", "lado", "espalda")


def escribir_geom(geom, ruta, arriba, frente):
    """Lleva la geometria del adaptador 3dmigoto al espacio canonico (Y arriba, +z frente) y la guarda para el script de Blender."""
    u, f = vector_eje(arriba), vector_eje(frente)
    piezas = []
    for p in geom["piezas"]:
        piezas.append({"nombre": p["nombre"], "pos": [a_canonico(v, u, f) for v in p["pos"]],
                       "nrm": [a_canonico(v, u, f) if v else None for v in p["nrm"]], "uv": p["uv"], "tris": p["tris"]})
    with open(ruta, "w") as fh:
        json.dump({"piezas": piezas}, fh)


def renderizar(carpeta, hoja, entrada=None, geom=None, frente_canonico=(0, 0, 1), textura=None, tam=1024, timeout=900):
    """Corre Blender; escribe las 3 imagenes y la hoja `hoja` (JPEG, 3 x 333 px). Lanza SystemExit con la cola del log si falla."""
    os.makedirs(carpeta, exist_ok=True)
    cmd = [globalcfg.blender_exe(), "-b", "-P", SCRIPT, "--", "--salida", carpeta, "--tam", str(tam),
           "--hoja", hoja, "--frente", ",".join(str(float(v)) for v in frente_canonico)]
    cmd += ["--geom", geom] if geom else ["--entrada", entrada]
    if textura:
        cmd += ["--textura", textura]
    log_path = os.path.join(carpeta, "blender.log")
    with open(log_path, "w") as lf:
        p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
    rutas = [os.path.join(carpeta, v + ".png") for v in VISTAS]
    if p.returncode != 0 or not all(os.path.exists(r) for r in rutas + [hoja]):
        cola = open(log_path, errors="replace").read()[-800:]
        raise SystemExit(f"fallo el render en Blender (rc={p.returncode}); log: {log_path}\n{cola}")
    return rutas
