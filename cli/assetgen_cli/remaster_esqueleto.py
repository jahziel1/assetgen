"""Esqueleto aportado aparte (`--esqueleto`) para originales cuyo buffer no lo trae (p. ej. 3DMigoto).
Acepta: .glb/.gltf (skin 0), .fbx (armature, via Blender), .json/.yaml con {huesos: [...]} o una lista; cada hueso:
  indice|hueso|i, nombre|name, padre|parent (-1/null = raiz), reposo|cabeza|head [x, y, z] (opcional)."""
import json, os, subprocess, tempfile

from . import globalcfg
from .remaster_comun import leer_datos

RAIZ_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRIPT_BLENDER = os.path.join(RAIZ_REPO, "blender", "remaster", "medir_blender.py")


def medir_con_blender(ruta, timeout=900):
    """Corre el adaptador Blender sobre `ruta` y devuelve el modelo normalizado."""
    d = tempfile.mkdtemp(prefix="assetgen-medir-")
    salida = os.path.join(d, "modelo.json")
    log_path = os.path.join(d, "blender.log")
    with open(log_path, "w") as lf:
        p = subprocess.run([globalcfg.blender_exe(), "-b", "-P", SCRIPT_BLENDER, "--", "--entrada", ruta, "--salida", salida],
                           stdout=lf, stderr=subprocess.STDOUT, timeout=timeout)
    if p.returncode != 0 or not os.path.exists(salida):
        raise SystemExit(f"fallo el adaptador Blender sobre {os.path.basename(ruta)} (rc={p.returncode}):\n" + open(log_path, errors="replace").read()[-800:])
    return json.load(open(salida))


def _pri(d, *claves, defecto=None):
    for k in claves:
        if k in d and d[k] is not None:
            return d[k]
    return defecto


def leer(ruta):
    ext = os.path.splitext(ruta)[1].lower()
    base = os.path.basename(ruta)
    if ext in (".glb", ".gltf"):
        from .remaster_gltf import esqueleto_externo
        esq = esqueleto_externo(ruta)
    elif ext == ".fbx":
        esq = medir_con_blender(ruta).get("esqueleto")
        if not esq:
            raise SystemExit(f"{base}: no tiene armature")
    else:
        d = leer_datos(ruta)
        lista = d.get("huesos") if isinstance(d, dict) else d
        if not isinstance(lista, list) or not lista:
            raise SystemExit(f"{base}: se esperaba una lista de huesos (o {{huesos: [...]}})")
        huesos = []
        for k, h in enumerate(lista):
            i = int(_pri(h, "indice", "hueso", "i", defecto=k))
            p = _pri(h, "padre", "parent", defecto=-1)
            r = _pri(h, "reposo", "cabeza", "head")
            huesos.append({"indice": i, "nombre": str(_pri(h, "nombre", "name", defecto=f"h{i}")), "padre": int(p),
                           "reposo": [round(float(x), 6) for x in r] if r else None})
        esq = {"fuente": "", "huesos": huesos}
    esq["fuente"] = f"externo: {base}"
    return esq
