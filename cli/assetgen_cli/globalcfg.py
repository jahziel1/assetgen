"""Configuracion global del usuario (~/.config/assetgen/config.yaml) y registro de costos de todos los proyectos.

config.yaml (ver ejemplos/config_global.yaml):
    tope_mensual_usd: 20            # opcional: pide confirmacion si el gasto del mes + la estimacion lo supera
    blender: /Applications/Blender 5.3.app/Contents/MacOS/Blender      # opcional (tambien variable BLENDER o 'blender' en el PATH)
    quaternius:
      ual1: /ruta/a/UAL1_Standard.glb      # pack Universal Animation Library (CC0, no se redistribuye)
      animales: /ruta/a/Ultimate Animated Animals   # carpeta del pack (con glTF/Wolf.gltf ...), o un mapa nombre: ruta

La variable ASSETGEN_CONFIG cambia la ruta del archivo; el registro global vive junto a el (costos_global.jsonl).
"""
import datetime, json, os, shutil, sys

from .yamlmini import leer

RUTA = os.environ.get("ASSETGEN_CONFIG") or os.path.expanduser("~/.config/assetgen/config.yaml")
BLENDER_MAC = "/Applications/Blender 5.3.app/Contents/MacOS/Blender"


def ruta_global():
    return os.environ.get("ASSETGEN_CONFIG") or os.path.expanduser("~/.config/assetgen/config.yaml")


def ruta_costos():
    return os.path.join(os.path.dirname(ruta_global()), "costos_global.jsonl")


def cargar():
    r = ruta_global()
    if not os.path.exists(r):
        return {}
    cfg = leer(r) or {}      # si existe pero no se puede leer, falla en voz alta (no seguir sin tope)
    if not isinstance(cfg, dict):
        raise SystemExit(f"No se pudo leer {r}: se esperaba un mapa de claves")
    tope = cfg.get("tope_mensual_usd")
    if tope is not None:
        try:
            float(tope)
        except (TypeError, ValueError):
            raise SystemExit(f"{r}: tope_mensual_usd no es un numero ({tope!r})")
    return cfg


def blender_exe():
    cfg = cargar()
    for c in (os.environ.get("BLENDER"), cfg.get("blender"), shutil.which("blender"), BLENDER_MAC):
        if c and os.path.exists(os.path.expanduser(c)):
            return os.path.expanduser(c)
    raise SystemExit("No encuentro Blender: define 'blender:' en ~/.config/assetgen/config.yaml o la variable BLENDER")


def ual1():
    cfg = cargar()
    p = os.environ.get("ASSETGEN_UAL1") or (cfg.get("quaternius") or {}).get("ual1")
    if not p:
        raise SystemExit(f"Falta la ruta del pack Quaternius UAL1: agrega\n  quaternius:\n    ual1: /ruta/UAL1_Standard.glb\na {ruta_global()} (o variable ASSETGEN_UAL1)")
    p = os.path.expanduser(p)
    if not os.path.exists(p):
        raise SystemExit(f"No existe el archivo de Quaternius UAL1: {p}")
    return p


def fuente_animal(animales, fuente):
    """Ruta al .gltf/.glb de Quaternius `fuente` (p.ej. 'Wolf') del pack `animales`.
    `animales` = ruta a la carpeta del pack, o un nombre que se busca en quaternius.animales del config global (ruta, o mapa nombre: ruta);
    si `animales` es true/vacio se usa quaternius.animales (variable ASSETGEN_ANIMALES). Falla en voz alta si no existe."""
    q = (cargar().get("quaternius") or {}).get("animales")
    cand = []
    if isinstance(animales, str) and animales.strip():
        cand.append(animales)
        if isinstance(q, dict) and animales in q:
            cand.append(q[animales])
        elif isinstance(q, str) and q:
            cand.append(q)
    else:
        cand.append(os.environ.get("ASSETGEN_ANIMALES") or (q if isinstance(q, str) else None) or (next(iter(q.values()), None) if isinstance(q, dict) else None))
    for c in cand:
        if not c:
            continue
        base = os.path.expanduser(str(c))
        if os.path.isfile(base):
            return base
        for sub in ("glTF", "", "GLB", "glb"):
            for ext in (".gltf", ".glb"):
                f = os.path.join(base, sub, f"{fuente}{ext}")
                if os.path.isfile(f):
                    return f
    raise SystemExit(f"No encuentro la fuente '{fuente}' del pack de animales Quaternius ({animales!r}): agrega\n  quaternius:\n    animales: /ruta/al/pack\n"
                     f"a {ruta_global()} (o variable ASSETGEN_ANIMALES), o pon una ruta en 'animales:' del asset")


def registrar(reg):
    """Agrega una linea al registro global de costos (un archivo para todos los proyectos)."""
    r = ruta_costos()
    os.makedirs(os.path.dirname(r), exist_ok=True)
    with open(r, "a") as f:
        f.write(json.dumps(reg, ensure_ascii=False) + "\n")


def gasto_mes(ahora=None):
    mes = (ahora or datetime.datetime.now()).strftime("%Y-%m")
    tot = 0.0
    try:
        with open(ruta_costos()) as f:
            for linea in f:
                try:
                    j = json.loads(linea)
                except ValueError:
                    continue
                if str(j.get("fecha", "")).startswith(mes):
                    tot += float(j.get("costo_usd", 0))
    except OSError:
        pass
    return round(tot, 4)


def confirmar_tope(estimacion, si=False):
    """Si el gasto del mes + la estimacion supera `tope_mensual_usd`, pide confirmacion (o aborta sin terminal)."""
    tope = cargar().get("tope_mensual_usd")
    if tope is None:
        return
    from .sesion import log
    gasto = gasto_mes()
    log(f"gasto del mes ${gasto:.2f} + estimacion ${estimacion:.2f} (tope mensual ${float(tope):.2f})")
    if gasto + estimacion <= float(tope):
        return
    msg = f"Se superaria el tope mensual de ${float(tope):.2f} (gastado ${gasto:.2f} + esta corrida ~${estimacion:.2f})."
    if si:
        log("AVISO: " + msg + " Continuando por --si."); return
    if not sys.stdin.isatty():
        raise SystemExit(msg + " Sin terminal para confirmar: usa --si o sube tope_mensual_usd.")
    if input(msg + " Continuar? [s/N] ").strip().lower() not in ("s", "si", "y", "yes"):
        raise SystemExit("Cancelado por el usuario (tope mensual).")
