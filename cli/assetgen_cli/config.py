"""Lectura de assetgen.yaml."""
import os


def cargar(ruta):
    try:
        import yaml
    except ImportError:
        raise SystemExit("Falta PyYAML: pip install pyyaml (o pip install -e cli/)")
    with open(ruta) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("preset", "medio")
    cfg.setdefault("assets", [])
    base = os.path.dirname(os.path.abspath(ruta))
    for a in cfg["assets"]:
        if not a.get("nombre"):
            raise SystemExit("assetgen.yaml: cada asset necesita 'nombre'")
        if a.get("imagen") and not os.path.isabs(a["imagen"]):
            a["imagen"] = os.path.join(base, a["imagen"])
    return cfg
