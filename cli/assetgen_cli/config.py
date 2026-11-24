"""Lectura de assetgen.yaml."""
import os

from .yamlmini import leer

TIPOS = ("humanoide", "mecanico", "criatura")   # tipos con pipeline completo; otros (edificio, objeto...) solo imagen + 3D
PLANTILLAS = ("vehiculo", "catapulta", "rotor")

# sufijo de estilo del proyecto (clave 'estilo') y sufijo segun tipo de asset; 'sufijo_estilo' en el yaml reemplaza al primero
ESTILO_SUFIJO = {
    "realista": ", realistic, detailed textures, single object, plain white background, game asset",
    "estilizado": ", stylized hand-painted look, single object, plain white background, game asset",
    "low-poly": ", low poly style, flat colors, single object, plain white background, game asset",
}
TIPO_SUFIJO = {
    "humanoide": ", standing in A-pose with arms slightly away from the body, empty hands, full body",
    "mecanico": ", full body, centered",
    "criatura": ", full body, centered",
}
TIPO_SUFIJO_DEF = ", full body, centered"


def cargar(ruta):
    cfg = leer(ruta) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"{ruta}: se esperaba un mapa de claves")
    if not isinstance(cfg.get("assets") or [], list) or not all(isinstance(a, dict) for a in cfg.get("assets") or []):
        raise SystemExit(f"{ruta}: 'assets' debe ser una lista de mapas")
    cfg["assets"] = cfg.get("assets") or []
    cfg.setdefault("preset", "medio")
    if cfg["preset"] not in ("boceto", "medio", "alto", "aaa"):
        raise SystemExit(f"{ruta}: preset desconocido '{cfg['preset']}' (boceto|medio|alto|aaa)")
    cfg.setdefault("assets", [])
    base = os.path.dirname(os.path.abspath(ruta))
    cfg["_base"] = base
    vistos = set()
    for a in cfg["assets"]:
        if not a.get("nombre"):
            raise SystemExit("assetgen.yaml: cada asset necesita 'nombre'")
        if a["nombre"] in vistos:
            raise SystemExit(f"assetgen.yaml: asset repetido '{a['nombre']}'")
        vistos.add(a["nombre"])
        if a.get("imagen") and not os.path.isabs(a["imagen"]):
            a["imagen"] = os.path.join(base, a["imagen"])
        if a.get("tipo") == "mecanico" and a.get("plantilla") not in PLANTILLAS:
            raise SystemExit(f"assetgen.yaml: '{a['nombre']}' es mecanico y necesita 'plantilla' ({'|'.join(PLANTILLAS)})")
    return cfg


def prompt_completo(cfg, asset):
    """prompt del asset + sufijo de estilo del proyecto + sufijo segun tipo."""
    estilo = cfg.get("sufijo_estilo") or ESTILO_SUFIJO.get(cfg.get("estilo", "realista"), ESTILO_SUFIJO["realista"])
    tipo = TIPO_SUFIJO.get(asset.get("tipo"), TIPO_SUFIJO_DEF)
    return asset["prompt"].rstrip(" .,") + estilo + tipo
