"""Piezas comunes de `assetgen remaster`: formato del modelo normalizado que producen los adaptadores, lectura de datos
auxiliares (json/yaml), tamano de imagenes sin dependencias y ejes.

Modelo normalizado (dict, lo que devuelve cada adaptador):
  formato, origen (nombre del archivo, nunca una ruta), unidades (None si no se sabe), avisos [str],
  piezas: [{nombre, vertices, triangulos, tipo_indice ('uint16'|'uint32'|None), max_pesos, suma_pesos [min,max]|None,
            huesos_usados [indices del esqueleto o de la paleta global], uvs (canales), material, bbox [[min],[max]], con_pesos}]
  esqueleto: None | {fuente, huesos: [{indice, nombre, padre (-1 raiz), reposo [x,y,z]}]}
  materiales: [{nombre, texturas: [{rol, archivo, ancho, alto, ...}]}]
  bbox: [[min xyz], [max xyz]] en el espacio del archivo; muestra_puntos: ~20000 puntos para inferir ejes (no se escribe al yaml)
  ejes_declarados: None | {arriba: '+y', frente: '+z'}   (los da el layout o la linea de comandos)"""
import json, os, re, struct


def modelo_vacio(formato, origen):
    return {"formato": formato, "origen": origen, "unidades": None, "avisos": [], "piezas": [], "esqueleto": None,
            "materiales": [], "bbox": None, "muestra_puntos": [], "ejes_declarados": None}


def muestrear(puntos, maximo=20000):
    """Submuestreo uniforme (determinista) a lo mas `maximo` puntos."""
    n = len(puntos)
    if n <= maximo:
        return [list(p) for p in puntos]
    paso = n / float(maximo)
    return [list(puntos[int(i * paso)]) for i in range(maximo)]


def leer_datos(ruta, clave=None):
    """JSON o YAML segun la extension; `clave` = ruta con puntos dentro del documento ('poses.A')."""
    try:
        with open(ruta, encoding="utf-8") as f:
            if ruta.lower().endswith(".json"):
                d = json.load(f)
            else:
                from . import yamlmini
                d = yamlmini.yaml().safe_load(f)
    except Exception as e:
        raise SystemExit(f"No se pudo leer {os.path.basename(ruta)}: {e}")
    if clave:
        for k in clave.split("."):
            try:
                d = d[k] if isinstance(d, dict) else d[int(k)]
            except (KeyError, IndexError, ValueError, TypeError):
                raise SystemExit(f"{os.path.basename(ruta)}: no existe la clave '{clave}' (falla en '{k}')")
    return d


def tam_imagen(datos):
    """(ancho, alto) de PNG/JPEG/WEBP(VP8X, VP8, VP8L) leyendo solo la cabecera; (None, None) si no se reconoce."""
    try:
        if datos[:8] == b"\x89PNG\r\n\x1a\n":
            return struct.unpack(">II", datos[16:24])
        if datos[:2] == b"\xff\xd8":
            i = 2
            while i + 9 < len(datos):
                if datos[i] != 0xFF:
                    i += 1
                    continue
                marca = datos[i + 1]
                if marca in (0xD8, 0x01) or 0xD0 <= marca <= 0xD7:
                    i += 2
                    continue
                n = struct.unpack(">H", datos[i + 2:i + 4])[0]
                if 0xC0 <= marca <= 0xCF and marca not in (0xC4, 0xC8, 0xCC):
                    alto, ancho = struct.unpack(">HH", datos[i + 5:i + 9])
                    return ancho, alto
                i += 2 + n
        if datos[:4] == b"RIFF" and datos[8:12] == b"WEBP":
            if datos[12:16] == b"VP8X":
                return 1 + int.from_bytes(datos[24:27], "little"), 1 + int.from_bytes(datos[27:30], "little")
            if datos[12:16] == b"VP8 ":
                a, b = struct.unpack("<HH", datos[26:30])
                return a & 0x3FFF, b & 0x3FFF
            if datos[12:16] == b"VP8L":
                v = int.from_bytes(datos[21:25], "little")
                return (v & 0x3FFF) + 1, ((v >> 14) & 0x3FFF) + 1
    except Exception:
        pass
    return None, None


EJES = {"x": 0, "y": 1, "z": 2}


def parse_eje(s):
    """'+y' / '-z' / 'y' -> (indice, signo)."""
    m = re.fullmatch(r"\s*([+-]?)\s*([xyzXYZ])\s*", str(s))
    if not m:
        raise SystemExit(f"eje invalido '{s}' (usa +x, -x, +y, -y, +z o -z)")
    return EJES[m.group(2).lower()], (-1 if m.group(1) == "-" else 1)


def fmt_eje(i, s):
    return ("-" if s < 0 else "+") + "xyz"[i]


def vector_eje(s):
    i, sg = parse_eje(s)
    v = [0.0, 0.0, 0.0]
    v[i] = float(sg)
    return v


def a_canonico(p, arriba, frente):
    """Punto del espacio del archivo -> espacio canonico (x derecha-de-escena, +y arriba, +z frente, mano derecha).
    `arriba` y `frente` son vectores unitarios sobre ejes (listas de 3)."""
    u, f = arriba, frente
    lat = (u[1] * f[2] - u[2] * f[1], u[2] * f[0] - u[0] * f[2], u[0] * f[1] - u[1] * f[0])      # u x f
    return (sum(p[k] * lat[k] for k in range(3)), sum(p[k] * u[k] for k in range(3)), sum(p[k] * f[k] for k in range(3)))
