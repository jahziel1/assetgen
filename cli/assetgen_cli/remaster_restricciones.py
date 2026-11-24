"""Del modelo normalizado (ver remaster_comun) a `restricciones.yaml`: infiere ejes/escala, arma el diccionario con el origen de
cada valor (medido | inferido | manual) y lo escribe como YAML con comentarios (seccion `motor:` para rellenar a mano).

Convencion de salida: un valor suelto es `{valor: X, origen: Y}`; un bloque con varios valores lleva `origen:` propio."""
import json, math, re

from .remaster_comun import parse_eje, fmt_eje, vector_eje

LIMITE_INDICE = {"uint8": 255, "uint16": 65535, "uint32": 4294967295}
UMBRAL_PIES = 0.08


def V(valor, origen, nota=None):
    d = {"valor": valor, "origen": origen}
    if nota:
        d["nota"] = nota
    return d


# ---------------------------------------------------------------- ejes
def inferir_ejes(mod, arriba=None, frente=None):
    """-> dict(arriba=(i,s), frente=(i,s), origen_arriba, origen_frente, notas). `arriba`/`frente`: '+y'... de la linea de comandos
    (manual); si no, los declarados por el layout (manual); si no, se infieren."""
    notas = []
    dec = mod.get("ejes_declarados") or {}
    if arriba:
        u, ou = parse_eje(arriba), "manual"
    elif dec.get("arriba"):
        u, ou = parse_eje(dec["arriba"]), "manual"
    elif mod["formato"] in ("gltf", "fbx"):
        u, ou = (1, 1), "inferido"
        notas.append("arriba +y por la convencion glTF (los fbx se normalizan a Y arriba al importar en Blender)")
    else:
        bb = mod["bbox"]
        ext = [bb[1][k] - bb[0][k] for k in range(3)]
        u, ou = (max(range(3), key=lambda k: ext[k]), 1), "inferido"
        notas.append("arriba = eje mas largo de la caja; con piezas sin ensamblar puede ser falso: usa --arriba o 'ejes' en el layout")
    if frente:
        f, of = parse_eje(frente), "manual"
    elif dec.get("frente"):
        f, of = parse_eje(dec["frente"]), "manual"
    else:
        f, of, nota = _inferir_frente(mod["muestra_puntos"], u)
        notas.append(nota)
    if f[0] == u[0]:
        raise SystemExit("--arriba y --frente no pueden estar en el mismo eje")
    return {"arriba": u, "frente": f, "origen_arriba": ou, "origen_frente": of, "notas": notas}


def _inferir_frente(puntos, u):
    """Los pies sobresalen hacia delante: en el 8 % inferior, el lado donde la nube se extiende mas respecto a su media."""
    i, s = u
    if not puntos:
        return (2, 1), "inferido", "sin puntos para inferir el frente: se asume +z"
    alt = [s * p[i] for p in puntos]
    lo, hi = min(alt), max(alt)
    pies = [p for p, h in zip(puntos, alt) if h <= lo + UMBRAL_PIES * (hi - lo)]
    asim = {}
    for a in range(3):
        if a != i and len(pies) >= 3:
            vals = [p[a] for p in pies]
            m = sum(vals) / len(vals)
            asim[a] = (max(vals) - m) - (m - min(vals))
    if not asim:
        return (2, 1), "inferido", "sin puntos suficientes para inferir el frente: se asume +z; pasa --frente"
    orden = sorted(asim, key=lambda a: -abs(asim[a]))
    a = orden[0]
    if abs(asim[a]) < 1e-4 * max(hi - lo, 1e-9):
        return (2, 1), "inferido", "no se pudo inferir el frente (pies simetricos): se asume +z; pasa --frente"
    otro = abs(asim[orden[1]]) if len(orden) > 1 else 0.0
    conf = "alta" if abs(asim[a]) > 2 * otro else "baja"
    return (a, 1 if asim[a] > 0 else -1), "inferido", \
        f"frente por la punta de los pies (asimetria en el 8 % inferior; confianza {conf}); revisa vistas.jpg y corrige con --frente si esta mal"


# ---------------------------------------------------------------- construccion
def _limite_indice(tipos):
    tipos = [t for t in tipos if t]
    if not tipos:
        return None, None
    if len(set(tipos)) == 1:
        return tipos[0], LIMITE_INDICE.get(tipos[0])
    peor = min(tipos, key=lambda t: LIMITE_INDICE.get(t, 0))
    return "mixto", LIMITE_INDICE.get(peor)


def construir(mod, ejes):
    avisos = list(mod["avisos"])
    bb = mod["bbox"]
    ext = [bb[1][k] - bb[0][k] for k in range(3)]
    iu, su = ejes["arriba"]
    ifr, sf = ejes["frente"]
    ilat = 3 - iu - ifr
    piezas = mod["piezas"]
    d = {"version": 1, "# 1": "Generado por `assetgen remaster analizar`. Cada valor lleva su origen: medido (leido del archivo), inferido (deducido)",
         "# 2": "o manual (lo puso una persona o la linea de comandos). Corrige a mano lo que este mal; los pasos siguientes leen este archivo."}
    d["original"] = {"archivo": V(mod["origen"], "medido"), "formato": V(mod["formato"], "medido"),
                     "unidades": V(mod["unidades"], "inferido" if mod["formato"] == "gltf" else "manual",
                                   "glTF define metros" if mod["formato"] == "gltf" else None) if mod["unidades"] else
                     V(None, "inferido", "desconocidas; la altura se da en las unidades del archivo")}
    # --- esqueleto
    esq = mod.get("esqueleto")
    if esq and esq["huesos"]:
        origen = "manual" if esq["fuente"].startswith("externo") else "medido"
        d["esqueleto"] = {"origen": origen, "fuente": esq["fuente"], "numero_huesos": len(esq["huesos"]),
                          "raices": [h["indice"] for h in esq["huesos"] if h["padre"] < 0],
                          "# huesos": "indice, nombre, padre (-1 = raiz) y posicion de reposo [x, y, z] en el espacio del archivo",
                          "huesos": [{"i": h["indice"], "nombre": h["nombre"], "padre": h["padre"], "reposo": h["reposo"]} for h in esq["huesos"]]}
    else:
        d["esqueleto"] = {"origen": "medido", "fuente": None, "numero_huesos": 0,
                          "# nota": "el original no trae esqueleto (ni skin ni jerarquia); aportalo con --esqueleto o rellena huesos a mano"}
    # --- pesos
    maxp = max([p["max_pesos"] for p in piezas] + [0])
    con_pesos = any(p["con_pesos"] for p in piezas)
    sm = [p["suma_pesos"] for p in piezas if p["suma_pesos"]]
    if con_pesos:
        d["pesos"] = {"max_por_vertice": V(maxp, "medido", "maximo observado en todos los vertices"),
                      "suma_por_vertice": {"origen": "medido", "minima": min(s[0] for s in sm), "maxima": max(s[1] for s in sm)}}
    else:
        d["pesos"] = {"max_por_vertice": V(None, "medido", "el original no trae pesos de piel")}
    # --- piezas
    tipo, lim = _limite_indice([p["tipo_indice"] for p in piezas])
    lista = []
    for p in piezas:
        e = {"nombre": p["nombre"], "vertices": p["vertices"], "triangulos": p["triangulos"], "tipo_indice": p["tipo_indice"],
             "max_pesos": p["max_pesos"] if p["con_pesos"] else None, "uvs": p["uvs"], "material": p["material"],
             "huesos_usados": p["huesos_usados"]}
        if p.get("paleta_local"):
            e["paleta_local"] = p["paleta_local"]
        if p.get("paleta_global") is not None:
            e["paleta_global"] = p["paleta_global"]
        lista.append(e)
    d["piezas"] = {
        "numero": V(len(piezas), "medido"),
        "tipo_indice": V(tipo, "medido") if tipo else V(None, "medido", "sin indices"),
        "max_vertices_por_pieza": V(lim, "inferido", f"limite implicito del tipo de indice {tipo}" + (" (65535 reserva el reinicio de primitiva)" if tipo == "uint16" else "")) if lim else V(None, "inferido"),
        "maximos_observados": {"origen": "medido", "vertices": max(p["vertices"] for p in piezas), "triangulos": max(p["triangulos"] for p in piezas)},
        "totales": {"origen": "medido", "vertices": sum(p["vertices"] for p in piezas), "triangulos": sum(p["triangulos"] for p in piezas)},
        "# division": "por pieza: vertices, triangulos y huesos usados (indices de esqueleto/paleta global; paleta_local = slots del buffer)",
        "division": {"origen": "medido", "lista": lista}}
    # --- escala y ejes
    d["escala"] = {
        "altura": V(round(ext[iu], 5), "medido", f"extension de la caja sobre el eje arriba ({fmt_eje(iu, su)}), en unidades del archivo"),
        "ancho": V(round(ext[ilat], 5), "medido"), "profundidad": V(round(ext[ifr], 5), "medido"),
        "caja": {"origen": "medido", "min": bb[0], "max": bb[1]}}
    d["orientacion"] = {"eje_arriba": V(fmt_eje(iu, su), ejes["origen_arriba"]), "eje_frente": V(fmt_eje(ifr, sf), ejes["origen_frente"])}
    for k, n in enumerate(ejes["notas"]):
        d["orientacion"][f"# nota{k}"] = n
    # --- materiales, UVs
    d["materiales"] = {"origen": "medido", "lista": mod["materiales"]} if mod["materiales"] else \
        {"origen": "medido", "lista": [], "# nota": "el original no trae materiales/texturas; pasa --texturas para registrar sus resoluciones"}
    d["uvs"] = {"canales": V(max(p["uvs"] for p in piezas), "medido", "maximo de conjuntos de UV por pieza")}
    # --- motor (manual)
    d["motor"] = {
        "origen": "manual",
        "# 1": "LIMITES DEL MOTOR que no se ven en el archivo: rellenalos a mano (null = sin dato). Los pasos siguientes los usan ademas",
        "# 2": "de lo medido arriba. Ejemplos: tamano de la paleta de huesos del shader, tope de triangulos, formatos de textura.",
        "max_huesos_por_pieza": None, "# max_huesos_por_pieza": "p. ej. registros de constantes disponibles para la paleta de huesos",
        "max_huesos_total": None, "max_pesos_por_vertice": None, "# max_pesos_por_vertice": "solo si el motor impone menos que lo observado",
        "max_triangulos_por_pieza": None, "max_triangulos_total": None, "max_materiales": None,
        "tamano_maximo_textura": None, "formato_textura": None, "# formato_textura": "p. ej. BC7/DXT5, con o sin mipmaps",
        "tipo_vertice_exigido": None, "# tipo_vertice_exigido": "p. ej. stride/atributos que el shader del juego lee",
        "notas": None}
    d["avisos"] = avisos
    return d


# ---------------------------------------------------------------- YAML con comentarios
_SEGURO = re.compile(r"[A-Za-z_][A-Za-z0-9_./+-]*\Z")
_RESERVADOS = {"true", "false", "null", "yes", "no", "on", "off", "~"}


def _esc(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(round(v, 6)) if math.isfinite(v) else "null"
    if isinstance(v, int):
        return str(v)
    s = str(v)
    if _SEGURO.match(s) and s.lower() not in _RESERVADOS:
        return s
    return json.dumps(s, ensure_ascii=False)


def _flujo(v):
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_flujo(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_flujo(x) for x in v) + "]"
    return _esc(v)


def _es_hoja(v):
    return isinstance(v, dict) and set(v) <= {"valor", "origen", "nota"} and "valor" in v


def _emitir(v, ind, out, clave_lista=False):
    pad = "  " * ind
    for k, x in v.items():
        if k.startswith("#"):
            for linea in str(x).splitlines():
                out.append(f"{pad}# {linea}")
            continue
        if _es_hoja(x):
            nota = x.get("nota")
            h = {"valor": x["valor"], "origen": x["origen"]}
            out.append(f"{pad}{k}: {_flujo(h)}" + (f"  # {nota}" if nota else ""))
        elif isinstance(x, dict):
            out.append(f"{pad}{k}:")
            _emitir(x, ind + 1, out)
        elif isinstance(x, list) and x and all(isinstance(e, dict) for e in x):
            out.append(f"{pad}{k}:")
            for e in x:
                out.append(f"{pad}  - {_flujo(e)}")
        elif isinstance(x, list) and not x:
            out.append(f"{pad}{k}: []")
        elif isinstance(x, list) and any(isinstance(e, str) and len(e) > 40 for e in x):
            out.append(f"{pad}{k}:")
            for e in x:
                out.append(f"{pad}  - {_esc(e)}")
        else:
            out.append(f"{pad}{k}: {_flujo(x)}")


def a_yaml(d):
    out = []
    _emitir(d, 0, out)
    return "\n".join(out) + "\n"
