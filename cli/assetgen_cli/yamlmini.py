"""Lector/escritor YAML minimo de respaldo (se usa solo si PyYAML no esta instalado).

Cubre lo que usan assetgen.yaml, seleccion.yaml, prompts.yaml y la config global:
mapas y listas por indentacion, listas de mapas ("- clave: v"), escalares (texto, entre comillas, enteros, decimales,
true/false, null/~), listas y mapas en una linea ([1, 2], {a: 1}) y comentarios (#). No soporta anclas, etiquetas,
bloques multilinea (| >) ni documentos multiples: si los encuentra, falla en voz alta (YamlError) en vez de adivinar.
"""
import json
import re
import sys


class YamlError(ValueError):
    pass


def _quita_comentario(s):
    """Quita ' # comentario' respetando comillas (una comilla solo abre texto si empieza un valor)."""
    q, i = None, 0
    while i < len(s):
        c = s[i]
        if q:
            if c == "\\" and q == '"':
                i += 1
            elif c == q:
                q = None
        elif c in "\"'" and (i == 0 or s[i - 1] in " \t[{,:-"):
            q = c
        elif c == "#" and (i == 0 or s[i - 1] in " \t"):
            return s[:i]
        i += 1
    return s


def _escalar(t, n=0):
    t = t.strip()
    if t == "" or t in ("null", "Null", "NULL", "~"):
        return None
    if t[0] in "&*!|>@`%":
        raise YamlError(f"linea {n}: sintaxis YAML no soportada por el lector minimo: {t[:30]!r}")
    if t[0] == '"':
        if len(t) < 2 or t[-1] != '"':
            raise YamlError(f"linea {n}: comillas sin cerrar: {t[:40]!r}")
        try:
            return json.loads(t)
        except ValueError:
            return t[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    if t[0] == "'":
        if len(t) < 2 or t[-1] != "'":
            raise YamlError(f"linea {n}: comillas sin cerrar: {t[:40]!r}")
        return t[1:-1].replace("''", "'")
    if t[0] == "[" or t[0] == "{":
        return _flujo(t, n)
    if t in ("true", "True", "TRUE"):
        return True
    if t in ("false", "False", "FALSE"):
        return False
    if re.fullmatch(r"[-+]?\d+", t):
        return int(t)
    if re.fullmatch(r"[-+]?(\d+\.\d*|\.\d+)([eE][-+]\d+)?", t):
        return float(t)
    return t


def _parte_flujo(s, n):
    """Separa por comas de primer nivel (respeta comillas y anidados)."""
    partes, nivel, q, cur = [], 0, None, ""
    for c in s:
        if q:
            cur += c
            if c == q:
                q = None
        elif c in "\"'":
            q = c; cur += c
        elif c in "[{":
            nivel += 1; cur += c
        elif c in "]}":
            nivel -= 1; cur += c
        elif c == "," and nivel == 0:
            partes.append(cur); cur = ""
        else:
            cur += c
    if q or nivel:
        raise YamlError(f"linea {n}: lista/mapa en linea mal cerrado")
    if cur.strip():
        partes.append(cur)
    return partes


def _flujo(t, n):
    if t[0] == "[":
        if t[-1] != "]":
            raise YamlError(f"linea {n}: lista en linea sin cerrar (las multilinea no estan soportadas): {t[:40]!r}")
        return [_escalar(p, n) for p in _parte_flujo(t[1:-1], n)]
    if t[-1] != "}":
        raise YamlError(f"linea {n}: mapa en linea sin cerrar: {t[:40]!r}")
    d = {}
    for p in _parte_flujo(t[1:-1], n):
        k, sep, v = _separa_clave(p.strip(), n)
        if not sep:
            raise YamlError(f"linea {n}: se esperaba 'clave: valor' en {p!r}")
        d[k] = _escalar(v, n)
    return d


def _separa_clave(s, n):
    """Devuelve (clave, True, resto) si s es 'clave: resto' o 'clave:'; si no, (None, False, s)."""
    if s and s[0] in "\"'":
        q = s[0]
        i = 1
        while i < len(s) and s[i] != q:
            i += 2 if (s[i] == "\\" and q == '"') else 1
        if i < len(s):
            resto = s[i + 1:]
            if resto.startswith(":") and (len(resto) == 1 or resto[1] in " \t"):
                return _escalar(s[:i + 1], n), True, resto[1:]
        return None, False, s
    m = re.match(r"^([^\s\[\]{},#&*!|>'\"%@`][^:#]*?|[^\s\[\]{},#&*!|>'\"%@`]):(?:\s+(.*)|$)", s)
    if m:
        return m.group(1).strip(), True, m.group(2) or ""
    return None, False, s


def _lineas(texto):
    res = []
    for n, crudo in enumerate(texto.splitlines(), 1):
        if "\t" in crudo[:len(crudo) - len(crudo.lstrip())]:
            raise YamlError(f"linea {n}: tabuladores en la indentacion no permitidos")
        s = _quita_comentario(crudo).rstrip()
        if not s.strip():
            continue
        if s.strip() in ("---", "..."):
            if res:
                raise YamlError(f"linea {n}: documentos multiples no soportados")
            continue
        res.append((len(s) - len(s.lstrip(" ")), s.strip(), n))
    return res


def _bloque(ls, i, ind):
    """Parsea el bloque que empieza en ls[i] con indentacion ind. Devuelve (valor, siguiente_i)."""
    if i >= len(ls):
        return None, i
    if ls[i][1] == "-" or ls[i][1].startswith("- "):
        return _lista(ls, i, ind)
    return _mapa(ls, i, ind)


def _continuacion(ls, i, ind, texto):
    """Escalares de varias lineas (los que PyYAML parte al escribir textos largos): las lineas mas indentadas se unen con espacio."""
    while i < len(ls) and ls[i][0] > ind:
        texto += " " + ls[i][1]; i += 1
    return texto, i


def _valor_tras_clave(ls, i, ind, resto, n):
    """Valor de 'clave: resto' donde ls[i-1] era la linea de la clave (i apunta a la siguiente)."""
    resto = resto.strip()
    if resto:
        resto, i = _continuacion(ls, i, ind, resto)
        return _escalar(resto, n), i
    if i < len(ls) and (ls[i][0] > ind or (ls[i][0] == ind and (ls[i][1] == "-" or ls[i][1].startswith("- ")))):
        return _bloque(ls, i, ls[i][0])
    return None, i


def _mapa(ls, i, ind):
    d = {}
    while i < len(ls) and ls[i][0] == ind:
        _, s, n = ls[i]
        if s == "-" or s.startswith("- "):
            break
        k, ok, resto = _separa_clave(s, n)
        if not ok:
            raise YamlError(f"linea {n}: se esperaba 'clave: valor' y hay {s[:50]!r}")
        if k in d:
            raise YamlError(f"linea {n}: clave repetida {k!r}")
        d[k], i = _valor_tras_clave(ls, i + 1, ind, resto, n)
    if i < len(ls) and ls[i][0] > ind:
        raise YamlError(f"linea {ls[i][2]}: indentacion inesperada")
    return d, i


def _lista(ls, i, ind):
    out = []
    while i < len(ls) and ls[i][0] == ind and (ls[i][1] == "-" or ls[i][1].startswith("- ")):
        _, s, n = ls[i]
        resto = s[1:].strip()
        if not resto:
            if i + 1 < len(ls) and ls[i + 1][0] > ind:
                v, i = _bloque(ls, i + 1, ls[i + 1][0])
            else:
                v, i = None, i + 1
            out.append(v); continue
        # "- clave: valor ..." abre un mapa cuyo contenido se alinea con la clave
        k, ok, _r = _separa_clave(resto, n)
        if ok:
            hueco = len(s) - len(s[1:].lstrip())          # columna relativa de la clave
            sub = [(ind + hueco, resto, n)]
            j = i + 1
            while j < len(ls) and ls[j][0] > ind:
                sub.append(ls[j]); j += 1
            v, fin = _mapa(sub, 0, ind + hueco)
            if fin != len(sub):
                raise YamlError(f"linea {sub[fin][2]}: indentacion inesperada en elemento de lista")
            out.append(v); i = j
        else:
            resto, i = _continuacion(ls, i + 1, ind, resto)
            out.append(_escalar(resto, n))
    if i < len(ls) and ls[i][0] > ind:
        raise YamlError(f"linea {ls[i][2]}: indentacion inesperada")
    return out, i


def safe_load(f):
    """Como yaml.safe_load: acepta texto o archivo abierto."""
    texto = f if isinstance(f, str) else f.read()
    ls = _lineas(texto)
    if not ls:
        return None
    if len(ls) == 1 and not (ls[0][1] == "-" or ls[0][1].startswith("- ")) and not _separa_clave(ls[0][1], ls[0][2])[1]:
        return _escalar(ls[0][1], ls[0][2])
    v, i = _bloque(ls, 0, ls[0][0])
    if i != len(ls):
        raise YamlError(f"linea {ls[i][2]}: contenido inesperado")
    return v


# ---------------------------------------------------------------- escritura
_RESERVADAS = {"true", "false", "null", "~", "yes", "no", "on", "off", ""}


def _fmt_escalar(v):
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v)
    if (s.lower() in _RESERVADAS or re.fullmatch(r"[-+]?[\d.][\d.eE+-]*", s) or s != s.strip() or "\n" in s
            or re.search(r"(^[-?:,\[\]{}#&*!|>'\"%@`])|(: )|( #)|(:$)", s)):
        return json.dumps(s, ensure_ascii=False)
    return s


def _vol(v, ind, out):
    pad = " " * ind
    if isinstance(v, dict) and v:
        for k, x in v.items():
            if isinstance(x, (dict, list)) and x:
                out.append(f"{pad}{_fmt_escalar(k)}:")
                _vol(x, ind + 2 if isinstance(x, dict) else ind + 2, out)
            else:
                out.append(f"{pad}{_fmt_escalar(k)}: {'{}' if x == {} and isinstance(x, dict) else '[]' if isinstance(x, list) else _fmt_escalar(x)}")
    elif isinstance(v, list) and v:
        for x in v:
            if isinstance(x, (dict, list)) and x:
                sub = []
                _vol(x, ind + 2, sub)
                out.append(f"{pad}- {sub[0].lstrip()}")
                out.extend(sub[1:])
            else:
                out.append(f"{pad}- {'{}' if isinstance(x, dict) else '[]' if isinstance(x, list) else _fmt_escalar(x)}")
    else:
        out.append(pad + ("{}" if isinstance(v, dict) else "[]" if isinstance(v, list) else _fmt_escalar(v)))


def safe_dump(datos, f=None, allow_unicode=True, sort_keys=False, **_):
    out = []
    _vol(datos, 0, out)
    texto = "\n".join(out) + "\n"
    if f is None:
        return texto
    f.write(texto)


def yaml():
    """PyYAML si esta instalado; si no, este mismo modulo (misma interfaz: safe_load, safe_dump)."""
    try:
        import yaml as y
        return y
    except ImportError:
        return sys.modules[__name__]


def leer(ruta):
    """Lee un archivo YAML; cualquier fallo (permiso, sintaxis) sube como SystemExit con la ruta, nunca se ignora."""
    try:
        with open(ruta, encoding="utf-8") as f:
            return yaml().safe_load(f)
    except Exception as e:
        raise SystemExit(f"No se pudo leer {ruta}: {e}")
