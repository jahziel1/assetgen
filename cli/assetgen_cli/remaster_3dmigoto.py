"""Adaptador 3DMigoto para `assetgen remaster analizar`: lee piezas `.vb` (vertices) + `.ib` (indices) volcadas por 3DMigoto
con un `layout.yaml` que describe el vertice. Nada de ningun juego va fijo aqui; el layout lo pone el usuario.

layout.yaml (ejemplo en cli/ejemplos/layout_3dmigoto.yaml):
  atributos: "3f p, 3f n, 4h huesos, 3f pesos, 2f uv"   # <n><tipo>[n] <nombre>; tipos f h H b B i I e; sufijo n = normalizado
  stride: 52                  # opcional: se comprueba contra la suma de los atributos
  indices: uint16             # uint16 | uint32 | auto
  huesos_divisor: 3           # opcional: el indice de hueso del buffer es un registro (indice local = valor / divisor)
  orden_bytes: little         # little | big
  tolerar_cola: false         # true: un vertex buffer con bytes sobrantes al final (relleno del volcado) se recorta al ultimo vertice entero
  unidades: m                 # opcional
  ejes: {arriba: +y, frente: +z}   # opcional; si falta se infieren (inferido, dudoso) o se pasan por linea de comandos
Nombres reconocidos: p/pos/posicion/position, n/normal, huesos/bones/joints, pesos/weights, uv/uv0/uv1.../texcoord, color, tangente.

Un 3DMigoto no trae esqueleto en el buffer: los huesos son indices LOCALES a la paleta de cada pieza. Entradas opcionales:
  paletas: {pieza: [hueso_global_del_slot_0, hueso_global_del_slot_1, ...]}   (null para slots sin usar)
  esqueleto: ver remaster_esqueleto (glb/json/yaml)
  pose: matrices por hueso global (12 floats = 3 columnas de la base + origen, o 16 = fila mayor 4x4); con paletas+pose se
        aplica skinning lineal para llevar cada pieza a un espacio comun (si no, las piezas quedan en su espacio bind y pueden
        no ensamblarse)."""
import hashlib, math, os, re, struct

from .remaster_comun import modelo_vacio, muestrear, tam_imagen, leer_datos

TAM = {"f": 4, "h": 2, "H": 2, "b": 1, "B": 1, "i": 4, "I": 4, "e": 2}
NORMAL = {"h": 32767.0, "H": 65535.0, "b": 127.0, "B": 255.0}
SEM = {"p": "posicion", "pos": "posicion", "posicion": "posicion", "position": "posicion",
       "n": "normal", "normal": "normal", "normales": "normal",
       "huesos": "huesos", "bones": "huesos", "joints": "huesos", "blendindices": "huesos",
       "pesos": "pesos", "weights": "pesos", "blendweights": "pesos",
       "uv": "uv", "texcoord": "uv", "color": "color", "tangente": "tangente", "tangent": "tangente"}


class Layout:
    def __init__(self, atributos, stride, indices, huesos_divisor, orden, unidades, ejes, tolerar_cola=False):
        self.atributos, self.stride, self.indices = atributos, stride, indices
        self.huesos_divisor, self.orden, self.unidades, self.ejes = huesos_divisor, orden, unidades, ejes
        self.tolerar_cola = tolerar_cola

    def attr(self, sem):
        return [a for a in self.atributos if a["sem"] == sem]


def parsear_atributos(texto):
    """'3f p, 3f n, 4h huesos' -> [{n, tipo, norm, nombre, sem, off, size, fmt}]"""
    attrs, off = [], 0
    for tok in [t.strip() for t in str(texto).split(",") if t.strip()]:
        m = re.fullmatch(r"(\d*)([fhHbBiIe])(n?)\s+([A-Za-z_]\w*)", tok)
        if not m:
            raise SystemExit(f"layout: no entiendo el atributo '{tok}' (forma: '3f posicion', tipos f h H b B i I e)")
        n = int(m.group(1) or 1)
        nombre = m.group(4)
        base = re.sub(r"\d+$", "", nombre.lower())
        sem = SEM.get(nombre.lower()) or SEM.get(base)
        if sem is None:
            raise SystemExit(f"layout: nombre de atributo desconocido '{nombre}' (usa p, n, huesos, pesos, uv, color, tangente)")
        tam = TAM[m.group(2)] * n
        attrs.append({"n": n, "tipo": m.group(2), "norm": bool(m.group(3)), "nombre": nombre, "sem": sem, "off": off, "size": tam})
        off += tam
    if not any(a["sem"] == "posicion" and a["n"] >= 3 for a in attrs):
        raise SystemExit("layout: falta el atributo de posicion (p) con 3 componentes")
    return attrs, off


def leer_layout(ruta):
    d = leer_datos(ruta)
    if not isinstance(d, dict) or "atributos" not in d:
        raise SystemExit(f"{os.path.basename(ruta)}: el layout necesita 'atributos'")
    attrs, suma = parsear_atributos(d["atributos"])
    stride = int(d.get("stride", suma))
    if stride != suma:
        raise SystemExit(f"layout: stride {stride} no coincide con la suma de los atributos ({suma} bytes)")
    ind = str(d.get("indices", "auto")).lower()
    if ind not in ("uint16", "uint32", "auto"):
        raise SystemExit("layout: 'indices' debe ser uint16, uint32 o auto")
    orden = str(d.get("orden_bytes", "little")).lower()
    if orden not in ("little", "big"):
        raise SystemExit("layout: 'orden_bytes' debe ser little o big")
    ejes = d.get("ejes")
    if ejes and not (ejes.get("arriba") and ejes.get("frente")):
        raise SystemExit("layout: 'ejes' necesita 'arriba' y 'frente'")
    return Layout(attrs, stride, ind, int(d.get("huesos_divisor", 1) or 1), orden, d.get("unidades"), ejes,
                  bool(d.get("tolerar_cola", False)))


def _formato_struct(lay):
    return ("<" if lay.orden == "little" else ">") + "".join(("%d%s" % (a["n"], a["tipo"])) for a in lay.atributos)


def listar_piezas(entrada):
    """Archivo .vb o carpeta -> [(nombre, ruta_vb, ruta_ib)] ordenado por nombre."""
    if os.path.isfile(entrada):
        vbs = [entrada]
    else:
        vbs = sorted(os.path.join(entrada, f) for f in os.listdir(entrada) if f.lower().endswith(".vb"))
    if not vbs:
        raise SystemExit("no hay archivos .vb en la entrada")
    piezas = []
    for vb in vbs:
        ib = vb[:-3] + ".ib"
        if not os.path.exists(ib):
            raise SystemExit(f"falta el .ib de {os.path.basename(vb)}")
        piezas.append((os.path.splitext(os.path.basename(vb))[0], vb, ib))
    return piezas


def _leer_indices(datos, tipo, orden, nverts):
    pref = "<" if orden == "little" else ">"
    cand = ["uint16", "uint32"] if tipo == "auto" else [tipo]
    ultimo = None
    for t in cand:
        tam = 2 if t == "uint16" else 4
        if len(datos) % (tam * 3) and tipo == "auto":
            ultimo = f"{len(datos)} bytes no es multiplo de 3 indices {t}"
            continue
        n = len(datos) // tam
        idx = struct.unpack(pref + str(n) + ("H" if tam == 2 else "I"), datos[:n * tam])
        if idx and max(idx) >= nverts:
            ultimo = f"indice {max(idx)} fuera de rango ({nverts} vertices) leyendo como {t}"
            continue
        return idx, t
    raise SystemExit("indices invalidos: " + str(ultimo))


def _matriz12(m):
    """12 floats (3 columnas + origen) o 16 (fila mayor 4x4) -> (3x3 por filas, traslacion)."""
    if len(m) == 12:
        return [[m[0], m[3], m[6]], [m[1], m[4], m[7]], [m[2], m[5], m[8]]], (m[9], m[10], m[11])
    if len(m) == 16:
        return [[m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]], (m[3], m[7], m[11])
    raise SystemExit(f"pose: cada matriz debe tener 12 o 16 numeros (tiene {len(m)})")


def cargar_pose(ruta):
    if "#" in ruta:
        ruta, clave = ruta.split("#", 1)
    else:
        clave = None
    d = leer_datos(ruta, clave)
    if not isinstance(d, list):
        raise SystemExit("pose: se esperaba una lista de matrices por hueso global (usa 'archivo#clave' para elegirla)")
    return [_matriz12(m) for m in d]


def cargar_paletas(ruta):
    d = leer_datos(ruta)
    if not isinstance(d, dict):
        raise SystemExit("paletas: se esperaba {pieza: [huesos globales por slot]}")
    return {str(k): list(v) for k, v in d.items()}


def _paleta_de(paletas, nombre):
    """Busca la paleta por nombre exacto o por prefijo antes del primer '_' (000055_eb47a112 -> 000055)."""
    if not paletas:
        return None
    return paletas.get(nombre) or paletas.get(nombre.split("_")[0])


def analizar(entrada, lay, paletas=None, pose=None, texturas=None, esqueleto=None):
    """-> (modelo, geom). `geom` = {piezas: [{nombre, pos [(x,y,z)], nrm, uv, tris [(a,b,c)]}]} en el espacio del archivo
    (con skinning si hay paletas + pose) para los renders."""
    pref_ord = lay.orden
    fmt = struct.Struct(_formato_struct(lay))
    a_pos = lay.attr("posicion")[0]
    a_nrm, a_hue, a_pes, a_uv = lay.attr("normal"), lay.attr("huesos"), lay.attr("pesos"), lay.attr("uv")
    nombre_origen = os.path.basename(entrada.rstrip("/")) or "original"
    mod = modelo_vacio("3dmigoto", nombre_origen)
    mod["unidades"] = lay.unidades
    mod["ejes_declarados"] = dict(lay.ejes) if lay.ejes else None
    if pose and not paletas:
        raise SystemExit("--pose necesita --paletas (la paleta de cada pieza lleva el hueso local al global de la pose)")
    if (pose or paletas) and a_hue and not a_pes:
        raise SystemExit("layout: hay huesos pero no pesos; el skinning necesita ambos")

    # offsets dentro del vertice
    def idx_attr(a):
        base = sum(x["n"] for x in lay.atributos if x["off"] < a["off"])
        return base, base + a["n"]
    i_pos = idx_attr(a_pos)
    i_nrm = idx_attr(a_nrm[0]) if a_nrm else None
    i_hue = idx_attr(a_hue[0]) if a_hue else None
    i_pes = idx_attr(a_pes[0]) if a_pes else None
    i_uv = idx_attr(a_uv[0]) if a_uv else None

    def valor(a, v, i0, i1):
        r = v[i0:i1]
        if a["norm"] and a["tipo"] in NORMAL:
            return tuple(c / NORMAL[a["tipo"]] for c in r)
        return r

    geom = {"piezas": []}
    todos = []
    mn, mx = [math.inf] * 3, [-math.inf] * 3
    hay_pose = bool(pose)
    sin_paleta, con_cola = [], []
    buffers = {}      # md5 del .vb -> (vertices, indices usados por todas las piezas que lo comparten)
    for nombre, vb, ib in listar_piezas(entrada):
        datos = open(vb, "rb").read()
        if len(datos) % lay.stride:
            if not lay.tolerar_cola:
                raise SystemExit(f"{os.path.basename(vb)}: {len(datos)} bytes no es multiplo del stride {lay.stride}; revisa el layout "
                                 f"(si el volcado trae relleno al final, pon 'tolerar_cola: true' en el layout)")
            cola = len(datos) % lay.stride
            con_cola.append(f"{nombre} ({cola} B)")
            datos = datos[:len(datos) - cola]
        n = len(datos) // lay.stride
        compartido = buffers.setdefault(hashlib.md5(datos).hexdigest(), [n, set()])
        idx, tipo_idx = _leer_indices(open(ib, "rb").read(), lay.indices, lay.orden, n)
        if len(idx) % 3:
            mod["avisos"].append(f"{nombre}: {len(idx)} indices no es multiplo de 3 (se ignoran los sobrantes)")
        tris = [tuple(idx[k:k + 3]) for k in range(0, len(idx) - len(idx) % 3, 3)]
        usados = set(idx)
        compartido[1] |= usados
        pal = _paleta_de(paletas, nombre)
        if paletas and pal is None:
            sin_paleta.append(nombre)
        pos, nrm, uvs = [], [], []
        max_p, smin, smax = 0, None, None
        slots, globales = set(), set()
        for v in fmt.iter_unpack(datos):
            p = tuple(float(c) for c in valor(a_pos, v, *i_pos)[:3])
            nr = tuple(float(c) for c in valor(a_nrm[0], v, *i_nrm)[:3]) if a_nrm else None
            uvs.append(tuple(float(c) for c in valor(a_uv[0], v, *i_uv)[:2]) if a_uv else (0.0, 0.0))
            ws = []
            if a_hue and a_pes:
                hs = valor(a_hue[0], v, *i_hue)
                wv = valor(a_pes[0], v, *i_pes)
                ws = [(int(h) // lay.huesos_divisor, float(w)) for h, w in zip(hs, wv) if w > 1e-6]
                suma = sum(w for _, w in ws)
                max_p = max(max_p, len(ws))
                smin = suma if smin is None else min(smin, suma)
                smax = suma if smax is None else max(smax, suma)
                for s, _ in ws:
                    slots.add(s)
                    if pal and s < len(pal) and pal[s] is not None:
                        globales.add(int(pal[s]))
            if hay_pose and pal and ws:     # skinning lineal: p = sum w_k (P[h_k] p)
                q = [0.0, 0.0, 0.0]; qn = [0.0, 0.0, 0.0]; tw = 0.0
                for s, w in ws:
                    g = pal[s] if s < len(pal) else None
                    if g is None or g >= len(pose):
                        continue
                    R, t = pose[int(g)]
                    for r in range(3):
                        q[r] += w * (R[r][0] * p[0] + R[r][1] * p[1] + R[r][2] * p[2] + t[r])
                        if nr:
                            qn[r] += w * (R[r][0] * nr[0] + R[r][1] * nr[1] + R[r][2] * nr[2])
                    tw += w
                if tw > 0:
                    p = (q[0] / tw, q[1] / tw, q[2] / tw)
                    nr = (qn[0] / tw, qn[1] / tw, qn[2] / tw) if nr else None
            pos.append(p)
            nrm.append(nr)
        pmn = [min(p[k] for p in pos) for k in range(3)]
        pmx = [max(p[k] for p in pos) for k in range(3)]
        for k in range(3):
            mn[k], mx[k] = min(mn[k], pmn[k]), max(mx[k], pmx[k])
        todos.extend(pos)
        con = bool(a_hue and a_pes)
        if smin is not None and smin < 0.99:
            mod["avisos"].append(f"{nombre}: hay vertices cuyos pesos suman {smin:.3f} (<1)")
        mod["piezas"].append({
            "nombre": nombre, "vertices": n, "vertices_usados": len(usados), "triangulos": len(tris), "tipo_indice": tipo_idx,
            "max_pesos": max_p, "suma_pesos": [round(smin, 5), round(smax, 5)] if smin is not None else None,
            "huesos_usados": sorted(globales) if globales else sorted(slots) if not paletas else [],
            "paleta_local": sorted(slots) if con else [],
            "paleta_global": sorted(globales) if (paletas and pal) else None,
            "uvs": len(a_uv), "material": None,
            "bbox": [[round(v, 6) for v in pmn], [round(v, 6) for v in pmx]], "con_pesos": con})
        geom["piezas"].append({"nombre": nombre, "pos": pos, "nrm": nrm, "uv": uvs, "tris": tris})
    if not (hay_pose and paletas):
        mod["avisos"].append("sin --pose y --paletas las piezas quedan en su espacio bind; pueden no ensamblarse en un solo cuerpo "
                             "(la altura y los renders no son fiables)")
    if con_cola:
        mod["avisos"].append("bytes sobrantes al final del vertex buffer ignorados (tolerar_cola) en " + str(len(con_cola)) + " pieza(s): " + ", ".join(con_cola))
    if sin_paleta:
        mod["avisos"].append("piezas sin paleta en --paletas: " + ", ".join(sin_paleta))
    if a_hue and not paletas:
        mod["avisos"].append("los huesos del buffer son indices LOCALES a la paleta de cada pieza (huesos_usados = slots locales); "
                             "pasa --paletas para llevarlos a huesos globales")
    mod["vertices_unicos_usados"] = sum(len(u) for _, u in buffers.values())     # un .vb compartido por varias piezas cuenta una vez
    mod["bbox"] = [[round(v, 6) for v in mn], [round(v, 6) for v in mx]]
    mod["muestra_puntos"] = muestrear(todos)
    if texturas:
        mod["materiales"] = materiales_de_texturas(texturas)
    mod["esqueleto"] = esqueleto
    return mod, geom


def materiales_de_texturas(ruta):
    """Un material 'externo' con las imagenes dadas (archivo o carpeta): nombre y resolucion."""
    if os.path.isdir(ruta):
        archivos = sorted(os.path.join(ruta, f) for f in os.listdir(ruta) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    else:
        archivos = [ruta]
    texs = []
    for a in archivos:
        with open(a, "rb") as f:
            ancho, alto = tam_imagen(f.read())
        texs.append({"rol": "color" if not texs else "otra", "archivo": os.path.basename(a), "ancho": ancho, "alto": alto, "uv": 0})
    return [{"nombre": "texturas_aportadas", "texturas": texs, "doble_cara": None}] if texs else []
