"""Adaptador glTF/GLB para `assetgen remaster analizar`: lee el archivo directamente (sin Blender) y devuelve el
"modelo" normalizado que consume remaster_restricciones (ver `remaster_comun.modelo_vacio`).

Una pieza = una primitiva (una llamada de dibujo). Los conteos son los de la GPU (los del buffer del archivo, no los que
reporta Blender tras fusionar vertices). Soporta .glb y .gltf (buffers embebidos o externos), TRS/matrices de nodos,
skins (esqueleto, huesos usados por pieza, pesos maximos), UVs, materiales e imagenes (resolucion leida de la cabecera).
No soporta accesores dispersos (sparse) ni compresion Draco/meshopt: avisa y sigue."""
import base64, json, math, os, struct

from .remaster_comun import modelo_vacio, muestrear, tam_imagen

COMP = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
NORMALIZA = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}
TIPO_INDICE = {5121: "uint8", 5123: "uint16", 5125: "uint32"}
ROLES = (("baseColorTexture", "color"), ("metallicRoughnessTexture", "metal_rugosidad"), ("normalTexture", "normal"),
         ("occlusionTexture", "oclusion"), ("emissiveTexture", "emision"))


def cargar(ruta):
    """-> (json, chunk binario del .glb o None)."""
    with open(ruta, "rb") as f:
        datos = f.read()
    if datos[:4] == b"glTF":
        j, binario, off = None, None, 12
        while off + 8 <= len(datos):
            n, tipo = struct.unpack_from("<II", datos, off)
            cuerpo = datos[off + 8:off + 8 + n]
            if tipo == 0x4E4F534A:
                j = json.loads(cuerpo.decode("utf-8"))
            elif tipo == 0x004E4942 and binario is None:
                binario = cuerpo
            off += 8 + n + (-n % 4)
        if j is None:
            raise SystemExit(f"{os.path.basename(ruta)}: GLB sin chunk JSON")
        return j, binario
    return json.loads(datos.decode("utf-8")), None


class _Doc:
    def __init__(self, ruta):
        self.ruta, self.dir = ruta, os.path.dirname(os.path.abspath(ruta))
        self.j, self.glb_bin = cargar(ruta)
        self._bufs = {}
        self.avisos = []

    def buffer(self, i):
        if i not in self._bufs:
            b = self.j["buffers"][i]
            uri = b.get("uri")
            if uri is None:
                self._bufs[i] = self.glb_bin or b""
            elif uri.startswith("data:"):
                self._bufs[i] = base64.b64decode(uri.split(",", 1)[1])
            else:
                with open(os.path.join(self.dir, _decodifica(uri)), "rb") as f:
                    self._bufs[i] = f.read()
        return self._bufs[i]

    def vista(self, i):
        bv = self.j["bufferViews"][i]
        return self.buffer(bv["buffer"]), bv.get("byteOffset", 0), bv.get("byteLength", 0), bv.get("byteStride")

    def accesor(self, i, normalizar=True):
        """-> lista de tuplas (o de escalares si SCALAR)."""
        a = self.j["accessors"][i]
        n, nc = a["count"], NCOMP[a["type"]]
        ch, tam = COMP[a["componentType"]]
        if "bufferView" not in a:
            return [0] * n if nc == 1 else [(0,) * nc] * n
        if "sparse" in a:
            self.avisos.append(f"accesor {i} con 'sparse': se ignora la parte dispersa")
        buf, off, _, stride = self.vista(a["bufferView"])
        off += a.get("byteOffset", 0)
        elem = tam * nc
        stride = stride or elem
        fmt = "<" + ch * nc
        div = NORMALIZA.get(a["componentType"]) if (a.get("normalized") and normalizar) else None
        if stride == elem:
            filas = list(struct.iter_unpack(fmt, buf[off:off + n * elem]))
        else:
            filas = [struct.unpack_from(fmt, buf, off + k * stride) for k in range(n)]
        if div:
            filas = [tuple(max(c / div, -1.0) for c in f) for f in filas]
        return [f[0] for f in filas] if nc == 1 else filas

    def bytes_imagen(self, img):
        if "bufferView" in img:
            buf, off, ln, _ = self.vista(img["bufferView"])
            return buf[off:off + ln]
        uri = img.get("uri", "")
        if uri.startswith("data:"):
            return base64.b64decode(uri.split(",", 1)[1])
        ruta = os.path.join(self.dir, _decodifica(uri))
        if os.path.exists(ruta):
            with open(ruta, "rb") as f:
                return f.read()
        self.avisos.append(f"imagen externa no encontrada: {uri}")
        return b""


def _decodifica(uri):
    from urllib.parse import unquote
    return unquote(uri)


# ---------------------------------------------------------------- matrices (fila mayor 4x4 como lista de 16)
def _mul(a, b):
    return [sum(a[r * 4 + k] * b[k * 4 + c] for k in range(4)) for r in range(4) for c in range(4)]


def _matriz_nodo(n):
    if "matrix" in n:                       # glTF guarda columna mayor
        m = n["matrix"]
        return [m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14], m[3], m[7], m[11], m[15]]
    tx, ty, tz = n.get("translation", (0, 0, 0))
    x, y, z, w = n.get("rotation", (0, 0, 0, 1))
    sx, sy, sz = n.get("scale", (1, 1, 1))
    return [(1 - 2 * (y * y + z * z)) * sx, 2 * (x * y - z * w) * sy, 2 * (x * z + y * w) * sz, tx,
            2 * (x * y + z * w) * sx, (1 - 2 * (x * x + z * z)) * sy, 2 * (y * z - x * w) * sz, ty,
            2 * (x * z - y * w) * sx, 2 * (y * z + x * w) * sy, (1 - 2 * (x * x + y * y)) * sz, tz,
            0, 0, 0, 1]


def _mundo(j):
    """-> (matrices mundo por nodo, padre por nodo)."""
    nodos = j.get("nodes", [])
    padre = {}
    for i, n in enumerate(nodos):
        for c in n.get("children", []):
            padre[c] = i
    mundo = {}

    def calc(i, pila=()):
        if i in mundo:
            return mundo[i]
        if i in pila:
            return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        m = _matriz_nodo(nodos[i])
        if i in padre:
            m = _mul(calc(padre[i], pila + (i,)), m)
        mundo[i] = m
        return m
    for i in range(len(nodos)):
        calc(i)
    return mundo, padre


def _aplica(m, p):
    x, y, z = p
    return (m[0] * x + m[1] * y + m[2] * z + m[3], m[4] * x + m[5] * y + m[6] * z + m[7], m[8] * x + m[9] * y + m[10] * z + m[11])


# ---------------------------------------------------------------- esqueleto
def esqueleto_de(doc, mundo, padre, idx_skin=0):
    """Esqueleto del skin `idx_skin`: huesos en el orden de skin.joints; reposo = traslacion mundo del nodo."""
    skins = doc.j.get("skins", [])
    if not skins:
        return None
    sk = skins[idx_skin]
    nodos = doc.j.get("nodes", [])
    joints = sk["joints"]
    en_skin = {n: i for i, n in enumerate(joints)}
    huesos = []
    for i, n in enumerate(joints):
        p = padre.get(n)
        while p is not None and p not in en_skin:
            p = padre.get(p)
        m = mundo[n]
        huesos.append({"indice": i, "nombre": nodos[n].get("name", f"nodo{n}"),
                       "padre": en_skin[p] if p is not None else -1, "reposo": [round(m[3], 6), round(m[7], 6), round(m[11], 6)]})
    if len(skins) > 1:
        doc.avisos.append(f"el archivo tiene {len(skins)} skins; el esqueleto sale del primero ({len(joints)} huesos)")
    return {"fuente": "skin del archivo", "huesos": huesos}


# ---------------------------------------------------------------- materiales
def _materiales(doc, usados):
    j = doc.j
    salida, vistos = [], {}
    for mi in usados:
        if mi is None:
            continue
        m = j.get("materials", [])[mi] if mi < len(j.get("materials", [])) else {}
        pbr = m.get("pbrMetallicRoughness", {})
        texs = []
        for clave, rol in ROLES:
            info = pbr.get(clave) or m.get(clave)
            if not info:
                continue
            t = j["textures"][info["index"]]
            src = t.get("source")
            if src is None:
                for ext in (t.get("extensions") or {}).values():
                    if isinstance(ext, dict) and "source" in ext:
                        src = ext["source"]
            if src is None:
                continue
            img = j["images"][src]
            if src not in vistos:
                ancho, alto = tam_imagen(doc.bytes_imagen(img))
                vistos[src] = {"archivo": img.get("name") or os.path.basename(img.get("uri", "")) or f"imagen{src}",
                               "ancho": ancho, "alto": alto, "mime": img.get("mimeType")}
            texs.append(dict(vistos[src], rol=rol, uv=info.get("texCoord", 0)))
        salida.append({"nombre": m.get("name", f"material{mi}"), "texturas": texs,
                       "doble_cara": bool(m.get("doubleSided", False))})
    return salida


# ---------------------------------------------------------------- analisis
def analizar(ruta):
    doc = _Doc(ruta)
    j = doc.j
    mod = modelo_vacio("gltf", os.path.basename(ruta))
    mod["unidades"] = "m"
    mundo, padre = _mundo(j)
    nodos = j.get("nodes", [])
    mod["esqueleto"] = esqueleto_de(doc, mundo, padre)
    joints_skin = {}
    for si, sk in enumerate(j.get("skins", [])):
        joints_skin[si] = sk["joints"]

    escena = j.get("scenes", [{}])[j.get("scene", 0)] if j.get("scenes") else {}
    raices = escena.get("nodes", list(range(len(nodos))))
    en_escena, pila = set(), list(raices)
    while pila:
        n = pila.pop()
        if n not in en_escena:
            en_escena.add(n)
            pila.extend(nodos[n].get("children", []))

    puntos, mat_usados = [], []
    mn, mx = [math.inf] * 3, [-math.inf] * 3
    for ni in sorted(en_escena):
        nd = nodos[ni]
        if "mesh" not in nd:
            continue
        malla = j["meshes"][nd["mesh"]]
        skin = nd.get("skin")
        m_mundo = None if skin is not None else mundo[ni]       # la especificacion ignora la matriz de un nodo con skin
        for pi, prim in enumerate(malla.get("primitives", [])):
            at = prim.get("attributes", {})
            if "POSITION" not in at:
                continue
            base = nd.get("name") or malla.get("name") or f"malla{nd['mesh']}"
            nombre = base if len(malla["primitives"]) == 1 else f"{base}.{pi}"
            pos = doc.accesor(at["POSITION"])
            if m_mundo is not None:
                pos = [_aplica(m_mundo, p) for p in pos]
            for k in range(3):
                mn[k] = min(mn[k], min(p[k] for p in pos))
                mx[k] = max(mx[k], max(p[k] for p in pos))
            puntos.append(pos)
            modo = prim.get("mode", 4)
            if "indices" in prim:
                ia = j["accessors"][prim["indices"]]
                nidx, tipo_idx = ia["count"], TIPO_INDICE.get(ia["componentType"], "desconocido")
            else:
                nidx, tipo_idx = len(pos), None
            tri = nidx // 3 if modo == 4 else max(nidx - 2, 0) if modo in (5, 6) else 0
            if modo not in (4, 5, 6):
                doc.avisos.append(f"{nombre}: modo de primitiva {modo} (no son triangulos)")
            # pesos / huesos
            sets_j = sorted(k for k in at if k.startswith("JOINTS_"))
            sets_w = sorted(k for k in at if k.startswith("WEIGHTS_"))
            max_p, smin, smax, usados = 0, None, None, set()
            if sets_j and sets_w and skin is not None:
                J = [doc.accesor(at[k]) for k in sets_j]
                W = [doc.accesor(at[k]) for k in sets_w]
                for v in range(len(pos)):
                    cuenta, suma = 0, 0.0
                    for js, ws in zip(J, W):
                        for jj, ww in zip(js[v], ws[v]):
                            if ww > 1e-6:
                                cuenta += 1; suma += ww; usados.add(int(jj))
                    max_p = max(max_p, cuenta)
                    smin = suma if smin is None else min(smin, suma)
                    smax = suma if smax is None else max(smax, suma)
            n_uv = len([k for k in at if k.startswith("TEXCOORD_")])
            mat = prim.get("material")
            if mat is not None and mat not in mat_usados:
                mat_usados.append(mat)
            mn_p = [min(p[k] for p in pos) for k in range(3)]
            mx_p = [max(p[k] for p in pos) for k in range(3)]
            mod["piezas"].append({
                "nombre": nombre, "vertices": len(pos), "triangulos": tri, "tipo_indice": tipo_idx,
                "max_pesos": max_p, "suma_pesos": [round(smin, 5), round(smax, 5)] if smin is not None else None,
                "huesos_usados": sorted(usados), "uvs": n_uv,
                "material": (j.get("materials", [{}])[mat].get("name", f"material{mat}") if mat is not None and mat < len(j.get("materials", [])) else None),
                "bbox": [[round(v, 6) for v in mn_p], [round(v, 6) for v in mx_p]],
                "con_pesos": bool(sets_j and sets_w and skin is not None)})
    if not mod["piezas"]:
        raise SystemExit(f"{os.path.basename(ruta)}: no se encontro ninguna malla en la escena")
    mod["bbox"] = [[round(v, 6) for v in mn], [round(v, 6) for v in mx]]
    mod["muestra_puntos"] = muestrear([p for pos in puntos for p in pos])
    mod["materiales"] = _materiales(doc, mat_usados)
    mod["avisos"].extend(doc.avisos)
    if mod["esqueleto"] is None:
        mod["avisos"].append("el archivo no trae esqueleto (skin) ni pesos: pasa --esqueleto con uno aparte si lo necesitas")
    return mod


def esqueleto_externo(ruta):
    """Esqueleto (skin 0) de un glTF/GLB aparte, para originales 3dmigoto sin esqueleto en el buffer."""
    doc = _Doc(ruta)
    mundo, padre = _mundo(doc.j)
    esq = esqueleto_de(doc, mundo, padre)
    if esq is None:
        raise SystemExit(f"{os.path.basename(ruta)}: no tiene skin (esqueleto)")
    return esq
