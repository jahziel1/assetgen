"""Pruebas de `assetgen remaster analizar` con datos sinteticos diminutos (sin Blender, sin RunPod, sin datos de ningun juego).
`python3 -m unittest discover -s tests` desde cli/."""
import json, os, struct, sys, tempfile, unittest, zlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from assetgen_cli import remaster as R, remaster_3dmigoto as M, remaster_comun as C, remaster_gltf as G, remaster_restricciones as RES, yamlmini

LAYOUT = """atributos: "3f p, 3f n, 4h huesos, 3f pesos, 2f uv"
stride: 52
indices: uint16
huesos_divisor: 3
unidades: m
ejes:
  arriba: +y
  frente: +z
"""


def vertice(p, w, h, uv=(0.5, 0.5)):
    return struct.pack("<3f3f4h3f2f", *p, 0, 1, 0, *h, *w, *uv)


def escribe_pieza(d, nombre, verts, idx):
    with open(os.path.join(d, nombre + ".vb"), "wb") as f:
        f.write(b"".join(verts))
    with open(os.path.join(d, nombre + ".ib"), "wb") as f:
        f.write(struct.pack("<%dH" % len(idx), *idx))


def pieza_falsa(d, nombre="pieza_a"):
    """3 vertices: 1 peso, 2 pesos, 3 pesos; indices de hueso locales = registros 0, 3, 6, 9."""
    escribe_pieza(d, nombre, [vertice((0, 0, 0), (1, 0, 0), (0, 0, 0, 0)),
                              vertice((1, 0, 0), (0.5, 0.5, 0), (3, 6, 0, 0)),
                              vertice((0, 2, 0), (0.2, 0.3, 0.5), (3, 6, 9, 0))], [0, 1, 2])


class Layout3DMigoto(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.lay_path = os.path.join(self.d, "layout.yaml")
        open(self.lay_path, "w").write(LAYOUT)

    def test_atributos_y_stride(self):
        lay = M.leer_layout(self.lay_path)
        self.assertEqual(lay.stride, 52)
        self.assertEqual([a["sem"] for a in lay.atributos], ["posicion", "normal", "huesos", "pesos", "uv"])
        self.assertEqual(lay.atributos[2]["off"], 24)
        self.assertEqual(lay.ejes["arriba"], "+y")

    def test_stride_incorrecto(self):
        open(self.lay_path, "w").write(LAYOUT.replace("stride: 52", "stride: 50"))
        with self.assertRaises(SystemExit):
            M.leer_layout(self.lay_path)

    def test_atributo_desconocido(self):
        with self.assertRaises(SystemExit):
            M.parsear_atributos("3f p, 2f cosa")

    def test_tres_vertices(self):
        pieza_falsa(self.d)
        mod, geom = M.analizar(self.d, M.leer_layout(self.lay_path))
        p = mod["piezas"][0]
        self.assertEqual((p["vertices"], p["triangulos"], p["max_pesos"], p["tipo_indice"], p["uvs"]), (3, 1, 3, "uint16", 1))
        self.assertEqual(p["paleta_local"], [0, 1, 2, 3])      # registros 0,3,6,9 / divisor 3
        self.assertEqual(p["suma_pesos"], [1.0, 1.0])
        self.assertEqual(mod["bbox"], [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
        self.assertEqual(len(geom["piezas"][0]["tris"]), 1)
        self.assertTrue(any("LOCALES" in a for a in mod["avisos"]))      # sin paletas lo avisa

    def test_paletas_y_pose(self):
        pieza_falsa(self.d)
        pose = [([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (0, 0, 0)) for _ in range(10)]
        pose[7] = ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (0, 10, 0))       # el hueso global 7 sube 10
        pal = {"pieza_a": [4, 7, 8, 9]}
        mod, geom = M.analizar(self.d, M.leer_layout(self.lay_path), pal, pose)
        p = mod["piezas"][0]
        self.assertEqual(p["paleta_global"], [4, 7, 8, 9])
        self.assertEqual(p["huesos_usados"], [4, 7, 8, 9])
        v2 = geom["piezas"][0]["pos"][1]                                  # pesos .5 slot1 (g7) + .5 slot2 (g8)
        self.assertAlmostEqual(v2[1], 5.0)

    def test_pose_sin_paletas(self):
        pieza_falsa(self.d)
        with self.assertRaises(SystemExit):
            M.analizar(self.d, M.leer_layout(self.lay_path), None, [([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (0, 0, 0))])

    def test_vb_de_tamano_incorrecto(self):
        pieza_falsa(self.d)
        with open(os.path.join(self.d, "pieza_a.vb"), "ab") as f:
            f.write(b"\0\0\0")
        with self.assertRaises(SystemExit):
            M.analizar(self.d, M.leer_layout(self.lay_path))

    def test_cola_sobrante_tolerada(self):
        pieza_falsa(self.d)
        with open(os.path.join(self.d, "pieza_a.vb"), "ab") as f:
            f.write(b"\0" * 16)
        with self.assertRaises(SystemExit):                              # por defecto sigue siendo un error
            M.analizar(self.d, M.leer_layout(self.lay_path))
        open(self.lay_path, "w").write(LAYOUT + "tolerar_cola: true\n")
        mod, geom = M.analizar(self.d, M.leer_layout(self.lay_path))
        self.assertEqual(mod["piezas"][0]["vertices"], 3)
        self.assertTrue(any("sobrantes" in a for a in mod["avisos"]))

    def test_vertex_buffer_compartido(self):
        """Dos dibujos con el mismo .vb y distinto .ib: los vertices unicos usados se cuentan una vez."""
        v = [vertice((0, 0, 0), (1, 0, 0), (0, 0, 0, 0)), vertice((1, 0, 0), (1, 0, 0), (0, 0, 0, 0)),
             vertice((0, 1, 0), (1, 0, 0), (0, 0, 0, 0)), vertice((1, 1, 0), (1, 0, 0), (0, 0, 0, 0))]
        escribe_pieza(self.d, "a", v, [0, 1, 2])
        escribe_pieza(self.d, "b", v, [1, 3, 2])
        mod, _ = M.analizar(self.d, M.leer_layout(self.lay_path))
        self.assertEqual([p["vertices_usados"] for p in mod["piezas"]], [3, 3])
        self.assertEqual(mod["vertices_unicos_usados"], 4)
        self.assertEqual(sum(p["vertices"] for p in mod["piezas"]), 8)

    def test_indices_auto_uint32(self):
        open(self.lay_path, "w").write(LAYOUT.replace("indices: uint16", "indices: auto"))
        escribe_pieza(self.d, "p", [vertice((0, 0, 0), (1, 0, 0), (0, 0, 0, 0)), vertice((1, 0, 0), (1, 0, 0), (0, 0, 0, 0)),
                                    vertice((0, 1, 0), (1, 0, 0), (0, 0, 0, 0))], [0, 1, 2])
        with open(os.path.join(self.d, "p.ib"), "wb") as f:
            f.write(struct.pack("<3I", 0, 1, 2))                         # 12 bytes: uint16 daria 6 indices validos: gana el primero valido
        mod, _ = M.analizar(self.d, M.leer_layout(self.lay_path))
        self.assertIn(mod["piezas"][0]["tipo_indice"], ("uint16", "uint32"))


def glb_sintetico(ruta, con_piel=True):
    """GLB minimo: 1 triangulo (uint16, 1 UV, 2 pesos por vertice) + skin de 2 huesos (raiz y un hijo a +1 en y)."""
    pos = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1.75, 0)
    uv = struct.pack("<6f", 0, 0, 1, 0, 0, 1)
    idx = struct.pack("<3H", 0, 1, 2) + b"\0\0"
    jn = struct.pack("<12B", 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0)
    wt = struct.pack("<12f", 1, 0, 0, 0, 0.5, 0.5, 0, 0, 0.25, 0.75, 0, 0)
    partes = [pos, uv, idx, jn, wt]
    off, vistas, binb = 0, [], b""
    for p in partes:
        vistas.append({"buffer": 0, "byteOffset": off, "byteLength": len(p)})
        binb += p
        off += len(p)
    acc = [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1.75, 0]},
           {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"},
           {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}]
    atr = {"POSITION": 0, "TEXCOORD_0": 1}
    nodos = [{"mesh": 0, "name": "cuerpo"}, {"name": "raiz", "children": [2]}, {"name": "hijo", "translation": [0, 1, 0]}]
    j = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, 1]}], "nodes": nodos, "buffers": [{"byteLength": len(binb)}],
         "bufferViews": vistas, "accessors": acc, "meshes": [{"primitives": [{"attributes": atr, "indices": 2}]}]}
    if con_piel:
        acc += [{"bufferView": 3, "componentType": 5121, "count": 3, "type": "VEC4"},
                {"bufferView": 4, "componentType": 5126, "count": 3, "type": "VEC4"}]
        atr.update({"JOINTS_0": 3, "WEIGHTS_0": 4})
        nodos[0]["skin"] = 0
        j["skins"] = [{"joints": [1, 2]}]
    js = json.dumps(j).encode()
    js += b" " * (-len(js) % 4)
    binb += b"\0" * (-len(binb) % 4)
    cuerpo = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(binb), 0x004E4942) + binb
    with open(ruta, "wb") as f:
        f.write(b"glTF" + struct.pack("<II", 2, 12 + len(cuerpo)) + cuerpo)


def png_falso(ancho, alto):
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", ancho, alto, 8, 2, 0, 0, 0))


class AdaptadorGltf(unittest.TestCase):
    def test_con_piel(self):
        d = tempfile.mkdtemp()
        r = os.path.join(d, "m.glb")
        glb_sintetico(r)
        mod = G.analizar(r)
        p = mod["piezas"][0]
        self.assertEqual((p["vertices"], p["triangulos"], p["tipo_indice"], p["max_pesos"], p["uvs"]), (3, 1, "uint16", 2, 1))
        self.assertEqual(p["huesos_usados"], [0, 1])
        self.assertEqual([h["nombre"] for h in mod["esqueleto"]["huesos"]], ["raiz", "hijo"])
        self.assertEqual(mod["esqueleto"]["huesos"][1]["padre"], 0)
        self.assertAlmostEqual(mod["esqueleto"]["huesos"][1]["reposo"][1], 1.0)
        self.assertEqual(mod["bbox"][1][1], 1.75)

    def test_sin_piel(self):
        d = tempfile.mkdtemp()
        r = os.path.join(d, "m.glb")
        glb_sintetico(r, con_piel=False)
        mod = G.analizar(r)
        self.assertIsNone(mod["esqueleto"])
        self.assertFalse(mod["piezas"][0]["con_pesos"])
        self.assertTrue(any("no trae esqueleto" in a for a in mod["avisos"]))

    def test_tam_imagen(self):
        self.assertEqual(C.tam_imagen(png_falso(768, 512)), (768, 512))
        self.assertEqual(C.tam_imagen(b"nada"), (None, None))


class Restricciones(unittest.TestCase):
    def _modelo(self):
        d = tempfile.mkdtemp()
        pieza_falsa(d)
        open(os.path.join(d, "layout.yaml"), "w").write(LAYOUT)
        mod, _ = M.analizar(d, M.leer_layout(os.path.join(d, "layout.yaml")), {"pieza_a": [4, 7, 8, 9]})
        mod["esqueleto"] = {"fuente": "externo: e.json", "huesos": [{"indice": 0, "nombre": "raiz", "padre": -1, "reposo": [0, 0, 0]},
                                                                     {"indice": 1, "nombre": "hijo", "padre": 0, "reposo": [0, 1, 0]}]}
        return mod

    def test_yaml_con_origenes_y_motor(self):
        mod = self._modelo()
        d = RES.construir(mod, RES.inferir_ejes(mod))
        texto = RES.a_yaml(d)
        for y in (yamlmini.yaml(), yamlmini):                              # PyYAML y el lector minimo de respaldo
            r = y.safe_load(texto) if y is not yamlmini else yamlmini.safe_load(texto)
            self.assertEqual(r["pesos"]["max_por_vertice"], {"valor": 3, "origen": "medido"})
            self.assertEqual(r["piezas"]["numero"]["valor"], 1)
            self.assertEqual(r["piezas"]["max_vertices_por_pieza"], {"valor": 65535, "origen": "inferido"})
            self.assertEqual(r["piezas"]["tipo_indice"]["valor"], "uint16")
            self.assertEqual(r["esqueleto"]["origen"], "manual")
            self.assertEqual(r["esqueleto"]["numero_huesos"], 2)
            self.assertEqual(r["esqueleto"]["huesos"][1]["padre"], 0)
            self.assertEqual(r["orientacion"]["eje_arriba"]["origen"], "manual")     # lo declara el layout
            self.assertEqual(r["motor"]["origen"], "manual")
            self.assertIsNone(r["motor"]["max_huesos_por_pieza"])
            self.assertEqual(r["piezas"]["division"]["lista"][0]["paleta_global"], [4, 7, 8, 9])
            self.assertAlmostEqual(r["escala"]["altura"]["valor"], 2.0)
        self.assertIn("# LIMITES DEL MOTOR", texto)
        self.assertNotIn("/Users/", texto)

    def test_ejes_inferidos_por_los_pies(self):
        # figura de pie sobre y: los pies (y ~ 0) sobresalen hacia +x
        pts = [[0, 1.0 + 0.01 * k, 0] for k in range(170)] + [[0.15, 0.0, 0.01 * (k % 3 - 1)] for k in range(10)] + [[0.0, 0.0, 0.01 * (k % 3 - 1)] for k in range(50)]
        mod = {"formato": "gltf", "muestra_puntos": pts, "ejes_declarados": None,
               "bbox": [[0, 0, -0.01], [0.15, 2.7, 0.01]]}
        e = RES.inferir_ejes(mod)
        self.assertEqual(e["arriba"], (1, 1))
        self.assertEqual(e["frente"], (0, 1))
        self.assertEqual(e["origen_frente"], "inferido")
        e2 = RES.inferir_ejes(mod, frente="-z")
        self.assertEqual((e2["frente"], e2["origen_frente"]), ((2, -1), "manual"))

    def test_ejes_mismo_eje(self):
        mod = {"formato": "gltf", "muestra_puntos": [], "ejes_declarados": None, "bbox": [[0, 0, 0], [1, 1, 1]]}
        with self.assertRaises(SystemExit):
            RES.inferir_ejes(mod, arriba="+y", frente="-y")

    def test_canonico(self):
        # arriba +z, frente -y -> canonico y=z, z=-y
        u, f = C.vector_eje("+z"), C.vector_eje("-y")
        x, y, z = C.a_canonico((3.0, -5.0, 7.0), u, f)
        self.assertEqual((y, z), (7.0, 5.0))
        self.assertEqual(x, 3.0 * (u[1] * f[2] - u[2] * f[1]) + 0)         # x_canonico = p . (u x f)

    def test_detectar_formato(self):
        d = tempfile.mkdtemp()
        open(os.path.join(d, "a.vb"), "wb").close()
        self.assertEqual(R.detectar_formato(d), "3dmigoto")
        self.assertEqual(R.detectar_formato("x.GLB"), "gltf")
        self.assertEqual(R.detectar_formato("x.fbx"), "fbx")
        with self.assertRaises(SystemExit):
            R.detectar_formato("x.abc")


if __name__ == "__main__":
    unittest.main()
