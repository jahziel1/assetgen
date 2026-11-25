"""Pruebas de assetgen_cli.visor_geom (matrices de hueso, enlace por pieza, skinning, normalizacion, capsulas) con datos sinteticos."""
import ast, math, os, random, sys, unittest
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".."))
from assetgen_cli import visor_geom as V


def cerca(t, a, b, d=7):
    for x, y in zip(a, b):
        t.assertAlmostEqual(x, y, places=d)


def rotz(a, t):
    c, s = math.cos(a), math.sin(a)
    return [c, s, 0, -s, c, 0, 0, 0, 1] + list(t)       # base por columnas + traslacion


class TestMatrices(unittest.TestCase):
    def test_ida_y_vuelta_12(self):
        m = rotz(0.7, [1, 2, 3])
        cerca(self, V.m12_de_4(V.m4_de_12(m)), m)
        with self.assertRaises(ValueError):
            V.m4_de_12([1, 2, 3])

    def test_inversa(self):
        M = V.m4_de_12([0.3, 0.1, 0.2, -0.4, 0.9, 0.05, 0.2, 0.3, 1.1, 5, -2, 7])
        I = V.mul4(M, V.inv4(M))
        for i in range(4):
            cerca(self, I[i], V.IDENT[i])
        with self.assertRaises(ValueError):
            V.inv4(V.m4_de_12([0] * 9 + [1, 2, 3]))

    def test_enlace_pieza_con_bind_propio(self):
        # vertice original en el bind propio de la pieza; el glb lo trae ya en el bind del cuerpo (enlace E)
        S = rotz(1.1, [0.5, -1, 2])                    # bind original -> pantalla
        E = rotz(-0.4, [3, 0, 1])                      # bind original -> bind del modelo
        v_orig = [0.2, 0.3, 0.4]
        v_glb = V.aplicar(V.m4_de_12(E), v_orig)
        M = V.matrices_hueso([S], [E])[0]
        cerca(self, V.aplicar(M, v_glb), V.aplicar(V.m4_de_12(S), v_orig))

    def test_pantalla_a_godot_es_giro(self):
        M = V.matrices_hueso([rotz(0, [0, 0, 0])], None, V.PANTALLA_A_GODOT)[0]
        cerca(self, V.aplicar(M, [1, 2, 3]), [1, -2, -3])
        det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
        self.assertAlmostEqual(det, 1.0)

    def test_enlace_de_otro_tamano(self):
        with self.assertRaises(ValueError):
            V.matrices_hueso([rotz(0, [0, 0, 0])] * 2, [rotz(0, [0, 0, 0])])


class TestSkinning(unittest.TestCase):
    def test_mezcla_de_dos_huesos(self):
        mats = [V.IDENT, V.m4_de_12([1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 0, 0])]
        out = V.skin_lineal([[1, 1, 1]], [[0, 1, 0, 0]], [[0.25, 0.75, 0.0, 0.0]], mats)
        cerca(self, out[0], [1 + 1.5, 1, 1])

    def test_mejores_pesos(self):
        h, w = V.mejores_pesos({3: 0.5, 7: 0.3, 9: 0.1, 11: 0.05, 12: 0.05})
        self.assertEqual(h[:2], [3, 7])
        self.assertAlmostEqual(sum(w), 1.0)
        h, w = V.mejores_pesos({5: 1.0})
        self.assertEqual((h, w), ([5, 0, 0, 0], [1.0, 0.0, 0.0, 0.0]))
        self.assertEqual(V.mejores_pesos({}), ([0] * 4, [0.0] * 4))


class TestNormalizar(unittest.TestCase):
    def test_pies_en_suelo_centrado_y_altura(self):
        P = [[3, 10, -2], [5, 14, 0], [4, 12, -1]]
        T, esc = V.normalizar_a_suelo(P, 1.75)
        Q = [V.aplicar(T, p) for p in P]
        lo, hi = V.caja(Q)
        self.assertAlmostEqual(lo[1], 0.0)
        self.assertAlmostEqual(hi[1], 1.75)
        self.assertAlmostEqual((lo[0] + hi[0]) / 2, 0.0)
        self.assertAlmostEqual((lo[2] + hi[2]) / 2, 0.0)
        self.assertEqual(V.normalizar_a_suelo([[0, 1, 0], [1, 1, 1]])[1], 1.0)    # sin altura: no divide por cero
        T2, esc2 = V.normalizar_a_suelo(P, escala=esc)
        self.assertEqual(esc, esc2)


class TestCapsulas(unittest.TestCase):
    def test_cilindro_a_lo_largo_de_x(self):
        random.seed(3)
        P = []
        for _ in range(400):
            t, a = random.uniform(0, 1), random.uniform(0, 2 * math.pi)
            P.append([t, 0.05 * math.cos(a), 0.05 * math.sin(a)])
        caps = V.capsulas(P, [4] * len(P))
        self.assertEqual(len(caps), 1)
        c = caps[0]
        self.assertEqual(c["hueso"], 4)
        self.assertAlmostEqual(c["r"], 0.75 * 0.05, places=3)
        self.assertAlmostEqual(abs(c["a"][1]) + abs(c["a"][2]) + abs(c["b"][1]) + abs(c["b"][2]), 0.0, delta=0.03)
        self.assertGreater(abs(c["b"][0] - c["a"][0]), 0.8 - 0.1)

    def test_pocos_vertices_no_dan_capsula(self):
        self.assertEqual(V.capsulas([[0, 0, 0], [1, 0, 0]], [1, 1]), [])

    def test_eje_principal(self):
        c, e, _ = V.eje_principal([[i * 0.1, i * 0.1, 0] for i in range(20)])
        cerca(self, [abs(x) for x in e], [math.sqrt(.5), math.sqrt(.5), 0], 5)


class TestError(unittest.TestCase):
    def test_error_max(self):
        d, i = V.error_max([[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 3, 4]])
        self.assertEqual((d, i), (5.0, 1))
        with self.assertRaises(ValueError):
            V.error_max([[0, 0, 0]], [])


class TestScriptBlender(unittest.TestCase):
    def test_sintaxis_del_exportador(self):
        ruta = os.path.join(AQUI, "..", "..", "blender", "remaster", "exportar_visor.py")
        ast.parse(open(ruta, encoding="utf-8").read())


if __name__ == "__main__":
    unittest.main()
