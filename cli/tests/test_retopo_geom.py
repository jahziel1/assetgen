"""Pruebas de assetgen_cli.retopo_geom (topologia de mallas, reparto de caras, densidad de texel, distorsion UV, composicion de atlas)."""
import math, os, sys, unittest
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".."))
from assetgen_cli import retopo_geom as R

TETRA = [(0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)]            # cerrado, manifold, Euler 2
CUBO = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


class TestTopologia(unittest.TestCase):
    def test_tetraedro_cerrado(self):
        t = R.topologia(TETRA)
        self.assertTrue(t["cerrada_manifold"])
        self.assertEqual((t["vertices"], t["aristas"], t["caras"], t["euler"], t["componentes"]), (4, 6, 4, 2, 1))

    def test_cubo_quads(self):
        t = R.topologia(CUBO)
        self.assertEqual((t["vertices"], t["aristas"], t["euler"]), (8, 12, 2))
        self.assertTrue(t["cerrada_manifold"])

    def test_borde_abierto(self):
        t = R.topologia(TETRA[:3])
        self.assertEqual(t["aristas_abiertas"], 3)
        self.assertFalse(t["cerrada_manifold"])

    def test_no_manifold(self):
        # tres triangulos comparten la arista (0,1)
        t = R.topologia([(0, 1, 2), (0, 1, 3), (0, 1, 4)])
        self.assertEqual(t["aristas_no_manifold"], 1)
        self.assertFalse(t["cerrada_manifold"])

    def test_componentes(self):
        otra = [tuple(v + 10 for v in f) for f in TETRA]
        self.assertEqual(R.componentes(TETRA + otra), [4, 4])
        self.assertEqual(R.topologia(TETRA + otra)["euler"], 4)

    def test_vacia(self):
        self.assertFalse(R.topologia([])["cerrada_manifold"])


class TestReparto(unittest.TestCase):
    def test_suma_exacta(self):
        r = R.reparto([5.4, 0.5, 1.0], 20000)
        self.assertEqual(sum(r), 20000)
        self.assertGreater(r[0], r[2])

    def test_minimo(self):
        r = R.reparto([100, 1], 1000, minimo=50)
        self.assertGreaterEqual(r[1], 50)

    def test_errores(self):
        with self.assertRaises(ValueError):
            R.reparto([], 10)
        with self.assertRaises(ValueError):
            R.reparto([0, 0], 10)


class TestTexel(unittest.TestCase):
    def test_densidad(self):
        # 1 m2 ocupando toda la UV de 4096 px -> 4096 px/m
        self.assertAlmostEqual(R.densidad_texel(1.0, 1.0, 4096), 4096.0)
        self.assertAlmostEqual(R.densidad_texel(4.0, 1.0, 4096), 2048.0)

    def test_factor_escala(self):
        f = R.factor_escala_uv(1.0, 0.25, 4096, 4096)       # actual 2048 px/m -> doble
        self.assertAlmostEqual(f, 2.0)

    def test_errores(self):
        with self.assertRaises(ValueError):
            R.densidad_texel(0, 1, 100)

    def test_hueco_voxel(self):
        self.assertAlmostEqual(R.hueco_minimo_voxel(0.008), 0.004)
        with self.assertRaises(ValueError):
            R.hueco_minimo_voxel(0)


class TestDistorsion(unittest.TestCase):
    def test_isometrico(self):
        p3 = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        self.assertAlmostEqual(R.distorsion_triangulo(p3, [(0, 0), (1, 0), (0, 1)]), 1.0, places=6)
        self.assertAlmostEqual(R.distorsion_triangulo(p3, [(0, 0), (0.3, 0), (0, 0.3)]), 1.0, places=6)    # escala uniforme

    def test_rotado_en_3d(self):
        p3 = [(0, 0, 0), (0, 1, 0), (0, 0, 1)]
        self.assertAlmostEqual(R.distorsion_triangulo(p3, [(0, 0), (1, 0), (0, 1)]), 1.0, places=6)

    def test_estirado(self):
        p3 = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        self.assertAlmostEqual(R.distorsion_triangulo(p3, [(0, 0), (3, 0), (0, 1)]), 3.0, places=6)

    def test_degenerado(self):
        p3 = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        self.assertTrue(math.isinf(R.distorsion_triangulo(p3, [(0, 0), (1, 0), (2, 0)])))
        self.assertTrue(math.isinf(R.distorsion_triangulo([(0, 0, 0)] * 3, [(0, 0), (1, 0), (0, 1)])))


class TestAtlas(unittest.TestCase):
    def test_elegir(self):
        ma, mb = [1, 0, 0, 0, 1], [0, 1, 0, 0, 1]
        aa, ab = [1, 1, 1, 0, 1], [0, 1, 0, 1, 1]
        self.assertEqual(R.elegir_origen(ma, mb, aa, ab), ["a", "b", "a", "b", "a"])
        self.assertEqual(R.elegir_origen([0], [0], [0], [0]), [None])

    def test_solapes(self):
        self.assertEqual(R.solapes([1, 1, 0], [0, 1, 1]), 1)


class TestOrientacion(unittest.TestCase):
    # tetraedro con normales hacia fuera (antihorario visto desde fuera)
    CO = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    FUERA = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]

    def test_volumen_signo(self):
        self.assertAlmostEqual(R.volumen_firmado(self.CO, self.FUERA), 1.0 / 6.0)
        dentro = [f[::-1] for f in self.FUERA]
        self.assertAlmostEqual(R.volumen_firmado(self.CO, dentro), -1.0 / 6.0)

    def test_quad_en_abanico(self):
        cubo = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
        caras = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (3, 0, 4, 7)]
        self.assertAlmostEqual(R.volumen_firmado(cubo, caras), 1.0)

    def test_componentes_y_voltear(self):
        # dos tetraedros separados: el segundo con normales hacia dentro
        co = self.CO + [(p[0] + 5, p[1], p[2]) for p in self.CO]
        caras = list(self.FUERA) + [tuple(x + 4 for x in f[::-1]) for f in self.FUERA]
        self.assertEqual(R.etiquetas_componentes(caras), [0] * 4 + [1] * 4)
        vols = R.orientacion_por_componente(co, caras)
        self.assertGreater(vols[0][0], 0)
        self.assertLess(vols[1][0], 0)
        self.assertEqual(R.caras_a_voltear(co, caras), [4, 5, 6, 7])
        self.assertEqual(R.caras_a_voltear(co, self.FUERA), [])

    def test_componente_abierta_no_se_voltea(self):
        # un triangulo suelto (tela de una piel) tiene "volumen" sin sentido: no se toca aunque salga negativo
        co = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        self.assertEqual(R.caras_a_voltear(co, [(0, 1, 2)]), [])
        self.assertEqual(R.caras_a_voltear(co, [(0, 2, 1)]), [])


class TestPielInterior(unittest.TestCase):
    def test_criterio(self):
        self.assertTrue(R.es_piel_interior(0.05, True))
        self.assertFalse(R.es_piel_interior(0.6, True))      # concavidad abierta (entre los muslos)
        self.assertFalse(R.es_piel_interior(0.05, False))    # sin pared fina detras: es un solido, no una tela doble

    def test_direcciones(self):
        d = R.direcciones_esfera(32)
        self.assertEqual(len(d), 32)
        for v in d:
            self.assertAlmostEqual(sum(x * x for x in v), 1.0)
        self.assertAlmostEqual(sum(v[2] for v in d), 0.0, places=6)      # equilibradas arriba/abajo


if __name__ == "__main__":
    unittest.main()
