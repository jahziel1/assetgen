"""Pruebas de assetgen_cli.piel_color (tono de piel: lectura, CIELAB, delta E, contraste). Sin Blender ni datos de ningun juego."""
import math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from assetgen_cli import piel_color as PC


class TestParse(unittest.TestCase):
    def test_formatos(self):
        self.assertEqual(PC.parse_rgb("172,128,100"), [172.0, 128.0, 100.0])
        self.assertEqual(PC.parse_rgb(" 172, 128 ,100 "), [172.0, 128.0, 100.0])
        self.assertEqual(PC.parse_rgb("#ac8064"), [172.0, 128.0, 100.0])

    def test_invalidos(self):
        for t in ("1,2", "1,2,3,4", "256,0,0", "-1,0,0", "a,b,c"):
            with self.assertRaises(ValueError):
                PC.parse_rgb(t)


class TestLab(unittest.TestCase):
    def test_blanco_negro_gris(self):
        l, a, b = PC.srgb_a_lab([255, 255, 255])
        self.assertAlmostEqual(l, 100, delta=0.05)
        self.assertAlmostEqual(a, 0, delta=0.05)
        self.assertAlmostEqual(b, 0, delta=0.05)
        self.assertAlmostEqual(PC.srgb_a_lab([0, 0, 0])[0], 0, delta=1e-6)
        l, a, b = PC.srgb_a_lab([119, 119, 119])
        self.assertAlmostEqual(l, 50, delta=0.6)
        self.assertAlmostEqual(a, 0, delta=0.05)

    def test_rojo_conocido(self):
        l, a, b = PC.srgb_a_lab([255, 0, 0])
        self.assertAlmostEqual(l, 53.24, delta=0.1)
        self.assertAlmostEqual(a, 80.09, delta=0.2)
        self.assertAlmostEqual(b, 67.20, delta=0.2)


class TestDeltaE(unittest.TestCase):
    def test_identidad_y_simetria(self):
        c1, c2 = [172, 128, 100], [120, 76, 55]
        self.assertEqual(PC.delta_e76(c1, c1), 0)
        self.assertAlmostEqual(PC.delta_e2000(c1, c1), 0, places=9)
        self.assertAlmostEqual(PC.delta_e2000(c1, c2), PC.delta_e2000(c2, c1), places=9)
        self.assertGreater(PC.delta_e2000(c1, c2), 10)

    def test_ciede2000_sharma(self):
        # pares de Sharma et al. (2005), dados en Lab: se comprueba la formula sobre Lab construidos a mano
        def de(l1, a1, b1, l2, a2, b2):
            orig = PC.srgb_a_lab
            tabla = {(1, 1, 1): (l1, a1, b1), (2, 2, 2): (l2, a2, b2)}
            PC.srgb_a_lab = lambda rgb: list(tabla[tuple(int(v) for v in rgb)])
            try:
                return PC.delta_e2000([1, 1, 1], [2, 2, 2])
            finally:
                PC.srgb_a_lab = orig
        self.assertAlmostEqual(de(50.0, 2.6772, -79.7751, 50.0, 0.0, -82.7485), 2.0425, places=3)
        self.assertAlmostEqual(de(50.0, 3.1571, -77.2803, 50.0, 0.0, -82.7485), 2.8615, places=3)
        self.assertAlmostEqual(de(50.0, 2.5, 0.0, 50.0, 0.0, -2.5), 4.3065, places=3)
        self.assertAlmostEqual(de(60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387), 1.2644, places=3)

    def test_orden_de_magnitud_cercano(self):
        self.assertLess(PC.delta_e2000([172, 128, 100], [170, 127, 100]), 1.5)


class TestContraste(unittest.TestCase):
    def test_igual_es_uno(self):
        self.assertAlmostEqual(PC.contraste_amortiguado([0.5, 0.4, 0.3], [0.5, 0.4, 0.3]), 1.0)

    def test_amortigua_la_razon(self):
        c = PC.contraste_amortiguado([0.8, 0.8, 0.8], [0.2, 0.2, 0.2])
        self.assertAlmostEqual(c, 2.0)          # raiz de la razon 4, no 4

    def test_recolorear_pixel(self):
        media, tono = [0.47, 0.30, 0.22], [0.67, 0.50, 0.39]
        self.assertEqual(PC.recolorear_pixel(media, media, tono), tono)        # la media cae en el tono
        self.assertEqual(PC.recolorear_pixel([0.1, 0.1, 0.1], media, tono, peso=0.0), [0.1, 0.1, 0.1])
        px = PC.recolorear_pixel([0.9, 0.9, 0.9], media, tono)
        self.assertTrue(all(0 <= v <= 1 for v in px))
        mas_claro = PC.recolorear_pixel([media[0] + 0.05] + media[1:], media, tono)
        self.assertGreater(mas_claro[0], tono[0])                              # conserva el detalle por encima de la media


if __name__ == "__main__":
    unittest.main()
