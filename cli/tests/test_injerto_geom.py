"""Pruebas de assetgen_cli.injerto_geom (borde de capucha por Fourier, Euler, similitudes) con datos sinteticos."""
import math, os, random, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from assetgen_cli import injerto_geom as IG


class TestResolver(unittest.TestCase):
    def test_sistema(self):
        x = IG.resolver([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], [8, -11, -3])
        for a, b in zip(x, [2, 3, -1]):
            self.assertAlmostEqual(a, b, places=9)

    def test_singular(self):
        with self.assertRaises(ValueError):
            IG.resolver([[1, 2], [2, 4]], [1, 2])


def lazo(y0=0.03, n=800, ruido=0.0, semilla=3):
    """Ribete sintetico: z(theta) con cuna delante (theta = -90 grados) y pico atras."""
    r = random.Random(semilla)
    pts = []
    for _ in range(n):
        t = r.uniform(-math.pi, math.pi)
        z = -0.11 + 0.05 * math.cos(t - math.pi / 2) - 0.03 * math.cos(2 * t) + 0.02 * math.sin(3 * t)
        rad = r.uniform(0.15, 0.24)
        pts.append((rad * math.cos(t), y0 + rad * math.sin(t), z + r.gauss(0, ruido)))
    return pts


class TestBorde(unittest.TestCase):
    def test_recupera_la_altura_sin_ruido(self):
        pts = lazo()
        c, rms, n = IG.ajustar_borde(pts)
        self.assertLess(rms, 1e-3)
        self.assertAlmostEqual(c[0], 0.03, delta=0.02)
        for x, y, z in pts[:50]:
            self.assertAlmostEqual(IG.z_borde(c, x, y), z, delta=2e-3)

    def test_robusto_a_atipicos(self):
        pts = lazo(ruido=0.002)
        r = random.Random(9)
        sucios = pts + [(r.uniform(-.2, .2), r.uniform(-.2, .2), r.uniform(-.45, -.3)) for _ in range(150)]   # ribete del chaleco, mas abajo
        c, rms, n = IG.ajustar_borde(sucios)
        for x, y, z in pts[:80]:
            self.assertAlmostEqual(IG.z_borde(c, x, y), z, delta=0.01)
        self.assertLess(n, len(sucios))

    def test_pocos_puntos(self):
        with self.assertRaises(ValueError):
            IG.ajustar_borde([(0, 0, 0)] * 5)


class TestEuler(unittest.TestCase):
    def test_ida_y_vuelta(self):
        for ang in ([10, -20, 30], [0, 0, 0], [-5.5, 12.25, 170], [89, 0, -45]):
            self.assertEqual(IG.euler_xyz(IG.matriz_euler_xyz(*ang)), [round(a, 3) for a in ang])

    def test_eje_z(self):
        r = IG.matriz_euler_xyz(0, 0, 90)
        p = IG.aplicar_similitud(1, r, [0, 0, 0], [1, 0, 0])
        self.assertAlmostEqual(p[1], 1.0, places=9)
        self.assertAlmostEqual(p[0], 0.0, places=9)


class TestSimilitud(unittest.TestCase):
    def test_inversa(self):
        r = IG.matriz_euler_xyz(12, -7, 40)
        s, t = 0.273, [0.01, -0.02, 0.33]
        si, ri, ti = IG.invertir_similitud(s, r, t)
        for p in ([1, 2, 3], [0, 0, 0], [-0.5, 0.2, 9]):
            q = IG.aplicar_similitud(s, r, t, p)
            back = IG.aplicar_similitud(si, ri, ti, q)
            for a, b in zip(back, p):
                self.assertAlmostEqual(a, b, places=9)


class TestCorteFrontal(unittest.TestCase):
    def test_suavizar(self):
        self.assertEqual(IG.suavizar(-1), 0.0)
        self.assertEqual(IG.suavizar(2), 1.0)
        self.assertAlmostEqual(IG.suavizar(0.5), 0.5)

    def test_peso_frente(self):
        sec, tr = math.radians(40), math.radians(30)
        self.assertEqual(IG.peso_frente(-math.pi / 2, sec, tr), 1.0)                      # justo al frente (-Y)
        self.assertEqual(IG.peso_frente(math.radians(-90 + 40), sec, tr), 1.0)            # borde del sector
        self.assertEqual(IG.peso_frente(math.pi / 2, sec, tr), 0.0)                       # atras
        self.assertEqual(IG.peso_frente(math.radians(-90 + 75), sec, tr), 0.0)            # pasada la transicion
        w = IG.peso_frente(math.radians(-90 + 55), sec, tr)                               # mitad de la transicion
        self.assertAlmostEqual(w, 0.5)

    def test_peso_frente_simetrico_y_con_vuelta(self):
        sec, tr = math.radians(30), math.radians(20)
        for d in (35, 42, 49):
            a = IG.peso_frente(math.radians(-90 + d), sec, tr)
            b = IG.peso_frente(math.radians(-90 - d), sec, tr)
            self.assertAlmostEqual(a, b)
        # 270 grados == -90 grados
        self.assertEqual(IG.peso_frente(math.radians(270), sec, tr), 1.0)

    def test_peso_frente_monotono(self):
        sec, tr = math.radians(30), math.radians(30)
        ws = [IG.peso_frente(math.radians(-90 + d), sec, tr) for d in range(0, 80, 5)]
        self.assertTrue(all(a >= b for a, b in zip(ws, ws[1:])))

    def test_peso_lateral(self):
        self.assertEqual(IG.peso_lateral(0.0, 0.25, 0.08), 1.0)
        self.assertEqual(IG.peso_lateral(0.25, 0.25, 0.08), 0.0)
        self.assertEqual(IG.peso_lateral(-0.3, 0.25, 0.08), 0.0)
        self.assertAlmostEqual(IG.peso_lateral(-0.21, 0.25, 0.08), 0.5)
        self.assertEqual(IG.peso_lateral(0.1, 0.25, 0.0), 1.0)                            # sin fundido: escalon


if __name__ == "__main__":
    unittest.main()
