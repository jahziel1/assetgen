"""Pruebas de assetgen_cli.pose_geom (rotaciones, angulos de brazo, cruce de pesos, similitud uniforme) con datos sinteticos."""
import math, os, random, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from assetgen_cli import pose_geom as G


def cerca(t, a, b, d=7):
    for x, y in zip(a, b):
        t.assertAlmostEqual(x, y, places=d)


class TestRotaciones(unittest.TestCase):
    def test_eje_angulo(self):
        cerca(self, G.mat_vec(G.rot_eje_angulo([0, 0, 1], math.pi / 2), [1, 0, 0]), [0, 1, 0])

    def test_entre(self):
        a, b = [0.4, -0.15, -0.86], [1, 0, -0.02]
        r = G.rot_entre(a, b)
        cerca(self, G.mat_vec(r, G.unitario(a)), G.unitario(b))

    def test_entre_iguales_y_opuestos(self):
        cerca(self, G.mat_vec(G.rot_entre([1, 2, 3], [1, 2, 3]), [1, 2, 3]), [1, 2, 3])
        cerca(self, G.mat_vec(G.rot_entre([0, 0, 1], [0, 0, -1]), [0, 0, 1]), [0, 0, -1])

    def test_pivote(self):
        r = G.rot_eje_angulo([0, 1, 0], math.pi)
        cerca(self, G.giro_en_pivote(r, [2, 0, 0], [1, 0, 0]), [0, 0, 0])

    def test_angulos(self):
        self.assertAlmostEqual(G.angulo_sobre_horizontal([1, 0, 0]), 0, places=9)
        self.assertAlmostEqual(G.angulo_sobre_horizontal(G.direccion_horizontal(1, -1.3)), -1.3, places=9)
        self.assertAlmostEqual(G.angulo_con_vertical_abajo([0, 0, -1]), 0, places=9)
        self.assertAlmostEqual(G.angulo_con_vertical_abajo(G.direccion_horizontal(-1, -1.3)), 88.7, places=9)
        self.assertEqual(G.direccion_horizontal(-1, 0)[0], -1)

    def test_pesos(self):
        self.assertEqual(G.suave(-1), 0)
        self.assertEqual(G.suave(2), 1)
        self.assertAlmostEqual(G.peso_tramo(0.5, 0, 1), 0.5)
        self.assertEqual(G.peso_tramo(0.2, 0.3, 0.3), 0)


class TestCruce(unittest.TestCase):
    def tubo(self, x_cruce=0.45, ancho=0.1, n=600, semilla=2):
        r = random.Random(semilla)
        pos, pes = [], []
        for _ in range(n):
            x = r.uniform(0.1, 0.9)
            f = G.peso_tramo(x, x_cruce - ancho, x_cruce + ancho)
            pos.append([x, r.uniform(-0.05, 0.05) + 0.02, 1.5 + r.uniform(-0.05, 0.05)])
            pes.append({1: 1 - f, 2: f})
        return pos, pes

    def test_cruce(self):
        pos, pes = self.tubo()
        p, n = G.cruce_pesos(pos, pes, [1], [2], 0, 1, (0.1, 0.9))
        self.assertAlmostEqual(p[0], 0.45, delta=0.02)
        self.assertAlmostEqual(p[1], 0.02, delta=0.02)
        self.assertAlmostEqual(p[2], 1.5, delta=0.02)
        self.assertGreater(n, 10)

    def test_signo_negativo(self):
        pos, pes = self.tubo()
        pos = [[-q[0], q[1], q[2]] for q in pos]
        p, _ = G.cruce_pesos(pos, pes, [1], [2], 0, -1, (0.1, 0.9))
        self.assertAlmostEqual(p[0], -0.45, delta=0.02)

    def test_sin_cruce(self):
        pos, pes = self.tubo()
        with self.assertRaises(ValueError):
            G.cruce_pesos(pos, pes, [1], [3], 0, 1, (0.1, 0.9))


class TestSimilitud(unittest.TestCase):
    def test_recupera(self):
        r = random.Random(5)
        a = [[r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(0, 2)] for _ in range(12)]
        s0, t0 = 1.07, [0.01, -0.03, 0.02]
        b = [G.add(G.mul(p, s0), t0) for p in a]
        s, t = G.ajustar_escala_traslacion(a, b)
        self.assertAlmostEqual(s, s0, places=9)
        cerca(self, t, t0, 9)
        self.assertLess(max(G.errores_alineacion(a, b, s, t)), 1e-9)

    def test_error_con_ruido(self):
        a = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.1]]
        s, t = G.ajustar_escala_traslacion(a, b)
        e = G.errores_alineacion(a, b, s, t)
        self.assertGreater(max(e), 0.01)

    def test_degenerado(self):
        with self.assertRaises(ValueError):
            G.ajustar_escala_traslacion([[1, 1, 1], [1, 1, 1]], [[0, 0, 0], [1, 1, 1]])


if __name__ == "__main__":
    unittest.main()
