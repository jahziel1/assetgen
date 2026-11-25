"""Pruebas de assetgen_cli.ajuste_geom con datos sinteticos."""
import math, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from assetgen_cli import ajuste_geom as A


def cerca(t, a, b, d=7):
    for x, y in zip(a, b):
        t.assertAlmostEqual(x, y, places=d)


class TestInterp(unittest.TestCase):
    def test_dentro_y_nudos(self):
        xs, ys = [0, 1, 3], [0, 2, 3]
        self.assertAlmostEqual(A.interp(0.5, xs, ys), 1.0)
        self.assertAlmostEqual(A.interp(2, xs, ys), 2.5)
        self.assertAlmostEqual(A.interp(1, xs, ys), 2.0)

    def test_fuera(self):
        xs, ys = [1, 2], [10, 12]
        self.assertAlmostEqual(A.interp(0, xs, ys), 10)
        self.assertAlmostEqual(A.interp(5, xs, ys), 12)
        self.assertAlmostEqual(A.interp(0, xs, ys, "pendiente1"), 9)
        self.assertAlmostEqual(A.interp(5, xs, ys, "pendiente1"), 15)


class TestTramo(unittest.TestCase):
    def test_extremos_caen_en_destino(self):
        p0, p1, q0, q1 = [0.2, 0.1, 1.3], [0.5, 0.12, 1.31], [0.25, 0.0, 1.5], [0.62, 0.03, 1.55]
        m = A.matriz_tramo(p0, p1, q0, q1)
        cerca(self, A.aplica(m, p0), q0)
        cerca(self, A.aplica(m, p1), q1)

    def test_rigido_conserva_distancias(self):
        p0, p1, q0, q1 = [0, 0, 0], [0.2, 0, 0], [1, 1, 1], [1, 1.3, 1.1]
        m = A.matriz_tramo(p0, p1, q0, q1, estirar=False)
        a, b = A.aplica(m, [0.05, 0.03, 0.02]), A.aplica(m, [0.17, -0.02, 0.01])
        self.assertAlmostEqual(A.largo(a, b), A.largo([0.05, 0.03, 0.02], [0.17, -0.02, 0.01]), places=7)

    def test_estira_solo_axial(self):
        m = A.matriz_tramo([0, 0, 0], [1, 0, 0], [0, 0, 0], [1.5, 0, 0])
        cerca(self, A.aplica(m, [0.4, 0.3, 0.2]), [0.6, 0.3, 0.2])


class TestLargosYErrores(unittest.TestCase):
    def test_cambio(self):
        self.assertAlmostEqual(A.cambio_pct(0.31, 0.37), 19.354838, places=5)
        refs = {"a": [0, 0, 0], "b": [0.3, 0.4, 0]}
        self.assertAlmostEqual(A.largos_tramos(refs, {"t": ("a", "b")})["t"], 0.5)

    def test_errores_mezclan_escalar_y_punto(self):
        e = A.errores_cm({"z": 1.0, "p": [0, 0, 0]}, {"z": 1.012, "p": [0.03, 0.04, 0]})
        self.assertEqual(e, {"p": 5.0, "z": 1.2})


class TestJuego(unittest.TestCase):
    def test_desfase_y_traslado(self):
        joints = {"codo": [0.468, 0.04, 1.54]}
        refs_o = {"fin_manga": [0.62, 0.016, 1.577]}
        pares = {"codo": "fin_manga"}
        d = A.desfases(joints, refs_o, pares)
        cerca(self, d["codo"], [-0.152, 0.024, -0.037])
        # si el nuevo ajustado coincide con el original, el codo vuelve a su sitio
        cerca(self, A.trasladar_juego(refs_o, d, pares)["codo"], joints["codo"])
        # y si queda 1 cm desplazado, el codo se desplaza igual
        r2 = {"fin_manga": [0.63, 0.016, 1.577]}
        cerca(self, A.trasladar_juego(r2, d, pares)["codo"], [0.478, 0.04, 1.54])


if __name__ == "__main__":
    unittest.main()
