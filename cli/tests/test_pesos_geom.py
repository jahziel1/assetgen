import os
import sys
import unittest

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".."))
from assetgen_cli import pesos_geom as PG


class Baricentricas(unittest.TestCase):
    def test_vertices_y_centro(self):
        a, b, c = [0, 0, 0], [1, 0, 0], [0, 1, 0]
        self.assertEqual(PG.coords_baricentricas(a, a, b, c), (1.0, 0.0, 0.0))
        u, v, w = PG.coords_baricentricas([1 / 3, 1 / 3, 0], a, b, c)
        for x in (u, v, w):
            self.assertAlmostEqual(x, 1 / 3)
        u, v, w = PG.coords_baricentricas([0.5, 0.0, 0.0], a, b, c)
        self.assertAlmostEqual(u, 0.5); self.assertAlmostEqual(v, 0.5); self.assertAlmostEqual(w, 0.0)

    def test_degenerado(self):
        self.assertEqual(PG.coords_baricentricas([1, 1, 1], [0, 0, 0], [1, 1, 1], [2, 2, 2]), (1.0, 0.0, 0.0))


class Pesos(unittest.TestCase):
    def test_limitar_y_permitidos(self):
        w = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
        r = PG.limitar_pesos(w, 3)
        self.assertEqual(sorted(r), [1, 2, 3])
        self.assertAlmostEqual(sum(r.values()), 1.0)
        r = PG.limitar_pesos(w, 3, permitidos={3, 4})
        self.assertEqual(sorted(r), [3, 4])
        self.assertAlmostEqual(r[3], 2 / 3)
        self.assertEqual(PG.limitar_pesos(w, 3, permitidos={9}), {})

    def test_interpolar_solo_compatibles(self):
        # un vertice del triangulo trae un hueso de torso (15): no debe pasar al segmento de brazo
        pv = [{20: 1.0}, {20: 0.5, 15: 0.5}, {21: 1.0}]
        r = PG.interpolar_pesos(pv, (0.2, 0.5, 0.3), permitidos={20, 21})
        self.assertNotIn(15, r)
        self.assertAlmostEqual(sum(r.values()), 1.0)
        self.assertAlmostEqual(r[21], 0.3 / (0.2 + 0.25 + 0.3))


class Mapa(unittest.TestCase):
    def test_mapa(self):
        piezas = {"cabeza": [4, 5], "daga": [0], "faldon": [44]}
        cen = {0: (0, 0, 1.5), 4: (0, 0, 1.7), 44: (0, 0, 1.7), 17: (0.02, 0, 1.3), 18: (-0.005, 0, 1.5), 20: (0.17, 0, 1.57),
               22: (-0.16, 0, 1.57), 49: (0.87, 0, 1.55), 59: (-0.87, 0, 1.55), 15: (0.03, 0, 1.0), 34: (0.22, 0, 0.4)}
        m = PG.mapa_hueso_segmento(piezas, cen, 0.2, 1.48)
        self.assertEqual(m[34], "cuerpo")      # cadera lateral: misma x que el brazo pero 1 m mas abajo
        self.assertEqual(m[0], "daga"); self.assertEqual(m[4], "cabeza"); self.assertEqual(m[44], "capucha")
        self.assertEqual(m[17], "cuerpo"); self.assertEqual(m[18], "cuerpo"); self.assertEqual(m[15], "cuerpo")
        self.assertEqual(m[20], "brazo_I"); self.assertEqual(m[49], "brazo_I")
        self.assertEqual(m[22], "brazo_D"); self.assertEqual(m[59], "brazo_D")
        self.assertEqual(PG.huesos_de(m, "brazo_I"), [20, 49])


class Suavizado(unittest.TestCase):
    def test_no_cruza_grupos(self):
        # cadena 0-1-2-3: 0,1 del grupo "a"; 2,3 del grupo "b"
        vec = [[1], [0, 2], [1, 3], [2]]
        W = [{10: 1.0}, {10: 1.0}, {20: 1.0}, {20: 1.0}]
        r = PG.suavizar_pesos(W, vec, ["a", "a", "b", "b"], iteraciones=5)
        self.assertEqual(r, [{10: 1.0}, {10: 1.0}, {20: 1.0}, {20: 1.0}])

    def test_mezcla_y_fijos(self):
        vec = [[1], [0, 2], [1]]
        W = [{1: 1.0}, {1: 0.5, 2: 0.5}, {2: 1.0}]
        r = PG.suavizar_pesos(W, vec, ["g"] * 3, fijos=[0, 2], iteraciones=3)
        self.assertEqual(r[0], {1: 1.0}); self.assertEqual(r[2], {2: 1.0})
        self.assertAlmostEqual(sum(r[1].values()), 1.0)
        self.assertAlmostEqual(r[1][1], 0.5)


class Estiramiento(unittest.TestCase):
    def test_percentiles(self):
        self.assertAlmostEqual(PG.percentil([1, 2, 3, 4, 5], 50), 3)
        self.assertAlmostEqual(PG.percentil([0, 10], 25), 2.5)
        r = PG.resumen_estiramiento([1.0] * 96 + [2.5, 2.1, 0.3, 0.4])
        self.assertEqual((r["mayores"], r["menores"], r["n"]), (2, 2, 100))
        self.assertEqual(r["max"], 2.5)


class Movimiento(unittest.TestCase):
    def test_relativo(self):
        import math
        I = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
        Rz = [c, s, 0, -s, c, 0, 0, 0, 1, 0.1, 0, 0]       # 30 grados sobre z y 0.1 en x
        ang, tr = PG.movimiento_relativo(PG.mul4_12(I, I), PG.mul4_12(Rz, I))
        self.assertAlmostEqual(ang, 30.0, places=4)
        self.assertAlmostEqual(tr, 0.1, places=6)
        ang, tr = PG.movimiento_relativo(PG.mul4_12(Rz, I), PG.mul4_12(Rz, I))
        self.assertAlmostEqual(ang, 0.0, places=4); self.assertAlmostEqual(tr, 0.0, places=6)

    def test_enlace_se_compone_con_la_inversa(self):
        I = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        T = [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 2, 3]
        M = PG.mul4_12(T, T)                                  # pose . enlace^-1 = identidad
        self.assertAlmostEqual(M[0][3], 0.0); self.assertAlmostEqual(M[2][3], 0.0)


if __name__ == "__main__":
    unittest.main()
