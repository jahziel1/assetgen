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


class Piernas(unittest.TestCase):
    def test_mapa_piernas(self):
        cen = {15: (0.03, 0, 1.04), 24: (0.1, 0, 0.78), 25: (0.09, 0, 0.35), 27: (-0.11, 0, 0.74), 28: (-0.11, 0, 0.35), 43: (0.23, 0, 0.81)}
        m = PG.mapa_hueso_segmento({}, cen, 0.2, 1.48, z_pierna=0.93)
        self.assertEqual(m[15], "cuerpo")
        self.assertEqual(PG.huesos_de(m, "pierna_I"), [24, 25, 43])
        self.assertEqual(PG.huesos_de(m, "pierna_D"), [27, 28])
        self.assertEqual(PG.mapa_hueso_segmento({}, cen, 0.2, 1.48)[24], "cuerpo")      # sin z_pierna: como antes

    CAD, IZQ, DER = {15: 1.0}, {24: 1.0}, {27: 1.0}

    def test_lejos_de_la_banda_pierna_pura(self):
        w = PG.pesos_entrepierna(self.CAD, self.IZQ, self.DER, 0.12, 0.6, 0.78)
        self.assertEqual(w, {24: 1.0})
        w = PG.pesos_entrepierna(self.CAD, self.IZQ, self.DER, -0.12, 0.6, 0.78)
        self.assertEqual(w, {27: 1.0})

    def test_bajo_la_entrepierna_el_plano_medio_separa(self):
        # bajo la entrepierna (mas de caida_z) cada pierna es pura incluso pegada al plano medio
        self.assertEqual(PG.pesos_entrepierna(self.CAD, self.IZQ, self.DER, 0.01, 0.5, 0.78), {24: 1.0})
        self.assertEqual(PG.pesos_entrepierna(self.CAD, self.IZQ, self.DER, -0.01, 0.5, 0.78), {27: 1.0})

    def test_entrepierna_mezcla_cadera_y_ambas_piernas(self):
        w = PG.pesos_entrepierna(self.CAD, self.IZQ, self.DER, 0.0, 0.80, 0.78)
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertAlmostEqual(w[15], 0.5); self.assertAlmostEqual(w[24], 0.25); self.assertAlmostEqual(w[27], 0.25)

    def test_continuidad_en_el_plano_medio(self):
        a = PG.pesos_entrepierna(self.CAD, self.IZQ, self.DER, 1e-4, 0.8, 0.78)
        b = PG.pesos_entrepierna(self.CAD, self.IZQ, self.DER, -1e-4, 0.8, 0.78)
        for h in (15, 24, 27):
            self.assertAlmostEqual(a[h], b[h], places=2)

    def test_suave_y_mezclar(self):
        self.assertEqual(PG.suave(-1, 0, 1), 0.0); self.assertEqual(PG.suave(2, 0, 1), 1.0); self.assertAlmostEqual(PG.suave(0.5, 0, 1), 0.5)
        r = PG.mezclar_pesos([(0.5, {1: 1.0}), (0.5, {2: 1.0})])
        self.assertAlmostEqual(r[1], 0.5); self.assertAlmostEqual(sum(r.values()), 1.0)


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


class CabezaReferencia(unittest.TestCase):
    def test_referencia_es_el_hueso_de_mas_masa_de_la_cara(self):
        masa = {4: 229.0, 5: 7.0, 9: 47.0, 44: 220.0}
        self.assertEqual(PG.hueso_referencia_cara(masa, [4, 5, 9]), 4)        # la capucha (44) no entra en la eleccion
        self.assertEqual(PG.hueso_referencia_cara([0, 0, 0, 0, 3.0, 9.0], [4, 5]), 5)   # tambien con lista, sin indice fijo

    def test_rigidos_y_descartados(self):
        mov = {7: [(0.1, 0.003), (0.0, 0.0)], 8: [(10.3, 0.3), (8.0, 0.2)], 12: [(1.0, 0.05), (1.0, 0.0)]}
        rig, fuera = PG.huesos_rigidos(4, [4, 7, 8, 12], mov)
        self.assertEqual(rig, {4, 7})
        self.assertEqual(set(fuera), {8, 12})
        self.assertEqual(fuera[8], (10.3, 0.3))


if __name__ == "__main__":
    unittest.main()
