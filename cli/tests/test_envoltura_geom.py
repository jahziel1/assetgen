import math
import os
import sys
import unittest

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, ".."))
from assetgen_cli import envoltura_geom as EG


def _area(P, c):
    a, b, d = P[c[0]], P[c[1]], P[c[2]]
    u = [b[i] - a[i] for i in range(3)]; v = [d[i] - a[i] for i in range(3)]
    n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    return 0.5 * math.sqrt(sum(x * x for x in n))


def _normal_z(P, c):
    a, b, d = P[c[0]], P[c[1]], P[c[2]]
    return (b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0])


class Subdivision(unittest.TestCase):
    def setUp(self):
        # dos triangulos que comparten la arista (1, 2); piezas distintas; pesos distintos por vertice
        self.P = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
        self.C = [(0, 1, 2), (1, 3, 2)]
        self.W = [{1: 1.0}, {1: 0.5, 2: 0.5}, {2: 1.0}, {3: 1.0}]
        self.Z = [0, 1]

    def test_sin_aristas_largas_no_cambia(self):
        P, C, W, Z, pad = EG.subdividir_por_longitud(self.P, self.C, self.W, self.Z, 10.0)
        self.assertEqual(len(P), 4); self.assertEqual(C, self.C); self.assertEqual(Z, [0, 1])

    def test_conserva_area_orientacion_y_pieza(self):
        P, C, W, Z, pad = EG.subdividir_por_longitud(self.P, self.C, self.W, self.Z, 0.4)
        self.assertAlmostEqual(sum(_area(P, c) for c in C), 1.0, places=9)
        self.assertTrue(all(_normal_z(P, c) > 0 for c in C))
        self.assertEqual(len(Z), len(C)); self.assertEqual(set(Z), {0, 1})
        # cada cara hija de la pieza k esta dentro del triangulo padre de la pieza k
        for c, z in zip(C, Z):
            cx = sum(P[i][0] for i in c) / 3; cy = sum(P[i][1] for i in c) / 3
            self.assertEqual(z, 0 if cx + cy < 1 else 1)
        ls = [math.dist(P[c[k]], P[c[(k + 1) % 3]]) for c in C for k in range(3)]
        self.assertLessEqual(max(ls), 0.4 * 1.0001 + 1e-9 + 0.5)       # las diagonales (raiz 2 /2^n) tambien bajan de 0.5

    def test_conforme_sin_grietas(self):
        P, C, W, Z, pad = EG.subdividir_por_longitud(self.P, self.C, self.W, self.Z, 0.3)
        # toda arista interior aparece exactamente dos veces con sentidos opuestos (malla cerrada salvo el borde del cuadrado)
        from collections import Counter
        cnt = Counter()
        for c in C:
            for k in range(3):
                cnt[(c[k], c[(k + 1) % 3])] += 1
        self.assertTrue(all(v == 1 for v in cnt.values()))
        interior = [e for e in cnt if (e[1], e[0]) in cnt]
        borde = [e for e in cnt if (e[1], e[0]) not in cnt]
        self.assertTrue(interior)
        for a, b in borde:       # solo el contorno del cuadrado unidad queda abierto: una grieta dejaria bordes dentro
            en_x = all(P[i][0] in (0.0, 1.0) and P[i][0] == P[a][0] for i in (a, b))
            en_y = all(P[i][1] in (0.0, 1.0) and P[i][1] == P[a][1] for i in (a, b))
            self.assertTrue(en_x or en_y, (P[a], P[b]))
        self.assertEqual(len(set(P)), len(P))        # sin vertices duplicados

    def test_pesos_del_punto_medio(self):
        P, C, W, Z, pad = EG.subdividir_por_longitud(self.P, self.C, self.W, self.Z, 0.9)
        for i in range(4, len(P)):
            a, b = pad[i]
            self.assertAlmostEqual(sum(W[i].values()), 1.0)
            esperado = set(W[a]) | set(W[b])
            self.assertTrue(set(W[i]) <= esperado)
            self.assertLessEqual(len(W[i]), 3)
        self.assertTrue(all(p is None for p in pad[:4]))

    def test_maximo_tres_pesos(self):
        W = [{1: 0.5, 2: 0.5}, {3: 0.5, 4: 0.5}, {5: 1.0}, {1: 1.0}]
        P, C, W2, Z, pad = EG.subdividir_por_longitud(self.P, self.C, W, self.Z, 0.9)
        self.assertTrue(all(len(w) <= 3 for w in W2))
        self.assertTrue(all(abs(sum(w.values()) - 1) < 1e-9 for w in W2))

    def test_factor_densifica_solo_donde_se_pide(self):
        P0 = [(0, 0, 0), (2, 0, 0), (0, 2, 0), (2, 2, 0), (4, 0, 0), (4, 2, 0)]
        C0 = [(0, 1, 2), (1, 3, 2), (1, 4, 3), (4, 5, 3)]
        W0 = [{1: 1.0}] * 6; Z0 = [0, 0, 1, 1]
        fac = [1.0, 1.0, 1.0, 1.0, 0.25, 0.25]       # la derecha pide mas densidad
        P, C, W, Z, pad = EG.subdividir_por_longitud(P0, C0, W0, Z0, 2.5, factor=fac)
        izq = sum(1 for z in Z if z == 0); der = sum(1 for z in Z if z == 1)
        self.assertGreater(der, 3 * izq)

    def test_hasta_objetivo(self):
        (P, C, W, Z, pad), lm = EG.subdividir_hasta(self.P, self.C, self.W, self.Z, 200)
        self.assertLess(abs(len(C) - 200), 0.6 * 200)      # la subdivision roja-verde sube a saltos: el objetivo es aproximado
        self.assertGreater(lm, 0)


class Posiciones(unittest.TestCase):
    def test_grupos_costuras(self):
        P = [(0, 0, 0), (1, 0, 0), (0, 0, 0.00001), (1, 0, 0), (0.5, 0, 0)]
        g = EG.agrupar_por_posicion(P, 1e-3)
        self.assertEqual(g[0], g[2]); self.assertEqual(g[1], g[3]); self.assertNotEqual(g[0], g[4])

    def test_factor_por_grupo_toma_el_menor(self):
        grupo = [0, 1, 2, 0]
        caras = [(0, 1, 2), (3, 1, 2)]
        f = EG.factor_por_grupo(grupo, caras, [0, 1], {1: 0.4})
        self.assertEqual(f[0], 0.4); self.assertEqual(f[3], 0.4)     # mismo grupo -> mismo factor
        self.assertEqual(f[1], 0.4)

    def test_vecinos(self):
        v = EG.vecinos_de_grupos([(0, 1, 2)], [0, 1, 1], 2)
        self.assertEqual(v, [{1}, {0}])


class Suavizado(unittest.TestCase):
    def test_cadena_se_suaviza_y_fijos_no_cambian(self):
        vec = [{1}, {0, 2}, {1, 3}, {2, 4}, {3}]
        D = [(0, 0, 0), (0, 0, 0), (1, 0, 0), (0, 0, 0), (0, 0, 0)]
        r = EG.suavizar_desplazamientos(D, vec, iteraciones=3, lam=0.5, fijos={0})
        self.assertEqual(r[0], (0, 0, 0))
        self.assertLess(r[2][0], 1.0); self.assertGreater(r[1][0], 0.0)

    def test_confianza_baja_no_arrastra(self):
        vec = [{1}, {0}]
        r = EG.suavizar_desplazamientos([(1, 0, 0), (0, 0, 0)], vec, iteraciones=1, lam=1.0, confianza=[1.0, 0.001])
        self.assertAlmostEqual(r[1][0], 1.0, places=6)

    def test_limitar_longitud(self):
        d = EG.limitar_longitud((3, 4, 0), 1.0)
        self.assertAlmostEqual(math.hypot(*d), 1.0)
        self.assertEqual(EG.limitar_longitud((0.1, 0, 0), 1.0), (0.1, 0, 0))


class Metricas(unittest.TestCase):
    def test_resumen(self):
        r = EG.resumen_distancias([0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertEqual(r["n"], 5); self.assertAlmostEqual(r["p50"], 2.0); self.assertAlmostEqual(r["max"], 4.0)
        self.assertEqual(EG.resumen_distancias([]), {"n": 0})

    def test_caras_invertidas(self):
        P0 = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        P1 = [(0, 0, 0), (1, 0, 0), (0, -1, 0)]
        self.assertEqual(EG.caras_invertidas(P0, P0, [(0, 1, 2)]), [])
        self.assertEqual(EG.caras_invertidas(P0, P1, [(0, 1, 2)]), [0])

    def test_dmax_por_pieza(self):
        d = EG.dmax_por_pieza(["000056_a", "000075_b"], 0.03, {"000075": 0.1})
        self.assertEqual(d, {"000056_a": 0.03, "000075_b": 0.1})


if __name__ == "__main__":
    unittest.main()
