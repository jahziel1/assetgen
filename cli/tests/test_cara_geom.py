"""Pruebas de assetgen_cli.cara_geom (Umeyama, RBF, poligono, iris) con datos sinteticos. Sin Blender ni datos de ningun juego."""
import colorsys, math, os, random, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from assetgen_cli import cara_geom as G


def rot(ax, ay, az):
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
    ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]
    rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]
    mm = lambda a, b: [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    return mm(rz, mm(ry, rx))


def pts_azar(n, seed=1):
    r = random.Random(seed)
    return [[r.uniform(-1, 1), r.uniform(-1, 1), r.uniform(-1, 1)] for _ in range(n)]


class TestUmeyama(unittest.TestCase):
    def test_recupera_similitud_exacta(self):
        src = pts_azar(12)
        R = rot(0.4, -0.7, 1.1)
        s, t = 2.5, [0.3, -1.0, 4.0]
        dst = [[s * sum(R[a][b] * p[b] for b in range(3)) + t[a] for a in range(3)] for p in src]
        s2, R2, t2 = G.umeyama(src, dst)
        self.assertAlmostEqual(s2, s, places=6)
        for a in range(3):
            self.assertAlmostEqual(t2[a], t[a], places=6)
            for b in range(3):
                self.assertAlmostEqual(R2[a][b], R[a][b], places=6)

    def test_sin_reflexion(self):
        src = pts_azar(10, 2)
        dst = [[-p[0], p[1], p[2]] for p in src]     # espejo: la mejor rotacion propia NO es una reflexion
        _, R, _ = G.umeyama(src, dst)
        det = (R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
               + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]))
        self.assertAlmostEqual(det, 1.0, places=6)

    def test_robusta_rechaza_atipico(self):
        src = pts_azar(20, 3)
        R = rot(0.1, 0.2, 0.3)
        dst = [[1.5 * sum(R[a][b] * p[b] for b in range(3)) + [1, 2, 3][a] for a in range(3)] for p in src]
        dst[5] = [x + 5 for x in dst[5]]
        sim, usados = G.umeyama_robusta(src, dst)
        self.assertNotIn(5, usados)
        self.assertAlmostEqual(sim[0], 1.5, places=5)


class TestRBF(unittest.TestCase):
    def test_interpola_los_puntos(self):
        src = pts_azar(15, 4)
        dst = [[p[0] + 0.1 * math.sin(3 * p[1]), p[1] * 1.2, p[2] - 0.05 * p[0]] for p in src]
        m = G.rbf_ajustar(src, dst)
        for a, b in zip(G.rbf_evaluar(m, src), dst):
            self.assertLess(math.dist(a, b), 1e-6)

    def test_reproduce_una_funcion_afin(self):
        src = pts_azar(10, 5)
        f = lambda p: [2 * p[0] + p[1] + 1, -p[2] + 0.5, p[0] - p[1] + 3 * p[2]]
        m = G.rbf_ajustar(src, [f(p) for p in src])
        nuevos = pts_azar(5, 6)
        for a, b in zip(G.rbf_evaluar(m, nuevos), map(f, nuevos)):
            self.assertLess(math.dist(a, b), 1e-6)

    def test_regularizacion_suaviza(self):
        src = pts_azar(15, 7)
        dst = [[p[0] + (0.2 if i == 3 else 0), p[1], p[2]] for i, p in enumerate(src)]
        exacto = G.rbf_evaluar(G.rbf_ajustar(src, dst, 0.0), [src[3]])[0]
        suave = G.rbf_evaluar(G.rbf_ajustar(src, dst, 0.5), [src[3]])[0]
        self.assertLess(abs(exacto[0] - dst[3][0]), 1e-6)
        self.assertGreater(abs(suave[0] - dst[3][0]), 1e-3)


class TestPoligono(unittest.TestCase):
    def test_distancia_con_signo(self):
        cuadro = [(0, 0), (2, 0), (2, 2), (0, 2)]
        self.assertAlmostEqual(G.distancia_poligono(cuadro, (1, 1)), 1.0)
        self.assertAlmostEqual(G.distancia_poligono(cuadro, (3, 1)), -1.0)
        self.assertAlmostEqual(G.distancia_poligono(cuadro, (2.0, 1)), 0.0)

    def test_suavizado(self):
        self.assertEqual(G.suavizado(-1), 0.0)
        self.assertEqual(G.suavizado(2), 1.0)
        self.assertAlmostEqual(G.suavizado(0.5), 0.5)


class TestIris(unittest.TestCase):
    def test_color_iris_ignora_pupila_y_brillo(self):
        px = [(120, 70, 30)] * 45 + [(5, 5, 5)] * 50 + [(250, 250, 250)] * 5
        self.assertEqual(G.color_iris(px), (120, 70, 30))

    def test_tinte_conserva_poco_saturados(self):
        ref = G.estadistica_iris([(120, 60, 30), (110, 50, 25)])
        self.assertEqual(G.tintar_pixel((230, 225, 220), ref, (40, 90, 160)), (230, 225, 220))

    def test_tinte_lleva_la_media_al_objetivo(self):
        base = [(130, 70, 30), (110, 55, 25), (90, 45, 20)]
        ref = G.estadistica_iris(base)
        objetivo = (40, 90, 160)
        nuevos = [G.tintar_pixel(p, ref, objetivo) for p in base]
        h = [colorsys.rgb_to_hsv(*(c / 255 for c in p))[0] for p in nuevos]
        ht = colorsys.rgb_to_hsv(*(c / 255 for c in objetivo))[0]
        for x in h:
            self.assertAlmostEqual(x, ht, places=1)
        medio_v = sum(colorsys.rgb_to_hsv(*(c / 255 for c in p))[2] for p in nuevos) / 3
        self.assertAlmostEqual(medio_v, colorsys.rgb_to_hsv(*(c / 255 for c in objetivo))[2], delta=0.05)
        # el relieve se conserva: el mas claro sigue siendo el mas claro
        self.assertGreater(max(nuevos[0]), max(nuevos[2]))


if __name__ == "__main__":
    unittest.main()
