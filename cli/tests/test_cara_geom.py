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


class TestUnir(unittest.TestCase):
    def test_apertura_relativa(self):
        self.assertAlmostEqual(G.apertura_relativa((0, 1), (0, -1), (-4, 0), (4, 0)), 0.25)

    def test_abrir_region_quita_rabito(self):
        # malla completa 9x5 de cuadrados; activa: bloque 5x5 + rabito de ancho 1 (el resto de la malla existe pero no esta activa)
        idx = lambda i, j: j * 14 + i
        caras, activas, bloque, rabito = [], set(), set(), set()
        for j in range(9):
            for i in range(12):
                k = len(caras)
                if 3 <= i <= 7 and 2 <= j <= 6:
                    bloque.add(k)
                elif j == 4 and i in (8, 9):
                    rabito.add(k)
                caras.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])
        activas = bloque | rabito
        r = G.abrir_region(caras, activas, 2)
        centro = {j * 12 + i for j in range(3, 6) for i in range(4, 7)}
        self.assertTrue(centro <= r)
        self.assertNotIn(4 * 12 + 9, r)          # la punta del rabito desaparece (el arranque pegado al bloque puede quedar)

    def test_componente_mayor(self):
        caras = [[0, 1, 2], [1, 2, 3], [10, 11, 12]]
        self.assertEqual(G.componente_mayor(caras, {0, 1, 2}), {0, 1})

    def test_lazo_ordenado(self):
        lz = G.lazo_ordenado([(0, 1), (1, 2), (2, 0), (5, 6), (6, 7), (7, 5)])
        self.assertEqual(sorted(map(len, lz)), [3, 3])
        self.assertEqual(set(lz[0]) | set(lz[1]), {0, 1, 2, 5, 6, 7})

    def test_taubin_quita_dentado_sin_encoger(self):
        n = 60
        pts = [[math.cos(2 * math.pi * i / n) * (1 + (0.05 if i % 2 else -0.05)), math.sin(2 * math.pi * i / n) * (1 + (0.05 if i % 2 else -0.05))]
               for i in range(n)]
        s = G.lazo_taubin(pts, 15)
        rad = [math.hypot(*p) for p in s]
        self.assertLess(max(rad) - min(rad), 0.02)
        self.assertAlmostEqual(sum(rad) / n, 1.0, delta=0.02)

    def test_proyectar_a_polilinea(self):
        i, t, q, d = G.proyectar_a_polilinea((0.5, 1.0), [(0, 0), (1, 0), (1, 1), (0, 1)])
        self.assertEqual(i, 2)
        self.assertAlmostEqual(d, 0.0)
        i, t, q, d = G.proyectar_a_polilinea((0.5, -1.0), [(0, 0), (1, 0), (1, 1), (0, 1)])
        self.assertEqual(i, 0)
        self.assertAlmostEqual(t, 0.5)
        self.assertAlmostEqual(d, 1.0)

    def test_emparejar_cadenas(self):
        pares = G.emparejar_cadenas([0.0, 1.0, 2.0, 3.0], [0.0, 1.1, 2.9])
        self.assertEqual(pares[0], (0, 0))
        self.assertEqual(pares[-1], (3, 2))
        self.assertEqual({i for i, _ in pares}, {0, 1, 2, 3})
        self.assertEqual({j for _, j in pares}, {0, 1, 2})

    def test_separar_parpados(self):
        xz = [(math.cos(a), 0.4 * math.sin(a)) for a in [2 * math.pi * k / 12 for k in range(12)]]
        sup, inf = G.separar_parpados(xz)
        self.assertTrue(all(xz[i][1] >= -1e-9 for i in sup))
        self.assertTrue(all(xz[i][1] <= 1e-9 for i in inf))
        self.assertEqual(len(sup) + len(inf), 14)         # las comisuras estan en ambos
        self.assertEqual(sup, sorted(sup, key=lambda i: xz[i][0]))

    def test_desplazar_margenes(self):
        ds, di = G.desplazar_margenes(0.5, -0.5, 0.6, 0.75)
        self.assertAlmostEqual((0.5 + ds) - (-0.5 + di), 0.6)
        self.assertLess(abs(di), abs(ds))

    def test_interpolar_lineal(self):
        self.assertAlmostEqual(G.interpolar_lineal([0, 1, 3], [0, 2, 0], 2), 1.0)
        self.assertEqual(G.interpolar_lineal([0, 1], [5, 7], -3), 5)

    def test_procrustes_con_reflexion(self):
        src = [(0, 0), (1, 0), (0, 2), (1.5, 1)]
        ang, s = 0.7, 2.0
        dst = [(s * (math.cos(ang) * -x - math.sin(ang) * y) + 3, s * (math.sin(ang) * -x + math.cos(ang) * y) - 1) for x, y in src]
        m, t = G.procrustes_2d(src, dst)
        for (x, y), (dx, dy) in zip(src, dst):
            self.assertAlmostEqual(m[0][0] * x + m[0][1] * y + t[0], dx, places=6)
            self.assertAlmostEqual(m[1][0] * x + m[1][1] * y + t[1], dy, places=6)

    def test_distorsion_uv(self):
        p3 = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        tris = [(0, 1, 2), (0, 2, 3)]
        d = G.distorsion_uv(p3, [(0, 0), (1, 0), (1, 1), (0, 1)], tris)
        self.assertAlmostEqual(d["aniso_media"], 1.0)
        self.assertEqual(d["invertidos"], 0)
        d = G.distorsion_uv(p3, [(0, 0), (2, 0), (2, 1), (0, 1)], tris)       # estirado x2 en U
        self.assertAlmostEqual(d["aniso_media"], 2.0)
        d = G.distorsion_uv(p3, [(0, 0), (1, 0), (1, 1), (2, -1)], tris)      # un triangulo invertido
        self.assertEqual(d["invertidos"], 1)



class TestParamArco(unittest.TestCase):
    def test_monotonizar(self):
        r = G.monotonizar_param([0.0, 1.0, 0.8, 0.8, 2.0], 0.01)
        self.assertTrue(all(b > a for a, b in zip(r, r[1:])))
        self.assertAlmostEqual(r[0], 0.0)
        self.assertAlmostEqual(r[-1], 2.0)
        self.assertLess(r[3], 1.1)

    def test_desenrollar(self):
        r = G.desenrollar_param([9.5, 9.9, 0.2, 0.6], 10.0)
        self.assertEqual([round(x, 3) for x in r], [9.5, 9.9, 10.2, 10.6])
        r = G.desenrollar_param([0.5, 0.1, 9.8, 9.4], 10.0)
        self.assertAlmostEqual(r[2], -0.2)


class TestFactorApertura(unittest.TestCase):
    def test_converge_en_pocas_iteraciones(self):
        medida = lambda f: 0.12 + 0.35 * f              # apertura medida con un desfase (como MediaPipe frente a la geometria)
        hist = []
        f = 1.0
        for _ in range(4):
            hist.append((f, medida(f)))
            if abs(hist[-1][1] - 0.30) < 0.003:
                break
            f = G.siguiente_factor(hist, 0.30)
        self.assertLess(abs(hist[-1][1] - 0.30), 0.01)
        self.assertLessEqual(len(hist), 4)

    def test_limites(self):
        self.assertEqual(G.siguiente_factor([(1.0, 0.5)], 0.0001), 0.2)
        self.assertEqual(G.siguiente_factor([(1.0, 0.1)], 5.0), 1.5)


class TestGiroParpado(unittest.TestCase):
    def test_conserva_distancia_y_logra_la_altura(self):
        dy, dz, ddz = -20.0, 8.0, -5.0
        th = G.giro_parpado(dy, dz, ddz)
        phi = math.atan2(dz, -dy) - th
        rho = math.hypot(dy, dz)
        self.assertAlmostEqual(rho * math.sin(phi), dz + ddz, places=9)
        self.assertGreater(th, 0)                         # bajar = giro positivo
        self.assertAlmostEqual(G.giro_parpado(dy, dz, 0.0), 0.0, places=12)

    def test_parpado_inferior_sube(self):
        self.assertLess(G.giro_parpado(-20.0, -8.0, 4.0), 0)


if __name__ == "__main__":
    unittest.main()
