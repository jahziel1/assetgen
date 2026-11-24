"""Pruebas de assetgen_cli.cara_imagen (Lab, transferencia, camara afin, TPS 2D, alta frecuencia, rasterizado UV) con datos sinteticos.
Necesitan numpy: sin el, se saltan (el venv del pipeline de cara lo trae)."""
import os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
try:
    import numpy as np
    from assetgen_cli import cara_imagen as I
except ImportError:      # pragma: no cover
    np = None


@unittest.skipIf(np is None, "numpy no disponible")
class TestColor(unittest.TestCase):
    def test_lab_ida_y_vuelta(self):
        rng = np.random.default_rng(1)
        c = rng.random((200, 3))
        self.assertLess(np.abs(I.lab_a_rgb(I.rgb_a_lab(c)) - c).max(), 1e-6)

    def test_lab_valores_conocidos(self):
        self.assertTrue(np.allclose(I.rgb_a_lab([1, 1, 1]), [100, 0, 0], atol=0.05))
        self.assertTrue(np.allclose(I.rgb_a_lab([0, 0, 0]), [0, 0, 0], atol=1e-6))
        lab = I.rgb_a_lab([1, 0, 0])
        self.assertTrue(np.allclose(lab, [53.24, 80.09, 67.20], atol=0.1))

    def test_transferencia_lleva_media_y_desviacion(self):
        rng = np.random.default_rng(2)
        lab = rng.normal([60, 15, 20], [8, 3, 4], (5000, 3))
        o = I.estadistica_lab(lab)
        d = dict(media=[70, 10, 25], desv=[10, 3.5, 5])
        t = I.transferir_lab(lab, o, d)
        e = I.estadistica_lab(t)
        self.assertTrue(np.allclose(e["media"], d["media"], atol=1e-6))
        self.assertTrue(np.allclose(e["desv"], d["desv"], atol=1e-6))

    def test_transferencia_recorta_la_razon(self):
        o = dict(media=[0, 0, 0], desv=[1, 1, 1])
        d = dict(media=[0, 0, 0], desv=[100, 0.001, 1])
        t = I.transferir_lab([[1.0, 1.0, 1.0]], o, d, 0.5, 1.5)
        self.assertTrue(np.allclose(t, [[1.5, 0.5, 1.0]]))

    def test_estadistica_robusta_ignora_valores_atipicos(self):
        rng = np.random.default_rng(3)
        piel = rng.normal([65, 15, 20], 1.5, (900, 3))
        pelo = rng.normal([15, 2, 3], 1.0, (300, 3))
        e = I.estadistica_lab(np.vstack([piel, pelo]), tol_de=15)
        self.assertLess(np.abs(np.array(e["media"]) - [65, 15, 20]).max(), 0.5)
        sin = I.estadistica_lab(np.vstack([piel, pelo]))
        self.assertGreater(abs(sin["media"][0] - 65), 5)

    def test_mascara(self):
        lab = np.zeros((4, 4, 3)); lab[:2] = 50
        m = np.zeros((4, 4), bool); m[:2] = True
        self.assertAlmostEqual(I.estadistica_lab(lab, m)["media"][0], 50)

    def test_es_piel(self):
        colores = {"piel": [0.80, 0.58, 0.48], "piel_oscura": [0.55, 0.36, 0.27], "pelo": [0.08, 0.06, 0.05], "tela_azul": [0.10, 0.15, 0.35],
                   "dorado": [0.85, 0.65, 0.15], "blanco": [0.95, 0.95, 0.95]}
        r = {k: bool(I.es_piel_lab(I.rgb_a_lab(v))) for k, v in colores.items()}
        self.assertEqual(r, {"piel": True, "piel_oscura": True, "pelo": False, "tela_azul": False, "dorado": False, "blanco": False})

    def test_delta_e(self):
        self.assertAlmostEqual(float(I.delta_e76([0, 0, 0], [3, 4, 0])), 5.0)


@unittest.skipIf(np is None, "numpy no disponible")
class TestGeometria(unittest.TestCase):
    def test_camara_afin_recupera_la_matriz(self):
        rng = np.random.default_rng(4)
        A = np.array([[3.0, 0.2, -0.5, 10], [0.1, -2.5, 0.3, 4]])
        P = rng.normal(0, 1, (60, 3))
        q = I.proyectar_afin(A, P) + rng.normal(0, 1e-9, (60, 2))
        self.assertTrue(np.allclose(I.ajustar_camara_afin(P, q), A, atol=1e-6))

    def test_camara_afin_con_ruido_tiene_residuo_del_tamano_del_ruido(self):
        rng = np.random.default_rng(5)
        A = np.array([[100.0, 0, 2, 50], [0, -100, 1, 80]])
        P = rng.normal(0, 1, (300, 3))
        q = I.proyectar_afin(A, P) + rng.normal(0, 0.5, (300, 2))
        r = np.linalg.norm(I.proyectar_afin(I.ajustar_camara_afin(P, q), P) - q, axis=1)
        self.assertLess(r.mean(), 1.0)

    def test_tps_interpola_exacto(self):
        rng = np.random.default_rng(6)
        s = rng.random((25, 2)) * 100
        d = s + rng.normal(0, 3, s.shape)
        self.assertLess(np.abs(I.tps_evaluar(I.tps_ajustar(s, d), s) - d).max(), 1e-6)

    def test_tps_reproduce_una_afin(self):
        rng = np.random.default_rng(7)
        s = rng.random((20, 2)) * 50
        M = np.array([[1.1, 0.2], [-0.1, 0.9]])
        d = s @ M.T + [5, -3]
        q = rng.random((30, 2)) * 50
        self.assertLess(np.abs(I.tps_evaluar(I.tps_ajustar(s, d), q) - (q @ M.T + [5, -3])).max(), 1e-6)

    def test_tps_regularizada_no_interpola_pero_se_acerca(self):
        rng = np.random.default_rng(8)
        s = rng.random((25, 2)) * 100
        d = s + rng.normal(0, 3, s.shape)
        e0 = np.abs(I.tps_evaluar(I.tps_ajustar(s, d, 0), s) - d).max()
        e1 = np.abs(I.tps_evaluar(I.tps_ajustar(s, d, 50.0), s) - d).max()
        self.assertGreater(e1, e0)
        self.assertLess(e1, 6)

    def test_tps_loo_afin_es_cero(self):
        rng = np.random.default_rng(9)
        s = rng.random((15, 2)) * 10
        d = s * 2 + 1
        self.assertLess(I.tps_loo(s, d).max(), 1e-6)


@unittest.skipIf(np is None, "numpy no disponible")
class TestImagen(unittest.TestCase):
    def test_bilineal_en_centros_y_a_medias(self):
        im = np.arange(16, dtype=float).reshape(4, 4)
        self.assertEqual(float(I.muestrear_bilineal(im, [1], [2])[0]), 9.0)
        self.assertAlmostEqual(float(I.muestrear_bilineal(im, [1.5], [2])[0]), 9.5)
        self.assertAlmostEqual(float(I.muestrear_bilineal(im, [-5], [99])[0]), 12.0)      # recorta al borde

    def test_gauss_conserva_constantes_y_la_media(self):
        im = np.full((40, 40), 0.7)
        self.assertLess(np.abs(I.desenfoque_gauss(im, 3) - 0.7).max(), 1e-12)
        rng = np.random.default_rng(10)
        r = rng.random((50, 50))
        self.assertAlmostEqual(float(I.desenfoque_gauss(r, 2).mean()), float(r.mean()), places=1)

    def test_alta_frecuencia_quita_lo_suave_y_deja_lo_fino(self):
        y, x = np.mgrid[0:128, 0:128]
        suave = 0.5 + 0.3 * np.sin(x / 40.0)
        fino = 0.05 * np.sin(x * 2.0)
        hf = I.alta_frecuencia(suave + fino, 4.0)[20:-20, 20:-20]
        self.assertLess(np.abs(hf).max(), 0.06)
        self.assertGreater(np.abs(hf).max(), 0.04)
        self.assertLess(abs(float(hf.mean())), 5e-3)
        self.assertLess(np.abs(I.alta_frecuencia(suave, 4.0)[20:-20, 20:-20]).max(), 0.01)

    def test_rasterizar_cubre_el_area_y_las_baricentricas_suman_uno(self):
        uv = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]]
        t, b = I.rasterizar_triangulos(uv, [[0, 1, 2]], 64)
        area = (t >= 0).sum() / 64 ** 2
        self.assertAlmostEqual(area, 0.5 * 0.8 * 0.8, delta=0.02)
        self.assertTrue(np.allclose(b[t >= 0].sum(axis=1), 1.0, atol=1e-5))
        j, i = 20, 10                                          # una baricentrica reproduce la posicion UV
        p = (b[j, i, :, None] * np.array(uv)).sum(axis=0)
        self.assertTrue(np.allclose(p * 64, [i + 0.5, j + 0.5], atol=1e-4))

    def test_rasterizar_dos_triangulos_sin_rendija(self):
        uv = [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]
        t, _ = I.rasterizar_triangulos(uv, [[0, 1, 2], [0, 2, 3]], 64)
        self.assertTrue((t[8:56, 8:56] >= 0).all())

    def test_rasterizar_solapados_gana_la_prioridad(self):
        uv = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.1], [0.9, 0.1], [0.1, 0.9]]
        t, _ = I.rasterizar_triangulos(uv, [[0, 1, 2], [3, 4, 5]], 32)
        self.assertEqual(int(t[5, 5]), 1)                      # sin prioridad gana el ultimo
        t, _ = I.rasterizar_triangulos(uv, [[0, 1, 2], [3, 4, 5]], 32, prioridad=[1.0, 0.0])
        self.assertEqual(int(t[5, 5]), 0)                      # con prioridad, el de mayor valor

    def test_rellenar_vacios(self):
        im = np.zeros((8, 8, 3)); im[:, :4] = [1, 0.5, 0.25]
        val = np.zeros((8, 8), bool); val[:, :4] = True
        out = I.rellenar_vacios(im, val, 8)
        self.assertTrue(np.allclose(out[3, 7], [1, 0.5, 0.25]))


if __name__ == "__main__":
    unittest.main()
