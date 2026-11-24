import unittest

from assetgen_cli.main import planear_3d
from assetgen_cli.presets import PRESETS_3D


class Planear3D(unittest.TestCase):
    def test_sin_opciones_igual_que_antes(self):
        b, e, s = planear_3d({"a": "a.png", "a_lado": "al.png"})
        self.assertEqual(set(b), {"a", "a_lado"}); self.assertEqual(e, {}); self.assertEqual(s["a"], ["a"])

    def test_semillas(self):
        _, _, s = planear_3d({"a": "a.png"}, [42, 7])
        self.assertEqual(s["a"], ["a_s42", "a_s7"])

    def test_multivista_separa_vistas_y_agrega_salida(self):
        b, e, s = planear_3d({"a": "a.png", "a_lado": "l.png", "a_espalda": "e.png"}, [1], True)
        self.assertEqual(list(b), ["a"]); self.assertEqual(set(e), {"a_lado", "a_espalda"})
        self.assertEqual(s["a"], ["a_s1", "a_multi"])

    def test_multivista_incompleta_no_genera_multi(self):
        _, e, s = planear_3d({"a": "a.png", "a_lado": "l.png"}, None, True)
        self.assertEqual(s["a"], ["a"]); self.assertEqual(set(e), {"a_lado"})

    def test_vista_sin_base_sigue_siendo_asset(self):
        b, e, _ = planear_3d({"x_lado": "l.png"}, None, True)
        self.assertEqual(list(b), ["x_lado"]); self.assertEqual(e, {})

    def test_preset_maximo(self):
        p = PRESETS_3D["maximo"]
        self.assertEqual(p["env"]["PIPELINE_TYPE"], "1536_cascade"); self.assertTrue(p["solo_3d"])


if __name__ == "__main__":
    unittest.main()
