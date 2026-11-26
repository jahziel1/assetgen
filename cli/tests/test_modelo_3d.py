import argparse, os, tempfile, unittest

from assetgen_cli import config
from assetgen_cli.main import comando_3d, cmd_3d
from assetgen_cli.pipeline import modo_pod_compartido
from assetgen_cli.presets import (MODELOS_3D, MOTORES_3D, PRESETS_3D, ENV_PIXAL3D, IMG_PIXAL3D, env_3d, seg_asset_3d)


class Resolucion(unittest.TestCase):
    def test_defecto(self):
        self.assertEqual(config.modelo_3d({}, argparse.Namespace()), "trellis2")
        self.assertEqual(config.modelo_3d(None, None), "trellis2")

    def test_yaml_sobre_defecto_y_bandera_sobre_yaml(self):
        cfg = {"modelo_3d": "pixal3d"}
        self.assertEqual(config.modelo_3d(cfg, argparse.Namespace(modelo_3d=None)), "pixal3d")
        self.assertEqual(config.modelo_3d(cfg, argparse.Namespace(modelo_3d="trellis2")), "trellis2")
        self.assertEqual(config.modelo_3d({}, argparse.Namespace(modelo_3d="pixal3d")), "pixal3d")

    def test_bandera_invalida(self):
        with self.assertRaises(SystemExit):
            config.modelo_3d({}, argparse.Namespace(modelo_3d="otro"))

    def _yaml(self, extra):
        d = tempfile.mkdtemp(); r = os.path.join(d, "a.yaml")
        open(r, "w").write("proyecto: x\nassets: []\n" + extra)
        return r

    def test_validacion_yaml(self):
        self.assertEqual(config.cargar(self._yaml("modelo_3d: pixal3d\n"))["modelo_3d"], "pixal3d")
        self.assertIsNone(config.cargar(self._yaml("")).get("modelo_3d"))
        with self.assertRaises(SystemExit):
            config.cargar(self._yaml("modelo_3d: hunyuan\n"))


class Presets(unittest.TestCase):
    def test_parametros_pixal3d(self):
        self.assertEqual(env_3d("boceto", "pixal3d"), {"RESOLUTION": "1024", "TEX_SIZE": "1024", "DECIM": "20000"})
        self.assertEqual(env_3d("medio", "pixal3d"), {"RESOLUTION": "1024", "TEX_SIZE": "2048", "DECIM": "50000"})
        self.assertEqual(env_3d("alto", "pixal3d"), {"RESOLUTION": "1536", "TEX_SIZE": "4096", "DECIM": "150000"})
        self.assertEqual(env_3d("maximo", "pixal3d"),
                         {"RESOLUTION": "1536", "TEX_SIZE": "4096", "DECIM": "1000000", "MAX_GLB_MB": "400"})

    def test_todos_los_presets_tienen_pixal3d(self):
        self.assertEqual(set(ENV_PIXAL3D), set(PRESETS_3D))

    def test_trellis_intacto(self):
        self.assertEqual(env_3d("medio"), PRESETS_3D["medio"]["env"])
        self.assertEqual(seg_asset_3d("medio"), PRESETS_3D["medio"]["seg_asset"])

    def test_estimacion_mas_30(self):
        for p in PRESETS_3D:
            self.assertEqual(seg_asset_3d(p, "pixal3d"), PRESETS_3D[p]["seg_asset"] + 30)

    def test_motores(self):
        self.assertEqual(set(MODELOS_3D), set(MOTORES_3D))
        self.assertEqual(MOTORES_3D["pixal3d"]["imagen"], IMG_PIXAL3D)
        self.assertEqual(IMG_PIXAL3D, "ghcr.io/jahziel1/assetgen-pixal3d:latest")


class PodsYComando(unittest.TestCase):
    def test_pixal3d_nunca_comparte_pod(self):
        self.assertTrue(modo_pod_compartido(True, None, True))
        self.assertTrue(modo_pod_compartido(True, None, True, "trellis2"))
        self.assertFalse(modo_pod_compartido(True, None, True, "pixal3d"))

    def test_comando_pixal3d(self):
        c = comando_3d("pixal3d", "RESOLUTION=1024 TEX_SIZE=2048")
        self.assertEqual(c, "mkdir -p /root/out/pixal3d && cd /root && RESOLUTION=1024 TEX_SIZE=2048 flock /root/gpu.lock "
                            "/root/venv-pixal3d/bin/python /root/run_pixal3d_local.py /root/in /root/out/pixal3d")

    def test_comando_trellis2_igual_que_antes(self):
        c = comando_3d("trellis2", "PIPELINE_TYPE=512")
        self.assertEqual(c, "mkdir -p /root/out/trellis2 && cd /root && PIPELINE_TYPE=512 flock /root/gpu.lock "
                            "/root/venv-trellis2/bin/python /root/run_trellis2_local.py /root/in /root/out/trellis2")

    def test_multivista_con_pixal3d_falla(self):
        d = tempfile.mkdtemp(); open(os.path.join(d, "a.png"), "wb").write(b"x")
        a = argparse.Namespace(preset="medio", config=None, input=d, out=os.path.join(d, "o"), semillas=None, multivista=True,
                               modelo_3d="pixal3d", forzar=False, dry_run=True)
        with self.assertRaises(SystemExit) as cm:
            cmd_3d(a)
        self.assertIn("multivista", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
