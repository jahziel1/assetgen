"""Pruebas de la integracion de animales Quaternius: campos `animales`/`fuente` y resolucion del pack. `python3 tests/test_animales.py`"""
import os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from assetgen_cli import config as c, globalcfg as g


class Animales(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.d, "pack", "glTF"))
        self.lobo = os.path.join(self.d, "pack", "glTF", "Wolf.gltf")
        open(self.lobo, "w").write("{}")
        self.cfg = os.path.join(self.d, "cfg.yaml")
        os.environ["ASSETGEN_CONFIG"] = self.cfg

    def test_ruta_directa(self):
        open(self.cfg, "w").write("tope_mensual_usd: 5\n")
        self.assertEqual(g.fuente_animal(os.path.join(self.d, "pack"), "Wolf"), self.lobo)

    def test_config_global_ruta_y_mapa(self):
        open(self.cfg, "w").write(f"quaternius:\n  animales: {self.d}/pack\n")
        self.assertEqual(g.fuente_animal(True, "Wolf"), self.lobo)
        self.assertEqual(g.fuente_animal("otro", "Wolf"), self.lobo)
        open(self.cfg, "w").write(f"quaternius:\n  animales:\n    ua: {self.d}/pack\n")
        self.assertEqual(g.fuente_animal("ua", "Wolf"), self.lobo)

    def test_falta(self):
        open(self.cfg, "w").write("tope_mensual_usd: 5\n")
        with self.assertRaises(SystemExit):
            g.fuente_animal(True, "Wolf")
        with self.assertRaises(SystemExit):
            g.fuente_animal(os.path.join(self.d, "pack"), "Zorro")

    def test_yaml_valida_campos(self):
        r = os.path.join(self.d, "a.yaml")
        open(r, "w").write("proyecto: t\nassets:\n  - nombre: l\n    tipo: criatura\n    animales: true\n    fuente: Wolf\n    prompt: x\n")
        self.assertEqual(c.cargar(r)["assets"][0]["fuente"], "Wolf")
        for malo in ("    tipo: criatura\n    animales: true\n", "    tipo: humanoide\n    animales: true\n    fuente: Wolf\n"):
            open(r, "w").write("proyecto: t\nassets:\n  - nombre: l\n" + malo + "    prompt: x\n")
            with self.assertRaises(SystemExit):
                c.cargar(r)


if __name__ == "__main__":
    unittest.main()
