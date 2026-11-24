"""`assetgen remaster concepto`: plan de trabajos, prompts, preparacion de imagenes y hoja, sin GPU ni RunPod."""
import os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from assetgen_cli import main as M, remaster_concepto as C


def imagen(ruta, tam=(200, 200), fondo=(120, 120, 120), caja=(80, 40, 120, 180)):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", tam, fondo)
    ImageDraw.Draw(im).rectangle(caja, fill=(200, 30, 30))
    im.save(ruta)


class Plan(unittest.TestCase):
    def test_trabajos_por_variante(self):
        jobs = C.plan_trabajos(4, 3, "blue turban")
        self.assertEqual(len(jobs), 12)
        self.assertEqual([j["id"] for j in jobs[:3]], ["v1_frente", "v1_lado", "v1_espalda"])
        self.assertEqual(len({j["seed"] for j in jobs}), 4)                      # semillas distintas por variante
        self.assertEqual(jobs[0]["seed"], jobs[1]["seed"])                       # las 3 vistas de una variante comparten semilla
        fr = jobs[0]["imagenes"]
        self.assertEqual(fr[0], {"ref": "vistas/frente.png"})
        self.assertEqual([i["ref"] for i in fr[1:]], ["refs/ref1.png", "refs/ref2.png", "refs/ref3.png"])
        self.assertEqual(jobs[1]["imagenes"], [{"gen": "v1_frente"}, {"ref": "vistas/lado.png"}])
        self.assertEqual(jobs[5]["imagenes"], [{"gen": "v2_frente"}, {"ref": "vistas/espalda.png"}])

    def test_prompt_frente(self):
        p = C.prompt_frente(3, "blue turban with gold band")
        self.assertIn("A-pose", p); self.assertIn("40 degrees", p); self.assertIn("pure white", p); self.assertIn("not fists", p)
        self.assertIn("pictures 2, 3 and 4", p); self.assertIn("blue turban with gold band.", p)
        self.assertIn("no props", p); self.assertIn("separated fingers", p); self.assertIn("PBR", p)
        self.assertNotIn("style reference", C.prompt_frente(0))
        self.assertIn("reference picture 2", C.prompt_frente(1))

    def test_sin_referencias(self):
        jobs = C.plan_trabajos(2, 0, "young prince")
        self.assertEqual(jobs[0]["imagenes"], [{"ref": "vistas/frente.png"}])      # solo el original, ninguna referencia
        self.assertFalse(any("refs/" in im.get("ref", "") for j in jobs for im in j["imagenes"]))
        self.assertNotIn("reference picture", jobs[0]["prompt"]); self.assertIn("young prince.", jobs[0]["prompt"])
        self.assertIn("pure white", jobs[1]["prompt"]); self.assertIn("not walking", jobs[1]["prompt"])
        self.assertEqual(C.listar_referencias(None), []); self.assertEqual(C.listar_referencias([]), [])

    def test_prompt_vistas(self):
        self.assertIn("side (profile)", C.prompt_vista("lado"))
        self.assertIn("from behind", C.prompt_vista("espalda"))
        self.assertIn("A-pose", C.prompt_vista("lado", "x"))

    def test_semillas(self):
        self.assertEqual(C.semillas_para(2), [11, 222])
        self.assertEqual(C.semillas_para(6)[4:], [44445, 44446])
        with self.assertRaises(SystemExit):
            C.plan_trabajos(0, 1)

    def test_estimacion_rondas(self):
        jobs = C.plan_trabajos(1, 1)
        self.assertGreater(C.segundos_estimados(jobs), C.segundos_estimados(jobs[:1]))   # 2 rondas cuestan una recarga mas


class Archivos(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_encontrar_vistas(self):
        os.makedirs(os.path.join(self.d, "renders"))
        for v in C.VISTAS:
            imagen(os.path.join(self.d, "renders", v + ".png"))
        self.assertEqual(sorted(C.encontrar_vistas(self.d)), sorted(C.VISTAS))
        os.remove(os.path.join(self.d, "renders", "lado.png"))
        with self.assertRaises(SystemExit) as e:
            C.encontrar_vistas(self.d)
        self.assertIn("lado", str(e.exception))

    def test_referencias(self):
        a = os.path.join(self.d, "a.png"); imagen(a)
        open(os.path.join(self.d, "nota.txt"), "w").close()
        self.assertEqual(C.listar_referencias([a]), [a])
        self.assertEqual(C.listar_referencias([self.d]), [a])
        with self.assertRaises(SystemExit):
            C.listar_referencias([os.path.join(self.d, "no.png")])

    def test_preparar_imagen_recorta_y_centra(self):
        from PIL import Image
        o, dst = os.path.join(self.d, "o.png"), os.path.join(self.d, "p.png")
        imagen(o)
        self.assertEqual(C.preparar_imagen(o, dst), (C.ANCHO, C.ALTO))
        im = Image.open(dst)
        self.assertEqual(im.size, (C.ANCHO, C.ALTO))
        self.assertEqual(im.getpixel((2, 2)), (255, 255, 255))                   # fondo uniforme pasa a blanco
        self.assertEqual(im.getpixel((C.ANCHO // 2, C.ALTO // 2)), (200, 30, 30))   # figura al centro
        figura = [x for x in range(C.ANCHO) if im.getpixel((x, C.ALTO // 2)) == (200, 30, 30)]
        self.assertGreater(len(figura), 0.2 * C.ANCHO)                           # la figura se agranda tras recortar

    def test_hoja(self):
        from PIL import Image
        for i in (1, 2):
            for v in C.VISTAS:
                Image.new("RGB", (C.ANCHO, C.ALTO), (i * 50, 9, 9)).save(os.path.join(self.d, f"v{i}_{v}.png"))
        h = C.hoja_concepto(self.d, 2, os.path.join(self.d, "hoja_concepto.jpg"))
        im = Image.open(h)
        self.assertLessEqual(im.width, 1000)
        self.assertGreater(im.height, im.width)


class Cli(unittest.TestCase):
    def test_dry_run_sin_gpu(self):
        d = tempfile.mkdtemp()
        v = os.path.join(d, "v"); os.makedirs(v)
        for n in C.VISTAS:
            imagen(os.path.join(v, n + ".png"))
        ref = os.path.join(d, "r.png"); imagen(ref)
        from unittest import mock
        cand = [("id", "NVIDIA A40", 0.4, 48)]
        with mock.patch.object(C, "_candidatas", return_value=cand), mock.patch.object(C, "Sesion", side_effect=AssertionError("no debe encender")):
            rc = M.main(["remaster", "concepto", "--vistas", v, "--referencias", ref, "--prompt", "x", "--variantes", "2",
                         "--salida", os.path.join(d, "o"), "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertEqual(os.listdir(os.path.join(d, "o")), [])


if __name__ == "__main__":
    unittest.main()
