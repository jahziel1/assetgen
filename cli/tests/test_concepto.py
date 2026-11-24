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


class Refinado(unittest.TestCase):
    def test_plan_refinar(self):
        jobs = C.plan_trabajos(4, 3, "young prince", refinar=True)
        self.assertEqual(len(jobs), 12)
        self.assertEqual(jobs[0]["imagenes"], [{"ref": "base/frente.png"}, {"ref": "refs/ref1.png"}, {"ref": "refs/ref2.png"}, {"ref": "refs/ref3.png"}])
        self.assertEqual(jobs[1]["imagenes"], [{"gen": "v1_frente"}, {"ref": "base/lado.png"}])
        self.assertEqual(jobs[7]["imagenes"], [{"gen": "v3_frente"}, {"ref": "base/lado.png"}])
        self.assertEqual(len({j["seed"] for j in jobs}), 4)
        self.assertNotEqual(jobs[0]["seed"], C.plan_trabajos(1, 0)[0]["seed"])      # semillas propias del refinado
        self.assertFalse(any("vistas/" in im.get("ref", "") for j in jobs for im in j["imagenes"]))

    def test_prompt_refinar(self):
        p = C.prompt_refinar_frente(3, "young prince")
        for t in ("EXACTLY the same person", "about 45 degrees", "reference only", "do not copy their clothing design", "Picture 2 is a material reference for photoreal skin",
                  "Picture 3", "Picture 4", "blue hood with the gold band", "pure white", "not fists", "young prince."):
            self.assertIn(t, p)
        self.assertNotIn("40 degrees", p)
        self.assertNotIn("Picture 2", C.prompt_refinar_frente(0))
        self.assertIn("Pictures 2 and 3 are material and realism references", C.prompt_refinar_frente(2))
        for v, t in (("lado", "side (profile)"), ("espalda", "from behind")):
            q = C.prompt_refinar_vista(v)
            self.assertIn(t, q); self.assertIn("45 degrees", q); self.assertIn("not walking", q); self.assertIn("earlier version", q)

    def test_base_y_referencia(self):
        d = tempfile.mkdtemp()
        for n in C.VISTAS:
            imagen(os.path.join(d, f"v2_{n}.png"))
        self.assertEqual(sorted(C.encontrar_base(d, 2)), sorted(C.VISTAS))
        with self.assertRaises(SystemExit) as e:
            C.encontrar_base(d, 1)
        self.assertIn("v1_frente", str(e.exception))
        imagen(os.path.join(d, "frente.png"))
        from PIL import Image
        r = os.path.join(d, "r.png"); Image.new("RGB", (1000, 800), (90, 40, 20)).save(r)
        o = os.path.join(d, "o.png")
        self.assertEqual(C.preparar_referencia(r, o), (C.LADO_REF, round(800 * C.LADO_REF / 1000)))   # reduce, conserva proporcion y fondo
        self.assertEqual(Image.open(o).getpixel((1, 1)), (90, 40, 20))

    def test_hoja_con_base(self):
        from PIL import Image
        d = tempfile.mkdtemp()
        base = {}
        for v in C.VISTAS:
            base[v] = os.path.join(d, "b_" + v + ".png"); Image.new("RGB", (C.ANCHO, C.ALTO), (9, 90, 9)).save(base[v])
            Image.new("RGB", (C.ANCHO, C.ALTO), (90, 9, 9)).save(os.path.join(d, f"v1_{v}.png"))
        sin = C.hoja_concepto(d, 1, os.path.join(d, "a.jpg")); con = C.hoja_concepto(d, 1, os.path.join(d, "b.jpg"), base=base)
        self.assertGreater(Image.open(con).height, Image.open(sin).height * 1.8)         # una fila mas
        self.assertLessEqual(Image.open(con).width, 1000)

    def test_dry_run_refinar(self):
        from unittest import mock
        d = tempfile.mkdtemp(); b = os.path.join(d, "b"); os.makedirs(b)
        for n in C.VISTAS:
            imagen(os.path.join(b, f"v1_{n}.png"))
        refs = []
        for n in ("a", "b", "c"):
            refs.append(os.path.join(d, n + ".png")); imagen(refs[-1])
        cand = [("id", "NVIDIA A40", 0.59, 48)]
        with mock.patch.object(C, "_candidatas", return_value=cand), mock.patch.object(C, "Sesion", side_effect=AssertionError("no debe encender")):
            rc = M.main(["remaster", "concepto", "--base", b, "--referencias"] + refs + ["--variantes", "4", "--salida", os.path.join(d, "o"), "--dry-run"])
            self.assertEqual(rc, 0)
            with self.assertRaises(SystemExit):                      # ni --vistas ni --base
                M.main(["remaster", "concepto", "--salida", os.path.join(d, "o2"), "--dry-run"])


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
