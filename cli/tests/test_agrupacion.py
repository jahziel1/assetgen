"""Pruebas de la agrupacion en pods (imagenes unificadas): `imagenes` = 1 pod para qwen + zimage, `construir` 3d+rig = 1 pod.
Sin RunPod: se reemplaza Sesion por una falsa. `python3 -m unittest discover -s tests` desde cli/."""
import argparse, os, sys, tempfile, time, unittest
from unittest import mock
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from assetgen_cli import main as M, pipeline as P, presets

YAML = """proyecto: t
modelo_imagen: qwen
assets:
  - nombre: barril
    tipo: otro
    prompt: "a wooden barrel"
  - nombre: guerrero
    tipo: humanoide
    modelo_imagen: zimage
    prompt: "a knight"
"""


class SesionFalsa:
    creadas = []

    def __init__(self, tipo, imagen, cands, disco, out, **kw):
        self.tipo, self.imagen, self.disco, self.out = tipo, imagen, disco, out
        self.cmds, self.subidos, self.bajados = [], [], []
        self.t_crear = time.time(); self.hitos = {"ssh_listo": 1.0}; self.gpu = "A6000"; self.precio_h = 0.5; self.costo_total = 0.1
        SesionFalsa.creadas.append(self)

    def __enter__(self): return self
    def __exit__(self, *a): return False
    def ssh(self, cmd, check=True, **kw):
        self.cmds.append(cmd)
        return 1 if cmd.startswith("test -s /root/rigin/") and "falla" in cmd else 0
    def subir(self, carpeta, archivos, destino): self.subidos.append((destino, list(archivos)))
    def subir_archivo(self, local, remoto): self.subidos.append((remoto, os.path.basename(local)))

    def bajar(self, remoto, local):
        os.makedirs(local, exist_ok=True)
        self.bajados.append(remoto)
        if remoto.startswith("/root/out/img_"):
            from PIL import Image
            for n in ("barril", "guerrero"):
                for v in range(1, 3):
                    Image.new("RGB", (64, 64)).save(os.path.join(local, f"{n}_v{v}.png"))
        elif remoto == "/root/out/trellis2":
            open(os.path.join(local, "a.glb"), "wb").write(b"g")
        elif remoto == "/root/out_rig":
            open(os.path.join(local, "a_rigged.glb"), "wb").write(b"g")


def args_comunes(out, **kw):
    d = dict(out=out, modo="normal", tope=1.1, gpu=None, nube="secure", presupuesto=None, max_min=60, timeout_arranque=15, disco=80,
             forzar=False, dry_run=False, si=True, config=None, variantes=2, solo=None, input=None, preset="medio")
    d.update(kw)
    return argparse.Namespace(**d)


def parches():
    SesionFalsa.creadas = []
    cand = [("NVIDIA RTX A6000", "A6000", 0.5, "SECURE")]
    return [mock.patch.object(P, "Sesion", SesionFalsa), mock.patch.object(M, "Sesion", SesionFalsa),
            mock.patch.object(P, "_candidatas", lambda *a, **k: cand), mock.patch.object(M, "_candidatas", lambda *a, **k: cand),
            mock.patch.object(P, "_revisar_presupuesto", lambda *a, **k: 0), mock.patch.object(M, "_revisar_presupuesto", lambda *a, **k: 0)]


class Agrupacion(unittest.TestCase):
    def test_unificadas_un_pod(self):
        g = P.agrupar_imagenes({"qwen": ["a"], "zimage": ["b"]}, unificadas=True)
        self.assertEqual(len(g), 1)
        self.assertEqual(g[0]["imagen"], presets.IMG_IMAGEN)
        self.assertEqual(set(g[0]["modelos"]), {"qwen", "zimage"})

    def test_viejas_un_pod_por_modelo(self):
        g = P.agrupar_imagenes({"qwen": ["a"], "zimage": ["b"]}, unificadas=False)
        self.assertEqual([x["imagen"] for x in g], [presets.IMG_QWEN, presets.IMG_ZIMAGE])
        self.assertEqual(P.venv_imagen("qwen", False), "/root/venv-qwen")
        self.assertEqual(P.venv_imagen("qwen", True), "/root/venv-imagen")

    def test_vacio_y_modelos_sin_assets(self):
        self.assertEqual(P.agrupar_imagenes({}, True), [])
        self.assertEqual(len(P.agrupar_imagenes({"qwen": [], "zimage": ["b"]}, True)[0]["modelos"]), 1)

    def test_estimacion_un_pod_menor_que_dos(self):
        uno = P.segundos_imagenes(P.agrupar_imagenes({"qwen": ["a"], "zimage": ["b"]}, True)[0], 4)
        dos = sum(P.segundos_imagenes(x, 4) for x in P.agrupar_imagenes({"qwen": ["a"], "zimage": ["b"]}, False))
        self.assertLess(uno, dos)

    def test_pod_compartido_3d_rig(self):
        self.assertTrue(P.modo_pod_compartido(True, None, True))
        self.assertFalse(P.modo_pod_compartido(True, {"caras": 1}, True))   # postproceso local entre etapas: 2 pods
        self.assertFalse(P.modo_pod_compartido(True, None, False))          # nada que riggear
        self.assertFalse(P.modo_pod_compartido(False, None, True))          # imagenes viejas

    def test_viejas_por_entorno(self):
        self.assertIn(presets.IMG_TRELLIS, (presets.IMG_3D, presets.IMG_TRELLIS_VIEJA))
        self.assertNotEqual(presets.IMG_TRELLIS_VIEJA, presets.IMG_PUPPETEER_VIEJA)


class FlujoPods(unittest.TestCase):
    def test_imagenes_un_solo_pod_con_los_dos_modelos(self):
        d = tempfile.mkdtemp(); y = os.path.join(d, "a.yaml"); open(y, "w").write(YAML)
        out = os.path.join(d, "o")
        ps = parches() + [mock.patch.object(P, "UNIFICADAS", True)]
        for p in ps: p.start()
        try:
            rc = P.cmd_imagenes(args_comunes(out, config=y))
        finally:
            mock.patch.stopall()
        self.assertEqual(rc, 0)
        self.assertEqual(len(SesionFalsa.creadas), 1)
        s = SesionFalsa.creadas[0]
        self.assertEqual(s.imagen, presets.IMG_IMAGEN)
        self.assertGreaterEqual(s.disco, 180)
        pesos = [c for c in s.cmds if "prepare_weights.sh" in c]
        self.assertEqual(pesos, ["/opt/assetgen/prepare_weights.sh qwen", "/opt/assetgen/prepare_weights.sh zimage"])
        self.assertTrue(all("/root/venv-imagen/bin/python" in c for c in s.cmds if "gen_images_" in c and "flock" in c))
        self.assertTrue(os.path.exists(os.path.join(out, "barril_v1.png")) and os.path.exists(os.path.join(out, "guerrero_v2.png")))

    def test_imagenes_viejas_dos_pods(self):
        d = tempfile.mkdtemp(); y = os.path.join(d, "a.yaml"); open(y, "w").write(YAML)
        ps = parches() + [mock.patch.object(P, "UNIFICADAS", False)]
        for p in ps: p.start()
        try:
            P.cmd_imagenes(args_comunes(os.path.join(d, "o"), config=y))
        finally:
            mock.patch.stopall()
        self.assertEqual([s.imagen for s in SesionFalsa.creadas], [presets.IMG_QWEN, presets.IMG_ZIMAGE])

    def test_3d_rig_un_solo_pod(self):
        d = tempfile.mkdtemp(); png = os.path.join(d, "a.png"); open(png, "wb").write(b"x")
        d3, drig = os.path.join(d, "3d"), os.path.join(d, "rig")
        for p in parches(): p.start()
        try:
            rc = M.cmd_3d_rig(args_comunes(d3), "medio", {"a": png}, ["a"], d3, drig)
        finally:
            mock.patch.stopall()
        self.assertEqual(rc, 0)
        self.assertEqual(len(SesionFalsa.creadas), 1)
        s = SesionFalsa.creadas[0]
        cmds = " | ".join(s.cmds)
        self.assertLess(cmds.index("prepare_weights.sh trellis2"), cmds.index("prepare_weights.sh puppeteer"))
        self.assertIn("/root/rig.sh a /root/rigin/a.glb", cmds)
        self.assertEqual(s.bajados, ["/root/out/trellis2", "/root/out_rig"])
        self.assertTrue(os.path.exists(os.path.join(d3, "a.glb")) and os.path.exists(os.path.join(drig, "a_rigged.glb")))
        import json
        c3 = json.load(open(os.path.join(d3, "stats.json")))["costo_usd"]; cr = json.load(open(os.path.join(drig, "stats.json")))["costo_usd"]
        self.assertAlmostEqual(c3 + cr, 0.1, places=3)

    def test_3d_rig_reanuda_solo_rig(self):
        """El GLB 3D ya existe: no se corre TRELLIS.2, se sube el GLB y solo se riggea (un pod)."""
        d = tempfile.mkdtemp(); d3, drig = os.path.join(d, "3d"), os.path.join(d, "rig"); os.makedirs(d3)
        open(os.path.join(d3, "a.glb"), "wb").write(b"g")
        for p in parches(): p.start()
        try:
            M.cmd_3d_rig(args_comunes(d3), "medio", {}, ["a"], d3, drig)
        finally:
            mock.patch.stopall()
        self.assertEqual(len(SesionFalsa.creadas), 1)
        s = SesionFalsa.creadas[0]
        self.assertNotIn("prepare_weights.sh trellis2", " ".join(s.cmds))
        self.assertIn(("/root/rigin", ["a.glb"]), s.subidos)

    def test_nada_pendiente_no_enciende_pods(self):
        d = tempfile.mkdtemp(); d3, drig = os.path.join(d, "3d"), os.path.join(d, "rig"); os.makedirs(drig)
        open(os.path.join(drig, "a_rigged.glb"), "wb").write(b"g")
        for p in parches(): p.start()
        try:
            M.cmd_3d_rig(args_comunes(d3), "medio", {}, ["a"], d3, drig)
        finally:
            mock.patch.stopall()
        self.assertEqual(SesionFalsa.creadas, [])


if __name__ == "__main__":
    unittest.main()
