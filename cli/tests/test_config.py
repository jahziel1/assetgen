"""Pruebas de modelo_imagen / pose / tamano en assetgen.yaml. `python3 cli/tests/test_config.py`"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from assetgen_cli import config as c

Y = """proyecto: t
modelo_imagen: qwen
assets:
  - nombre: a
    tipo: humanoide
    modelo_imagen: zimage
    prompt: "x"
  - nombre: b
    tipo: criatura
    prompt: "y"
  - nombre: c
    tipo: humanoide
    pose: libre
    tamano_imagen: 1024
    literal: true
    prompt: "z."
"""


def main():
    d = tempfile.mkdtemp(); r = os.path.join(d, "a.yaml"); open(r, "w").write(Y)
    cfg = c.cargar(r); A = {a["nombre"]: a for a in cfg["assets"]}
    assert [c.modelo_imagen(cfg, A[n]) for n in "abc"] == ["zimage", "qwen", "qwen"]
    assert [c.pose_de(A[n]) for n in "abc"] == ["a", "libre", "libre"]
    assert c.tamano_imagen(cfg, A["a"], "zimage") == 1536 and c.tamano_imagen(cfg, A["b"], "qwen") == 1328 and c.tamano_imagen(cfg, A["c"], "qwen") == 1024
    assert c.prompt_completo(cfg, A["c"]) == "z."
    for malo in ("modelo_imagen: flux\n", "assets:\n  - nombre: a\n    pose: b\n"):
        open(r, "w").write("proyecto: t\n" + malo)
        try:
            c.cargar(r); assert False
        except SystemExit:
            pass
    print("ok test_config")


main()
