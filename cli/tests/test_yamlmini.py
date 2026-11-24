"""Pruebas del lector YAML de respaldo. `python3 cli/tests/test_yamlmini.py` (si PyYAML esta, compara contra el)."""
import os, sys, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from assetgen_cli import yamlmini as m

CASOS = [
    "a: [1, 'x y', \"q\"]\nb: {c: 1, d: e}\n# hola\nk: v # c\nlst:\n- a\n- b: 1\n  c: 2\nn:\n  - x\n  -   y: 3\n      z:\n        - 1\nq: 'it''s'\nu: ~\nurl: http://x.com/a#b\n",
    "tope_mensual_usd: 25\nquaternius:\n  ual1: ~/Descargas/Pack [Standard]/UAL1.glb\n",
    "prompt: texto largo que\n  sigue en otra linea\nn: 3\n",
    "- 1\n- 2\n",
]
MALOS = ["a: [1, 2\n", "a: 'x\n", "a:\n  b: 1\n c: 2\n", "a: |\n  x\n", "a: &x 1\n", "a: 1\na: 2\n"]


def main():
    try:
        import yaml
    except ImportError:
        yaml = None
    for t in CASOS:
        r = m.safe_load(t)
        assert r is not None
        if yaml:
            assert r == yaml.safe_load(t), (t, r, yaml.safe_load(t))
    for t in MALOS:
        try:
            m.safe_load(t)
        except m.YamlError:
            continue
        raise AssertionError("debia fallar: %r" % t)
    d = {"size": 1024, "seeds": [1, 2], "prompts": {"a b": {"prompt": 'x: y, "z" #h', "variants": 4, "n": None, "t": True, "e": [], "m": {}}},
         "l": [{"a": 1, "b": {"c": [1, 2]}}, "yes", "3", ""]}
    assert m.safe_load(m.safe_dump(d)) == d
    for f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ejemplos", "*.yaml")):
        m.safe_load(open(f))
    print("yamlmini ok" + (" (comparado con PyYAML)" if yaml else " (sin PyYAML)"))


main()
