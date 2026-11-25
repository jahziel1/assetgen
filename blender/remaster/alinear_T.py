"""alinear_T.py -- escala uniforme y traslacion que alinean las articulaciones de un modelo en pose T con las de un original.

    python3 alinear_T.py --nuevo articulaciones_nuevo_T.json --original articulaciones_original.json --salida alineacion.json [--umbral-cm 4]

Python puro (sin Blender ni numpy). Ajusta s y t minimizando la suma de distancias al cuadrado entre s*articulacion_nueva + t y la del
original (sin giro ni deformacion), y reporta el error por articulacion en cm. Las articulaciones con error mayor que el umbral solo se
senalan en `fuera_de_umbral`: no se corrigen. Se informa tambien el ajuste sin cuello y la escala por altura (cabeza incluida) como
comparacion. Imprime 'ALINEACION {resumen}'."""
import json
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "..", "cli"))
from assetgen_cli import pose_geom as G  # noqa: E402


def alinear(nuevo, original, umbral_cm=4.0, excluir=()):
    claves = [k for k in nuevo["articulaciones"] if k in original["articulaciones"] and k not in excluir]
    A = [nuevo["articulaciones"][k] for k in claves]
    B = [original["articulaciones"][k] for k in claves]
    s, t = G.ajustar_escala_traslacion(A, B)
    err = G.errores_alineacion(A, B, s, t)
    por = {k: round(e * 100, 2) for k, e in zip(claves, err)}
    rms = (sum(e * e for e in err) / len(err)) ** 0.5
    return {"escala": round(s, 5), "traslacion_m": [round(x, 5) for x in t], "error_cm": por, "error_rms_cm": round(rms * 100, 2),
            "error_max_cm": round(max(err) * 100, 2), "fuera_de_umbral": sorted(k for k, v in por.items() if v > umbral_cm)}


def main():
    a = sys.argv[1:]

    def arg(n, d=None):
        return a[a.index(n) + 1] if n in a else d
    nuevo = json.load(open(arg("--nuevo")))
    orig = json.load(open(arg("--original")))
    um = float(arg("--umbral-cm", "4"))
    out = alinear(nuevo, orig, um)
    out["umbral_cm"] = um
    out["sin_cuello"] = alinear(nuevo, orig, um, excluir=("cuello",))
    out["metodo"] = "similitud uniforme (escala + traslacion, sin giro) por minimos cuadrados sobre las articulaciones comunes"
    out["altura_nuevo_m"], out["altura_original_m"] = nuevo.get("altura_m"), orig.get("altura_m")
    out["aplicar"] = "p_alineado = escala * p_nuevo + traslacion_m"
    json.dump(out, open(arg("--salida"), "w"), indent=1, ensure_ascii=False)
    print("ALINEACION", json.dumps({k: out[k] for k in ("escala", "traslacion_m", "error_rms_cm", "error_max_cm", "fuera_de_umbral")}))


if __name__ == "__main__":
    main()
