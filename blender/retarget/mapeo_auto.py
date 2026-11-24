"""mapeo_auto.py -- mapeo semantico automatico de un esqueleto humanoide (por POSICION y TOPOLOGIA, sin nombres de hueso).

Pensado para humanoides riggeados por Puppeteer (huesos genericos "joint0..N") pero sirve para cualquier humanoide con
caderas con >=2 patas hacia abajo + columna hacia arriba + brazos laterales (UAL/Quaternius, Mixamo, Rigify...).

Uso como modulo (dentro de Blender):
    from mapeo_auto import mapear
    m = mapear(armature_obj)          # dict (ver abajo)
Uso por CLI:
    blender -b -P mapeo_auto.py -- <modelo.glb|.blend> <salida.json> [--check]

Resultado de mapear():
    {"hips": nombre, "spine": [n1, n2, ...] (de la cadera hacia arriba; el ultimo es el pecho/chest),
     "neck": nombre|None, "head": nombre|None,
     "L": {"shoulder","upper_arm","forearm","hand","thigh","shin","foot","toe"}, "R": {...},   # valores nombre|None
     "forward": [x, y], "avisos": [...]}
L/R se decide con la direccion de los pies: con adelante f (en XY), izquierda = Z x f (la convencion de Blender: personaje mira a -Y => L en +X).

Algoritmo (todo en espacio mundo, Z arriba):
  1. hips = primer hueso (BFS desde las raices) con >=2 hijos cuyo subarbol baja bastante (piernas) y >=1 que sube (columna).
  2. piernas = los 2 hijos que bajan; cadena = seguir el hijo mas largo -> thigh, shin, foot, toe.
  3. adelante = desplazamiento horizontal de la cadena foot->punta.
  4. columna = hijo que sube; se sigue mientras haya 1 solo hijo; el hueso donde se ramifica (>=2 hijos) es el pecho.
  5. del pecho: hijos laterales = brazos (cadena hasta el hueso con ramas = mano, con los dedos), el hijo alto = cuello->cabeza.
  6. brazo: hand = hueso donde se ramifica (o el ultimo), forearm = su padre, upper_arm = el anterior, shoulder = el anterior (si existe).
Limitaciones: no maneja huesos de torsion (twist) intercalados ni piernas/brazos duplicados; los dedos no se mapean.
"""
import json
import math
import sys

try:
    import bpy
except ImportError:  # permite importar el modulo fuera de Blender para pruebas
    bpy = None


def _info(arm):
    """Posiciones mundo de cada hueso + hijos."""
    M = arm.matrix_world
    h, t, ch = {}, {}, {}
    for b in arm.data.bones:
        h[b.name] = M @ b.head_local
        t[b.name] = M @ b.tail_local
        ch[b.name] = [c.name for c in b.children]
    roots = [b.name for b in arm.data.bones if b.parent is None]
    return h, t, ch, roots


def mapear(arm):
    h, t, ch, roots = _info(arm)
    avisos = []
    # subarbol: min/max z y extension lateral
    memo = {}

    def sub(n):
        if n in memo:
            return memo[n]
        pts = [h[n], t[n]]
        cnt = 1
        for c in ch[n]:
            p, k = sub(c)
            pts += p
            cnt += k
        memo[n] = (pts, cnt)
        return memo[n]

    allp = [p for r in roots for p in sub(r)[0]]
    zmin = min(p.z for p in allp)
    zmax = max(p.z for p in allp)
    H = max(zmax - zmin, 1e-6)

    def bajada(n, c):  # cuanto baja el subarbol de c respecto a la cabeza de n
        return h[n].z - min(p.z for p in sub(c)[0])

    def subida(n, c):
        return max(p.z for p in sub(c)[0]) - h[n].z

    # 1. hips
    hips = None
    cola = list(roots)
    while cola:
        n = cola.pop(0)
        down = [c for c in ch[n] if bajada(n, c) > 0.15 * H]
        up = [c for c in ch[n] if subida(n, c) > 0.15 * H]
        if len(down) >= 2 and up:
            hips = n
            break
        cola += ch[n]
    if hips is None:
        raise RuntimeError("mapeo_auto: no se encontro la cadera (>=2 piernas + columna)")
    down = sorted([c for c in ch[hips] if bajada(hips, c) > 0.15 * H], key=lambda c: -bajada(hips, c))[:2]
    up = sorted([c for c in ch[hips] if subida(hips, c) > 0.15 * H], key=lambda c: -subida(hips, c))
    spine0 = up[0]

    def largo(c):  # hijo "principal": el del subarbol mas grande, luego el mas lejano
        return (sub(c)[1], (max(p.z for p in sub(c)[0]) - min(p.z for p in sub(c)[0])))

    # 2. piernas
    def cadena_pierna(c):
        cad = [c]
        while ch[cad[-1]]:
            cad.append(max(ch[cad[-1]], key=lambda x: largo(x)))
        return cad

    piernas = [cadena_pierna(c) for c in down]
    # 3. adelante
    fx = fy = 0.0
    for cad in piernas:
        a = cad[2] if len(cad) > 2 else cad[-1]
        end = t[cad[-1]]
        fx += end.x - h[a].x
        fy += end.y - h[a].y
    ln = math.hypot(fx, fy)
    if ln < 1e-6:
        avisos.append("no se pudo deducir 'adelante' de los pies; se asume -Y")
        fx, fy, ln = 0.0, -1.0, 1.0
    fx, fy = fx / ln, fy / ln
    lx, ly = -fy, fx  # izquierda = Z x f

    def lado(n, ref=None):
        ref = ref or h[hips]
        d = (h[n].x - ref.x) * lx + (h[n].y - ref.y) * ly
        return "L" if d > 0 else "R"

    out = {"hips": hips, "L": {}, "R": {}, "avisos": avisos, "forward": [round(fx, 3), round(fy, 3)]}
    names = ["thigh", "shin", "foot", "toe"]
    for cad in piernas:
        s = lado(cad[0])
        for i, k in enumerate(names):
            out[s][k] = cad[i] if i < len(cad) else None
    if set(["L", "R"]) != set(lado(c[0]) for c in piernas):
        avisos.append("las dos piernas cayeron del mismo lado; revisar")

    # 4. columna
    spine = [spine0]
    while len(ch[spine[-1]]) == 1:
        spine.append(ch[spine[-1]][0])
    chest = spine[-1]
    out["spine"] = spine
    # 5. del pecho: brazos y cuello
    cuello_c, brazos = None, []
    hijos = ch[chest]
    if not hijos:
        avisos.append("el pecho no tiene hijos (sin cuello ni brazos)")
    for c in hijos:
        pts = sub(c)[0]
        lat = max(abs((p.x - h[chest].x) * lx + (p.y - h[chest].y) * ly) for p in pts)
        if lat > 0.15 * H:
            brazos.append(c)
        elif cuello_c is None or subida(chest, c) > subida(chest, cuello_c):
            cuello_c = c
    out["neck"] = out["head"] = None
    if cuello_c:
        cad = [cuello_c]
        while len(ch[cad[-1]]) == 1:
            cad.append(ch[cad[-1]][0])
        if len(cad) >= 2:
            out["neck"], out["head"] = cad[0], cad[1]
        else:
            out["head"] = cad[0]
    for k in ["shoulder", "upper_arm", "forearm", "hand"]:
        out["L"][k] = out["R"][k] = None
    for c in brazos:
        cad = [c]
        while len(ch[cad[-1]]) == 1:
            cad.append(ch[cad[-1]][0])
        mano = cad[-1]  # ramifica (dedos) o es hoja
        i = len(cad) - 1
        s = lado(mano)
        if i >= 1: out[s]["forearm"] = cad[i - 1]
        if i >= 2: out[s]["upper_arm"] = cad[i - 2]
        if i >= 3: out[s]["shoulder"] = cad[i - 3]
        out[s]["hand"] = mano
    if len(brazos) != 2:
        avisos.append(f"se esperaban 2 brazos y hay {len(brazos)}")
    out["dedos"] = {s: [c for c in ch[out[s]["hand"]]] if out[s].get("hand") else [] for s in "LR"}
    return out


def pares(m):
    """Aplana el mapeo a {clave_semantica: hueso} (la columna en spine_1..spine_n)."""
    f = {"hips": m["hips"], "neck": m["neck"], "head": m["head"]}
    for i, n in enumerate(m["spine"]):
        f[f"spine_{i + 1}"] = n
    for s in "LR":
        for k, v in m[s].items():
            f[f"{k}_{s}"] = v
    return {k: v for k, v in f.items() if v}


if __name__ == "__main__":
    a = sys.argv[sys.argv.index("--") + 1:]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if a[0].endswith(".blend"):
        bpy.ops.wm.open_mainfile(filepath=a[0])
    else:
        bpy.ops.import_scene.gltf(filepath=a[0])
    arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
    m = mapear(arm)
    json.dump(m, open(a[1], "w"), indent=1)
    print("MAPEO", json.dumps(pares(m)))
    if m["avisos"]: print("AVISOS", m["avisos"])
