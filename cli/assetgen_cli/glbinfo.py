"""Lectura minima de un .glb (sin Blender): clips, triangulos, texturas y tamano."""
import json, os, struct


def glb_info(ruta):
    with open(ruta, "rb") as f:
        magic, ver, total = struct.unpack("<4sII", f.read(12))
        if magic != b"glTF":
            raise ValueError("no es un GLB")
        ln, tipo = struct.unpack("<I4s", f.read(8))
        j = json.loads(f.read(ln))
    acc = j.get("accessors", [])
    tris = 0
    for m in j.get("meshes", []):
        for p in m.get("primitives", []):
            if p.get("mode", 4) == 4 and "indices" in p:
                tris += acc[p["indices"]]["count"] // 3
    return {"clips": [a.get("name", f"anim{i}") for i, a in enumerate(j.get("animations", []))],
            "caras": tris, "texturas": len(j.get("images", [])), "mb": round(os.path.getsize(ruta) / 2**20, 2)}
