"""hoja_cuadros.py -- une imagenes en una hoja de contacto JPG (filas de N imagenes, cada fila a todo el ancho).

    blender -b -P hoja_cuadros.py -- --salida hoja.jpg [--ancho 1000] --fila a.png b.png c.png --fila d.png e.png ..."""
import sys

import bpy
import numpy as np

a = sys.argv[sys.argv.index("--") + 1:]
salida = a[a.index("--salida") + 1]
ancho = int(a[a.index("--ancho") + 1]) if "--ancho" in a else 1000
filas, actual, i = [], None, 0
while i < len(a):
    if a[i] == "--fila":
        actual = []; filas.append(actual)
    elif a[i] in ("--salida", "--ancho"):
        i += 1
    elif actual is not None:
        actual.append(a[i])
    i += 1
bandas = []
for fila in filas:
    w = ancho // len(fila)
    tiles = []
    for ruta in fila:
        im = bpy.data.images.load(ruta)
        h = int(round(im.size[1] * w / im.size[0]))
        im.scale(w, h)
        px = np.empty(w * h * 4, np.float32); im.pixels.foreach_get(px)
        tiles.append(px.reshape(h, w, 4))
    alto = max(t.shape[0] for t in tiles)
    banda = np.zeros((alto, w * len(fila), 4), np.float32); banda[..., 3] = 1
    for k, t in enumerate(tiles):
        banda[:t.shape[0], k * w:(k + 1) * w] = t
    bandas.append(banda)
ancho_real = max(b.shape[1] for b in bandas)
hoja = np.zeros((sum(b.shape[0] for b in bandas), ancho_real, 4), np.float32); hoja[..., 3] = 1
y = 0
for b in reversed(bandas):     # los pixeles de Blender van de abajo a arriba: la primera fila queda arriba
    hoja[y:y + b.shape[0], :b.shape[1]] = b; y += b.shape[0]
out = bpy.data.images.new("hoja", hoja.shape[1], hoja.shape[0])
out.pixels.foreach_set(hoja.ravel())
out.filepath_raw = salida; out.file_format = "JPEG"; out.save()
print("HOJA", hoja.shape[1], hoja.shape[0])
