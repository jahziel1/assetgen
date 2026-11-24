"""CLI: blender -b -P animar_mecanico.py -- <entrada.glb> <plantilla: vehiculo|catapulta|rotor> <salida.glb>"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plantillas import PLANTILLAS
a = sys.argv[sys.argv.index('--') + 1:]
if len(a) != 3 or a[1] not in PLANTILLAS:
    print("uso: -- <entrada.glb> <%s> <salida.glb>" % "|".join(PLANTILLAS)); sys.exit(2)
print("RESULTADO", PLANTILLAS[a[1]](a[0], a[2]))
