"""Remesh a quads (QuadriFlow) + UV + horneado alta->baja, en una sola pasada.

Uso: Blender -b -P postprocesar.py -- alta.glb salida.glb --caras 20000 --tex 2048
Salidas: salida.glb (triangulado, con normal map + ORM), salida_quads.blend (quads reales),
         salida.json (metodo, conteos, tiempos).
"""
import bpy, sys, os, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import remesh_quads as R
import bake as K


def main():
    argv = sys.argv[sys.argv.index('--') + 1:]
    p = argparse.ArgumentParser()
    p.add_argument('alta'); p.add_argument('salida')
    p.add_argument('--caras', type=int, default=20000)
    p.add_argument('--tex', type=int, default=2048)
    p.add_argument('--margen', type=float, default=0.004)
    a = p.parse_args(argv)

    R.reset()
    high = R.load_join(a.alta)
    # la baja parte de una copia de la alta (el remesh la limpia, la escala y la rehace)
    low = high.copy(); low.data = high.data.copy(); bpy.context.scene.collection.objects.link(low)
    low.name = 'low'
    t0 = time.time()
    low, rlog = R.remesh(low, a.caras, 'auto', a.margen)
    t_rem = time.time() - t0
    blog = K.run(high, low, a.salida, a.tex)
    log = dict(entrada=a.alta, salida=a.salida, caras_objetivo=a.caras, tex=a.tex,
               remesh=rlog, bake=blog, seg_remesh_total=round(t_rem, 1), seg_bake_total=blog['seg_bake'])
    json.dump(log, open(os.path.splitext(a.salida)[0] + '.json', 'w'), indent=1)
    print('[post] OK', json.dumps(log)[:600])


if __name__ == '__main__':
    main()
