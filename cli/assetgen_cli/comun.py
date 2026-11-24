"""Utilidades compartidas por los subcomandos: eleccion de GPU y revision de presupuesto/tope mensual."""
from . import runpod as rp
from . import globalcfg
from .presets import *
from .sesion import log


def _candidatas(args, min_gb=0, solo48=False):
    orden = list(MODOS_GPU[args.modo])
    if solo48:
        orden = [g for g in orden if g in GPU_48]
    if args.gpu:  # lista explicita separada por comas (gpuId completo)
        orden = [g.strip() for g in args.gpu.split(",")]
    c = rp.elegir_candidatas(orden, args.tope, args.nube.upper(), min_gb)
    if not c:
        raise SystemExit(f"Ninguna GPU disponible con precio <= ${args.tope}/h en la lista. Prueba --tope o --nube any.")
    return c


def _estimar(c, seg_total):
    precio = c[0][2]
    return precio * seg_total / 3600


def _revisar_presupuesto(args, c, seg_total):
    est = _estimar(c, seg_total)
    log(f"estimacion: ~{seg_total / 60:.0f} min x ${c[0][2]:.2f}/h = ${est:.2f}")
    if args.presupuesto is not None and est > args.presupuesto:
        raise SystemExit(f"Abortado antes de encender: estimacion ${est:.2f} excede el presupuesto ${args.presupuesto:.2f}")
    if not getattr(args, "dry_run", False):
        globalcfg.confirmar_tope(est, getattr(args, "si", False))
    return est
