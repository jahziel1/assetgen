"""Pruebas de la deteccion de pod listo y del corte tras 2 pods fallidos (sin RunPod: se simula `runpod`).
`python3 cli/tests/test_arranque.py`"""
import json, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from assetgen_cli import runpod as rp, sesion


def sesion_falsa(candidatas, espera=3):
    d = tempfile.mkdtemp()
    s = sesion.Sesion("t", "img", candidatas, 20, d, espera_ssh=espera)
    s._lanzar_watchdog_local = lambda: None
    s._parar_watchdog_local = lambda: None
    s._instalar_seguridad = lambda: None
    return s, d


def parchear(**kw):
    ant = {k: getattr(rp, k) for k in kw}
    for k, v in kw.items():
        setattr(rp, k, v)
    return ant


def restaurar(ant):
    for k, v in ant.items():
        setattr(rp, k, v)


def t_listo_por_pod_get():
    """pod get trae ssh.ip/port y el intento de conexion responde, aunque `ssh info` diga 'not ready'."""
    llamadas = {"info": 0}
    ant = parchear(create_pod=lambda *a, **k: "p1", instantanea=lambda pid: ("running", ("1.2.3.4", 2222), 40),
                   ssh_endpoint=lambda pid: llamadas.__setitem__("info", llamadas["info"] + 1), delete_pod=lambda pid: True)
    try:
        s, d = sesion_falsa([("g", "G", 0.5, "SECURE")])
        s._probar_ssh = lambda: True
        s.t_crear = __import__("time").time()
        s.pod_id = "p1"
        assert s._esperar_ssh() is True
        assert (s.host, s.port) == ("1.2.3.4", 2222) and s.hitos["fuente_listo"] == "pod_get" and llamadas["info"] == 0
    finally:
        restaurar(ant)


def t_respaldo_ssh_info():
    ant = parchear(instantanea=lambda pid: ("running", None, 40), ssh_endpoint=lambda pid: ("5.6.7.8", 22))
    try:
        s, d = sesion_falsa([("g", "G", 0.5, "SECURE")])
        s._probar_ssh = lambda: True
        s.t_crear = __import__("time").time(); s.pod_id = "p1"
        assert s._esperar_ssh() is True and s.hitos["fuente_listo"] == "ssh_info"
    finally:
        restaurar(ant)


def t_no_listo_si_no_conecta():
    """running + ssh.ip pero el puerto no acepta conexion: no se da por listo."""
    ant = parchear(instantanea=lambda pid: ("running", ("1.2.3.4", 2222), 5))
    try:
        s, d = sesion_falsa([("g", "G", 0.5, "SECURE")], espera=1)
        s._probar_ssh = lambda: False
        s.t_crear = __import__("time").time(); s.pod_id = "p1"
        assert s._esperar_ssh() is False and "sin SSH" in s._motivo
    finally:
        restaurar(ant)


def t_dos_fallos_seguidos_detienen():
    creados = []
    ant = parchear(create_pod=lambda n, i, g, d, nube: creados.append(g) or f"p{len(creados)}",
                   instantanea=lambda pid: ("initializing", None, 0), delete_pod=lambda pid: True)
    sesion.SONDEO_S = 0.1
    try:
        cand = [(f"g{i}", f"G{i}", 0.5, "SECURE") for i in range(4)]
        s, d = sesion_falsa(cand, espera=1)
        s.t_crear = None
        try:
            s._arrancar(); assert False, "debia fallar"
        except rp.RunpodError as e:
            assert "2 pods seguidos" in str(e), e
        assert len(creados) == 2, creados          # no sigue probando la 3a GPU
        assert [i["resultado"] for i in s.intentos] == ["sin_ssh", "sin_ssh"]
        s.t0 = __import__("time").time(); s._registrar_costo()
        reg = json.loads(open(os.path.join(d, "costos.jsonl")).read().splitlines()[-1])
        assert len(reg["pods"]) == 2 and reg["pods"][0]["estado_final"] == "initializing"
    finally:
        restaurar(ant)


if __name__ == "__main__":
    os.environ["ASSETGEN_CONFIG"] = os.path.join(tempfile.mkdtemp(), "c.yaml")   # el registro global no toca el real
    for n, f in list(globals().items()):
        if n.startswith("t_"):
            f(); print("ok", n)
