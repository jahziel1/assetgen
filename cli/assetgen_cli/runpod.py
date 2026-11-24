"""Envoltura de runpodctl: GPUs, pods, SSH."""
import json, os, re, shutil, subprocess, sys, time

from .presets import PREFIJO_POD

RUNPODCTL = os.environ.get("RUNPODCTL") or shutil.which("runpodctl") or os.path.expanduser("~/.local/bin/runpodctl")
SSH_KEY = os.environ.get("ASSETGEN_SSH_KEY", os.path.expanduser("~/.ssh/runpod_ed25519"))


class RunpodError(RuntimeError):
    pass


def ctl(*args, check=True, timeout=120):
    try:
        p = subprocess.run([RUNPODCTL, *args], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise RunpodError(f"runpodctl {' '.join(args[:2])}: timeout")
    if check and p.returncode != 0:
        raise RunpodError(f"runpodctl {' '.join(args[:2])}: {(p.stderr or p.stdout).strip()[:400]}")
    return p


def _json(text):
    try:
        return json.loads(text)
    except Exception:
        # a veces hay texto antes del JSON
        m = re.search(r"[\[{]", text or "")
        if m:
            try:
                return json.loads(text[m.start():])
            except Exception:
                pass
    return None


def gpu_list():
    return _json(ctl("gpu", "list").stdout) or []


def elegir_candidatas(orden, tope, nube="SECURE", min_gb=0):
    """Devuelve [(gpuId, nombre, precio_h, nube)] en el orden dado, con precio <= tope y disponibles."""
    info = {g["gpuId"]: g for g in gpu_list()}
    salida = []
    for gid in orden:
        g = info.get(gid)
        if not g or g.get("memoryInGb", 0) < min_gb:
            continue
        opciones = []
        if nube in ("SECURE", "ANY") and g.get("secureCloud") and g.get("securePricePerHr"):
            opciones.append(("SECURE", g["securePricePerHr"]))
        if nube in ("COMMUNITY", "ANY") and g.get("communityCloud") and g.get("communityPricePerHr"):
            opciones.append(("COMMUNITY", g["communityPricePerHr"]))
        for n, precio in sorted(opciones, key=lambda x: x[1]):
            if precio <= tope and g.get("available", True):
                salida.append((gid, g.get("displayName", gid), float(precio), n))
                break
    return salida


def list_pods(solo_assetgen=True):
    p = ctl("pod", "list", "--all", check=False)
    pods = _json(p.stdout)
    if p.returncode != 0 or pods is None:
        raise RunpodError(f"pod list fallo: {(p.stderr or p.stdout).strip()[:300]}")
    if isinstance(pods, dict):
        pods = pods.get("pods") or pods.get("items") or []
    if solo_assetgen:
        pods = [x for x in pods if str(x.get("name", "")).startswith(PREFIJO_POD)]
    return pods


def create_pod(name, image, gpu_id, disk_gb, nube="SECURE"):
    p = ctl("pod", "create", "--name", name, "--image", image, "--gpu-id", gpu_id,
            "--container-disk-in-gb", str(disk_gb), "--ports", "22/tcp", "--cloud-type", nube,
            "--min-cuda-version", "12.4", check=False, timeout=180)
    js = _json(p.stdout)
    if p.returncode != 0:
        raise RunpodError((p.stderr or p.stdout).strip()[:400])
    pid = (js or {}).get("id") if isinstance(js, dict) else None
    if not pid:  # recuperar por nombre
        for x in list_pods():
            if x.get("name") == name:
                pid = x.get("id")
    if not pid:
        raise RunpodError(f"no se obtuvo el id del pod: {p.stdout[:200]}")
    return pid


def delete_pod(pod_id, reintentos=4):
    """Borra el pod y verifica. Devuelve True si ya no existe."""
    for i in range(reintentos):
        ctl("pod", "delete", pod_id, check=False)
        time.sleep(2 + 2 * i)
        try:
            if not any(x.get("id") == pod_id for x in list_pods(solo_assetgen=False)):
                return True
        except RunpodError:
            pass
    return False


def ssh_endpoint(pod_id):
    """(host, puerto) directo por IP publica, o None si aun no esta."""
    p = ctl("ssh", "info", pod_id, check=False)
    js = _json(p.stdout)
    if p.returncode != 0 or not isinstance(js, dict) or js.get("error"):
        # respaldo: 'pod get' trae ssh.ip / ssh.port cuando el contenedor ya reporto
        g = _json(ctl("pod", "get", pod_id, check=False).stdout)
        s = (g or {}).get("ssh") if isinstance(g, dict) else None
        if isinstance(s, dict) and s.get("ip") and s.get("port"):
            return str(s["ip"]), int(s["port"])
        return None
    if js.get("ip") and js.get("port"):
        return str(js["ip"]), int(js["port"])
    cmd = js.get("ssh_command") or js.get("command") or ""
    m = re.search(r"@([\w.\-]+)\s+-p\s*(\d+)", cmd) or re.search(r"-p\s*(\d+).*?@([\w.\-]+)", cmd)
    if m:
        a, b = m.groups()
        return (a, int(b)) if a[0].isalpha() or "." in a else (b, int(a))
    return None


def pod_status(pod_id):
    """runtimeStatus del pod (running|initializing|error|...) o 'gone' si ya no existe."""
    p = ctl("pod", "get", pod_id, check=False)
    js = _json(p.stdout)
    if isinstance(js, list):
        js = js[0] if js else None
    if not isinstance(js, dict):
        # puede ser un fallo transitorio de la API: confirmar con la lista
        try:
            if not any(x.get("id") == pod_id for x in list_pods(solo_assetgen=False)):
                return "gone"
        except RunpodError:
            pass
        return "unknown"
    st = (js.get("runtimeStatus") or "").lower()
    if not st or st == "unknown":
        d = (js.get("desiredStatus") or "").lower()
        st = {"running": "initializing", "exited": "exited", "terminated": "terminated"}.get(d, st or "unknown")
    return st


def costo_hora(pod):
    for k in ("costPerHr", "adjustedCostPerHr"):
        if pod.get(k) is not None:
            return float(pod[k])
    return 0.0
