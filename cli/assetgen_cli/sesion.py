"""Ciclo de vida de un pod: crear -> SSH -> trabajar -> borrar SIEMPRE. Costos, presupuesto, senales."""
import datetime, json, os, shlex, signal, subprocess, sys, threading, time

from . import runpod as rp
from .presets import PREFIJO_POD


class Abortado(RuntimeError):
    pass


def log(msg):
    print(f"[assetgen {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# script que corre DENTRO del pod: apagado de seguridad via GraphQL (credenciales de /proc/1/environ)
SAFETY_SH = r"""#!/bin/bash
# uso: safety.sh <minutos>   -> borra este pod pase lo que pase tras N minutos
sleep $(( $1 * 60 ))
K=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_API_KEY=//p')
P=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p')
curl -s -m 30 -H 'Content-Type: application/json' "https://api.runpod.io/graphql?api_key=$K" \
  -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$P\\\"}) }\"}" > /root/safety.log 2>&1
"""


class Sesion:
    """Uso:  with Sesion(...) as s: s.ssh('...')   El pod se borra al salir (error, Ctrl-C o fin)."""

    def __init__(self, tipo, imagen, candidatas, disco_gb, out_dir, max_min=60, presupuesto=None,
                 espera_ssh=15 * 60, comando="", presets=None):
        self.tipo, self.imagen, self.candidatas = tipo, imagen, candidatas
        self.disco_gb, self.out_dir, self.max_min = disco_gb, out_dir, max_min
        self.presupuesto, self.espera_ssh = presupuesto, espera_ssh
        self.comando = comando
        self.pod_id = self.gpu = self.host = self.port = None
        self.precio_h = 0.0
        self.t_crear = None
        self.hitos = {}
        self.intentos = []          # pods creados y borrados que no llegaron a SSH (cuentan para el costo)
        self._proc = None
        self._abort = threading.Event()
        self._stop_wd = threading.Event()
        self._wd_local = None
        self._old_handlers = {}

    # ---- senales / vigilancia ----
    def _on_signal(self, signum, frame):
        self._abort.set(); self._matar_proc()
        raise KeyboardInterrupt(f"senal {signum}")

    def _matar_proc(self):
        p = self._proc
        if p and p.poll() is None:
            try: p.kill()
            except Exception: pass

    def _vigilante(self):
        while not self._stop_wd.wait(10):
            if self.presupuesto and self.t_crear and self.pod_id:
                gasto = self.gasto_actual()
                if gasto > self.presupuesto:
                    log(f"PRESUPUESTO excedido (${gasto:.2f} > ${self.presupuesto:.2f}): abortando")
                    self._abort.set(); self._matar_proc()
                    return

    def gasto_actual(self):
        prev = sum(i["costo"] for i in self.intentos)
        if self.t_crear and self.pod_id:
            return prev + (time.time() - self.t_crear) / 3600 * self.precio_h
        return prev

    def __enter__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self.t0 = time.time()
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            self._old_handlers[s] = signal.signal(s, self._on_signal)
        threading.Thread(target=self._vigilante, daemon=True).start()
        try:
            self._arrancar()
        except BaseException:
            self._cerrar()
            raise
        return self

    def __exit__(self, et, ev, tb):
        self._cerrar()
        return False

    # ---- arranque con reintento por GPU ----
    def _arrancar(self):
        errores = []
        for gid, nombre, precio, nube in self.candidatas:
            if self._abort.is_set():
                raise Abortado("abortado")
            name = f"{PREFIJO_POD}{self.tipo}-{time.strftime('%m%d-%H%M%S')}"
            log(f"creando pod {name} en {nombre} (${precio:.2f}/h, {nube})")
            self.gpu, self.precio_h = nombre, precio
            self.t_crear = time.time()
            self.pod_id = None
            try:
                self.pod_id = rp.create_pod(name, self.imagen, gid, self.disco_gb, nube)
            except rp.RunpodError as e:
                errores.append(f"{nombre}: {e}"); log(f"sin pod en {nombre}: {str(e)[:160]}")
                self.t_crear = None
                continue
            log(f"pod {self.pod_id} creado; esperando SSH (max {self.espera_ssh // 60} min)")
            self._lanzar_watchdog_local()
            if self._esperar_ssh():
                self.hitos["ssh_listo"] = round(time.time() - self.t_crear, 1)
                log(f"SSH listo en {self.hitos['ssh_listo']} s ({self.host}:{self.port})")
                self._instalar_seguridad()
                return
            # no hubo SSH: borrar y probar la siguiente GPU
            errores.append(f"{nombre}: sin SSH en {self.espera_ssh}s o pod en error")
            log(f"sin SSH en {nombre} (limite {self.espera_ssh // 60} min o error); borrando pod y reintentando con otra GPU")
            self._borrar_y_registrar_intento()
        raise rp.RunpodError("no se pudo arrancar ningun pod: " + " | ".join(errores))

    def _esperar_ssh(self):
        """Espera SSH hasta `espera_ssh` s. Mientras el pod siga 'initializing' (descarga de imagen) NO se rinde antes del limite;
        solo falla antes si el pod entra en error o desaparece."""
        limite = self.t_crear + self.espera_ssh
        ultimo = None
        while time.time() < limite:
            if self._abort.is_set():
                raise Abortado("abortado")
            st = rp.pod_status(self.pod_id)
            if st != ultimo:
                log(f"pod {self.pod_id}: estado {st} (+{time.time() - self.t_crear:.0f} s)")
                ultimo = st
                if st == "running":
                    self.hitos["contenedor_running"] = round(time.time() - self.t_crear, 1)
            if st in ("error", "terminated", "stopped", "exited", "gone"):
                log(f"pod en estado {st}: se descarta")
                return False
            if st == "running":
                ep = rp.ssh_endpoint(self.pod_id)
                if ep:
                    self.host, self.port = ep
                    try:
                        r = subprocess.run(self._ssh_base() + ["true"], capture_output=True, timeout=40)
                        if r.returncode == 0:
                            return True
                    except subprocess.TimeoutExpired:
                        pass
            time.sleep(10)
        return False

    def _borrar_y_registrar_intento(self):
        self._parar_watchdog_local()
        seg = time.time() - self.t_crear
        ok = rp.delete_pod(self.pod_id)
        self.intentos.append({"pod_id": self.pod_id, "gpu": self.gpu, "precio_h": self.precio_h,
                              "segundos": round(seg, 1), "costo": round(seg / 3600 * self.precio_h, 4), "borrado": ok})
        self.pod_id = None

    # ---- watchdog local (respaldo si este proceso muere con SIGKILL) ----
    def _lanzar_watchdog_local(self):
        seg = (self.max_min + 5) * 60
        cmd = f"sleep {seg}; {shlex.quote(rp.RUNPODCTL)} pod delete {shlex.quote(self.pod_id)} >/dev/null 2>&1"
        self._wd_local = subprocess.Popen(["/bin/sh", "-c", cmd], start_new_session=True,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _parar_watchdog_local(self):
        if self._wd_local:
            try: self._wd_local.kill()
            except Exception: pass
            self._wd_local = None

    def _instalar_seguridad(self):
        r = subprocess.run(self._ssh_base() + [f"cat > /root/safety.sh; chmod +x /root/safety.sh; "
                                               f"setsid nohup /root/safety.sh {int(self.max_min)} >/dev/null 2>&1 </dev/null & sleep 1; echo ok"],
                           input=SAFETY_SH.encode(), capture_output=True, timeout=60)
        if r.returncode == 0:
            log(f"apagado de seguridad en el pod armado ({self.max_min} min) y en local ({self.max_min + 5} min)")
        else:
            log("AVISO: no se pudo armar el apagado dentro del pod; queda el local")

    # ---- SSH ----
    def _ssh_base(self):
        return ["ssh", "-i", rp.SSH_KEY, "-p", str(self.port), "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "IdentitiesOnly=yes",
                "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=9",
                f"root@{self.host}"]

    def ssh(self, cmd, check=True, stdin=None, capture=False):
        """Ejecuta cmd (bash -lc) en el pod mostrando la salida en vivo. Devuelve el codigo (o la salida si capture)."""
        if self._abort.is_set():
            raise Abortado("abortado")
        full = self._ssh_base() + ["bash -lc " + shlex.quote(cmd)]
        if capture:
            self._proc = subprocess.Popen(full, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            out, _ = self._proc.communicate()
        else:
            self._proc = subprocess.Popen(full, stdin=stdin)
            out = None; self._proc.wait()
        rc = self._proc.returncode
        if self._abort.is_set():
            raise Abortado("abortado (presupuesto o senal)")
        if check and rc != 0:
            raise rp.RunpodError(f"fallo en el pod (rc={rc}): {cmd[:120]}")
        return out if capture else rc

    def subir(self, carpeta, archivos, destino):
        """tar sobre SSH: sube `archivos` (nombres relativos a `carpeta`) a `destino` del pod."""
        lista = " ".join(shlex.quote(a) for a in archivos)
        ssh = " ".join(shlex.quote(x) for x in self._ssh_base()) + " " + shlex.quote(f"mkdir -p {destino} && tar -C {destino} -xf -")
        self._sh(f"COPYFILE_DISABLE=1 tar --no-xattrs -C {shlex.quote(carpeta)} -cf - {lista} | {ssh}")

    def subir_archivo(self, local, remoto):
        ssh = " ".join(shlex.quote(x) for x in self._ssh_base()) + " " + shlex.quote(f"cat > {remoto}")
        self._sh(f"cat {shlex.quote(local)} | {ssh}")

    def bajar(self, remoto, local):
        os.makedirs(local, exist_ok=True)
        ssh = " ".join(shlex.quote(x) for x in self._ssh_base()) + " " + shlex.quote(f"tar -C {remoto} -cf - .")
        self._sh(f"{ssh} | tar -C {shlex.quote(local)} -xf -")

    def _sh(self, cmd):
        self._proc = subprocess.Popen(["/bin/bash", "-o", "pipefail", "-c", cmd])
        self._proc.wait()
        if self._abort.is_set():
            raise Abortado("abortado")
        if self._proc.returncode != 0:
            raise rp.RunpodError(f"transferencia fallo (rc={self._proc.returncode})")

    # ---- cierre ----
    def _cerrar(self):
        self._stop_wd.set()
        # durante la limpieza se ignoran Ctrl-C adicionales
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, signal.SIG_IGN)
        try:
            self._matar_proc()
            self._parar_watchdog_local()
            borrado = True
            if self.pod_id:
                seg = time.time() - self.t_crear
                borrado = rp.delete_pod(self.pod_id)
                self.hitos["segundos_pod"] = round(seg, 1)
                self.intentos.append({"pod_id": self.pod_id, "gpu": self.gpu, "precio_h": self.precio_h,
                                      "segundos": round(seg, 1), "costo": round(seg / 3600 * self.precio_h, 4), "borrado": borrado,
                                      "hitos": self.hitos})
                log(f"pod {self.pod_id} " + ("BORRADO" if borrado else "NO SE PUDO BORRAR: ejecuta `assetgen limpiar`"))
                self.pod_id = None
            self._registrar_costo()
        finally:
            for s, h in self._old_handlers.items():
                signal.signal(s, h)

    def _registrar_costo(self):
        total = round(sum(i["costo"] for i in self.intentos), 4)
        self.costo_total = total
        reg = {"fecha": datetime.datetime.now().isoformat(timespec="seconds"), "comando": self.comando, "tipo": self.tipo,
               "costo_usd": total, "segundos_totales": round(time.time() - self.t0, 1), "pods": self.intentos}
        try:
            with open(os.path.join(self.out_dir, "costos.jsonl"), "a") as f:
                f.write(json.dumps(reg, ensure_ascii=False) + "\n")
        except OSError as e:
            log(f"no se pudo escribir costos.jsonl: {e}")
        try:
            from . import globalcfg
            globalcfg.registrar({**reg, "salida": os.path.abspath(self.out_dir)})
        except Exception as e:
            log(f"no se pudo escribir el registro global de costos: {e}")
        log(f"costo de esta corrida: ${total:.3f} (registrado en costos.jsonl)")
