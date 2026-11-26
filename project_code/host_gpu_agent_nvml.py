#!/usr/bin/env python3
"""
GPU energy agent (NVML based).

Endpoints (all require "Authorization: Bearer <A_LEMS_TOKEN>"):
  POST /baseline?gpu=N&seconds=10   measure idle power of GPU N (GPU must be idle)
  POST /start?gpu=N&name=<run>      begin a run on GPU N; <run> names its files
  POST /stop?gpu=N                  end the run, return energy summary as JSON
  GET  /status                      list GPUs, counter support, active runs, baselines

Energy method:
  * Volta+ GPUs: difference of the hardware cumulative energy counter
    (nvmlDeviceGetTotalEnergyConsumption) between /start and /stop.
  * Older GPUs (or counter reset by a driver reload): trapezoidal integration
    of sampled power. Less accurate, and the response says which method was used.

A CSV time series (wall-clock timestamps) and a JSON summary are written per run.
They are named after the run when the caller passes ``name`` (the driver sends the
episode's own stem, so the mount and the batch folder speak the same name); a caller
that does not - or an older driver - gets the timestamped ``gpu<N>_<UTC>`` name.

Environment:
  A_LEMS_TOKEN       required
  A_LEMS_HOST        bind address            (default 127.0.0.1)
  A_LEMS_PORT        port                    (default 8765)
  A_LEMS_GPU         default GPU index       (default 0)
  A_LEMS_SAMPLE_MS   time-series interval    (default 1000, min 100)

Requires: pip install nvidia-ml-py
No nv-hostengine / DCGM needed.
"""

import csv
import hmac
import json
import os
import re
import signal
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pynvml

TOKEN = os.environ["A_LEMS_TOKEN"]
HOST = os.environ.get("A_LEMS_HOST", "0.0.0.0")
PORT = int(os.environ.get("A_LEMS_PORT", "8765"))
DEFAULT_GPU = int(os.environ.get("A_LEMS_GPU", "0"))
SAMPLE_S = max(100, int(os.environ.get("A_LEMS_SAMPLE_MS", "1000"))) / 1000.0

LOG_DIR = Path.home() / "alems-telemetry"
LOG_DIR.mkdir(parents=True, exist_ok=True)

lock = threading.Lock()
meters = {}      # gpu index -> GpuMeter (filled at startup, read-only afterwards)
active = {}      # gpu index -> Run
baselining = set()
baselines = {}   # gpu index -> idle watts

#: A run name has to survive as one file-name component on the host, so anything
#: that is not a plain character is folded to ``_`` before it reaches the disk.
UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(text):
    """A caller's run name reduced to one file-name component, or ``None``.

    ``basename`` first, so a name can never climb out of ``LOG_DIR``; the
    substitution then removes the characters that a network share, a Windows
    mount or a shell would object to.
    """
    if not text:
        return None
    stem = UNSAFE.sub("_", os.path.basename(str(text).strip().replace("\\", "/")))
    stem = stem.strip("._")[:120]
    return stem if stem and stem not in (".", "..") else None


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class GpuMeter:
    def __init__(self, index):
        self.index = index
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(self.handle)
        self.name = name.decode() if isinstance(name, bytes) else name
        self.has_counter = self._probe_counter()

    def _probe_counter(self):
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            return True
        except pynvml.NVMLError:
            return False

    def energy_mj(self):
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)

    def power_w(self):
        return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    def utilization_pct(self):
        try:
            return pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu
        except pynvml.NVMLError:
            return None


class Run:
    def __init__(self, meter, name=None):
        self.meter = meter
        self.start_dt = datetime.now(timezone.utc)
        stamp = "gpu%d_%s" % (meter.index, self.start_dt.strftime("%Y%m%dT%H%M%S"))
        # The caller's name when it has one, the timestamp otherwise; a re-measured
        # episode reuses its name, so the second attempt must not overwrite the
        # first attempt's evidence - it gets a ``_2``, ``_3``, ... suffix instead.
        stem = safe_name(name) or stamp
        candidate, n = stem, 1
        while (LOG_DIR / (candidate + ".csv")).exists() or \
                (LOG_DIR / (candidate + ".json")).exists():
            n += 1
            candidate = "%s_%d" % (stem, n)
        self.run_id = candidate
        self.requested_name = safe_name(name)
        self.csv_path = LOG_DIR / (self.run_id + ".csv")
        self.json_path = LOG_DIR / (self.run_id + ".json")

        self.fh = open(self.csv_path, "w", newline="", buffering=1)
        self.writer = csv.writer(self.fh)
        self.writer.writerow(["unix_time_s", "utc", "energy_mj", "power_w"])

        self.first = None
        self.last = None
        self.prev_mono = None
        self.prev_power = None
        self.integrated_j = 0.0

        self.stop_evt = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def sample(self):
        mono = time.monotonic()
        wall = time.time()
        energy = self.meter.energy_mj() if self.meter.has_counter else None
        power = self.meter.power_w()

        if self.prev_mono is not None:
            self.integrated_j += 0.5 * (power + self.prev_power) * (mono - self.prev_mono)
        self.prev_mono, self.prev_power = mono, power

        rec = {"wall": wall, "mono": mono, "energy_mj": energy, "power_w": power}
        if self.first is None:
            self.first = rec
        self.last = rec

        self.writer.writerow([
            "%.6f" % wall,
            datetime.fromtimestamp(wall, timezone.utc).isoformat(timespec="milliseconds"),
            "" if energy is None else energy,
            "%.3f" % power,
        ])

    def _loop(self):
        while not self.stop_evt.wait(SAMPLE_S):
            try:
                self.sample()
            except pynvml.NVMLError:
                pass

    def begin(self):
        self.sample()          # exact start reading, taken synchronously
        self.thread.start()

    def abort(self):
        self.stop_evt.set()
        if self.thread.is_alive():
            self.thread.join()
        self.fh.close()

    def finish(self, baseline_w):
        self.stop_evt.set()
        self.thread.join()
        self.sample()          # exact stop reading, taken synchronously
        self.fh.close()

        duration = self.last["mono"] - self.first["mono"]
        method = "integrated_power"
        energy_j = self.integrated_j
        counter_reset = False

        if self.meter.has_counter:
            delta_mj = self.last["energy_mj"] - self.first["energy_mj"]
            if delta_mj >= 0:
                method = "nvml_total_energy_counter"
                energy_j = delta_mj / 1000.0
            else:
                counter_reset = True  # driver reloaded mid-run; fall back

        summary = {
            "run_id": self.run_id,
            "gpu": self.meter.index,
            "gpu_name": self.meter.name,
            "method": method,
            "start_utc": datetime.fromtimestamp(self.first["wall"], timezone.utc).isoformat(),
            "stop_utc": datetime.fromtimestamp(self.last["wall"], timezone.utc).isoformat(),
            "start_unix_s": self.first["wall"],
            "stop_unix_s": self.last["wall"],
            "duration_s": round(duration, 6),
            "energy_j": round(energy_j, 3),
            "avg_power_w": round(energy_j / duration, 3) if duration > 0 else None,
            "csv": str(self.csv_path),
        }
        if counter_reset:
            summary["warning"] = "energy counter went backwards; used integrated power"
        if baseline_w is not None:
            summary["baseline_w"] = round(baseline_w, 3)
            summary["net_energy_j"] = round(energy_j - baseline_w * duration, 3)

        self.json_path.write_text(json.dumps(summary, indent=2))
        return summary


# ---------------------------------------------------------------- operations

def start_run(meter, name=None):
    with lock:
        if meter.index in active:
            raise ApiError(409, "Run already active on GPU %d" % meter.index)
        if meter.index in baselining:
            raise ApiError(409, "Baseline measurement in progress on GPU %d" % meter.index)
        run = Run(meter, name)
        try:
            run.begin()
        except Exception:
            run.abort()
            raise
        active[meter.index] = run
    return {
        "run_id": run.run_id,
        "gpu": meter.index,
        "gpu_name": meter.name,
        "method": "nvml_total_energy_counter" if meter.has_counter else "integrated_power",
        "start_utc": run.start_dt.isoformat(),
        "csv": str(run.csv_path),
    }


def stop_run(meter):
    with lock:
        run = active.pop(meter.index, None)
        baseline_w = baselines.get(meter.index)
    if run is None:
        raise ApiError(409, "No active run on GPU %d" % meter.index)
    return run.finish(baseline_w)


def measure_baseline(meter, seconds):
    with lock:
        if meter.index in active:
            raise ApiError(409, "Run active on GPU %d" % meter.index)
        if meter.index in baselining:
            raise ApiError(409, "Baseline already running on GPU %d" % meter.index)
        util = meter.utilization_pct()
        if util is not None and util > 5:
            raise ApiError(409, "GPU %d is not idle (utilization %d%%)" % (meter.index, util))
        baselining.add(meter.index)

    try:
        if meter.has_counter:
            e0, t0 = meter.energy_mj(), time.monotonic()
            time.sleep(seconds)
            e1, t1 = meter.energy_mj(), time.monotonic()
            watts = (e1 - e0) / 1000.0 / (t1 - t0)
        else:
            samples = []
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                samples.append(meter.power_w())
                time.sleep(0.1)
            watts = sum(samples) / len(samples)
    finally:
        with lock:
            baselining.discard(meter.index)

    with lock:
        baselines[meter.index] = watts
    return {"gpu": meter.index, "baseline_w": round(watts, 3), "seconds": seconds}


def status():
    with lock:
        return {
            "gpus": [
                {
                    "index": m.index,
                    "name": m.name,
                    "energy_counter": m.has_counter,
                    "baseline_w": baselines.get(m.index),
                    "active_run": active[m.index].run_id if m.index in active else None,
                }
                for m in meters.values()
            ]
        }


# ---------------------------------------------------------------- HTTP layer

class Handler(BaseHTTPRequestHandler):

    def _json(self, status_code, obj):
        body = json.dumps(obj).encode()
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self):
        supplied = self.headers.get("Authorization", "").encode()
        return hmac.compare_digest(supplied, ("Bearer " + TOKEN).encode())

    def _handle(self, method):
        if not self._authorized():
            self._json(401, {"error": "Unauthorized"})
            return

        url = urlparse(self.path)
        query = parse_qs(url.query)

        try:
            if method == "GET" and url.path == "/status":
                self._json(200, status())
                return

            if method != "POST" or url.path not in ("/start", "/stop", "/baseline"):
                raise ApiError(404, "Unknown endpoint")

            try:
                gpu = int(query.get("gpu", [DEFAULT_GPU])[0])
            except ValueError:
                raise ApiError(400, "gpu must be an integer")
            meter = meters.get(gpu)
            if meter is None:
                raise ApiError(400, "Unknown GPU index %d (have %s)" % (gpu, sorted(meters)))

            if url.path == "/start":
                self._json(200, start_run(meter, (query.get("name") or [None])[0]))
            elif url.path == "/stop":
                self._json(200, stop_run(meter))
            else:
                try:
                    seconds = float(query.get("seconds", ["10"])[0])
                except ValueError:
                    raise ApiError(400, "seconds must be a number")
                if not 1 <= seconds <= 300:
                    raise ApiError(400, "seconds must be between 1 and 300")
                self._json(200, measure_baseline(meter, seconds))

        except ApiError as exc:
            self._json(exc.status, {"error": str(exc)})
        except Exception as exc:  # NVML failures, disk errors, ...
            self._json(500, {"error": "%s: %s" % (type(exc).__name__, exc)})

    def do_POST(self):
        self._handle("POST")

    def do_GET(self):
        self._handle("GET")


# ---------------------------------------------------------------- lifecycle

def _terminate(signum, frame):
    raise SystemExit(0)


def shutdown_all():
    with lock:
        runs = list(active.values())
        active.clear()
    for run in runs:
        try:
            run.abort()
        except Exception:
            pass
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass


def main():
    pynvml.nvmlInit()
    for i in range(pynvml.nvmlDeviceGetCount()):
        m = GpuMeter(i)
        meters[i] = m
        print("GPU %d: %s | energy counter: %s" % (
            i, m.name, "yes" if m.has_counter else "no (falling back to integrated power)"))

    if DEFAULT_GPU not in meters:
        raise SystemExit("A_LEMS_GPU=%d does not exist" % DEFAULT_GPU)

    signal.signal(signal.SIGTERM, _terminate)

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("GPU energy agent listening on %s:%d (default GPU %d)" % (HOST, PORT, DEFAULT_GPU))
    try:
        server.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        server.server_close()
        shutdown_all()


if __name__ == "__main__":
    main()
