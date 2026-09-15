"""Background supervisor and lifecycle commands; no systemd or Docker required."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import secrets
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from . import config
from .store import Store


def process_stamp(pid: int) -> str | None:
    try:
        # /proc comm can contain spaces and parentheses; fields after ')' start at 3.
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text.rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (FileNotFoundError, ProcessLookupError, IndexError):
        return None


def running() -> dict | None:
    info = config.read_json(config.RUNTIME / "pid.json")
    pid = info.get("pid")
    if isinstance(pid, int) and info.get("stamp") and process_stamp(pid) == info["stamp"]:
        return info
    return None


def endpoint(info: dict) -> str:
    host = info.get("host", "0.0.0.0")
    if host == "0.0.0.0":
        host = "127.0.0.1"
    elif host == "::":
        host = "::1"
    if ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{info['port']}"


def child_command(*command: str) -> list[str]:
    # A small exec wrapper installs parent-death handling without preexec_fn.
    return [sys.executable, "-m", "navid.child", str(os.getpid()), *command]


def status() -> int:
    info = running()
    if not info:
        print("Service is stopped. Logs: .runtime/service.log and .runtime/worker.log")
        return 1
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"{endpoint(info)}/readyz", timeout=3) as response:
            value = json.load(response)
        print(json.dumps(value, ensure_ascii=False))
        return 0 if value.get("ready") else 1
    except (urllib.error.URLError, ValueError):
        worker = config.read_json(config.RUNTIME / "worker.json")
        print(f"Service PID {info['pid']} is starting/unhealthy: {worker.get('phase', 'starting')}")
        return 1


def stop() -> int:
    info = running()
    if not info:
        print("Service is already stopped.")
        return 0
    os.kill(info["pid"], signal.SIGTERM)
    for _ in range(300):
        if not running():
            print("Service stopped.")
            return 0
        time.sleep(0.1)
    print("Supervisor did not stop within 30 seconds; inspect .runtime/service.log", file=sys.stderr)
    return 1


def terminate(children: list[subprocess.Popen]) -> None:
    for child in children:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15
    for child in children:
        try:
            child.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)


def serve() -> int:
    with (config.RUNTIME / "supervisor.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another supervisor already owns this checkout.", flush=True)
            return 1
        run_id = uuid.uuid4().hex
        port = int(os.environ.get("PORT", "8000"))
        config.atomic_json(config.RUNTIME / "pid.json", {
            "pid": os.getpid(), "stamp": process_stamp(os.getpid()), "run_id": run_id,
            "port": port, "host": os.environ.get("HOST", "0.0.0.0")})
        children = []
        stopping = False
        reason = "Service stopped"

        def handle_signal(_sig, _frame):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
        store = Store(config.DATA)
        store.fail_pending("Previous service run ended before task completion")
        env = {**os.environ, "NAVID_RUN_ID": run_id, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        code = 0
        started = time.monotonic()
        ready_seen = False
        try:
            with (config.RUNTIME / "api.log").open("ab", buffering=0) as api_log, (config.RUNTIME / "worker.log").open("ab", buffering=0) as worker_log:
                children.append(subprocess.Popen(
                    child_command(sys.executable, "-m", "uvicorn", "navid.api:app", "--host", os.environ.get("HOST", "0.0.0.0"),
                                  "--port", str(port), "--workers", "1"), cwd=config.ROOT, env=env,
                    stdout=api_log, stderr=subprocess.STDOUT, start_new_session=True))
                children.append(subprocess.Popen(
                    child_command(sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=8",
                                  "--max-restarts=0", "--module", "navid.worker"), cwd=config.ROOT, env=env,
                    stdout=worker_log, stderr=subprocess.STDOUT, start_new_session=True))
                while not stopping:
                    for label, child in zip(("API", "GPU worker"), children):
                        result = child.poll()
                        if result is not None:
                            raise RuntimeError(f"{label} exited with code {result}")
                    worker = config.read_json(config.RUNTIME / "worker.json")
                    if worker.get("run_id") == run_id:
                        ready_seen |= worker.get("phase") in {"ready", "busy"}
                        if worker.get("phase") == "busy" and time.time() - worker["since"] > int(os.environ.get("TASK_TIMEOUT", "1800")):
                            raise RuntimeError(f"Task {worker.get('job_id')} exceeded TASK_TIMEOUT")
                    if not ready_seen and time.monotonic() - started > int(os.environ.get("READY_TIMEOUT", "7200")):
                        raise RuntimeError("Model loading / Ref2VA warmup exceeded READY_TIMEOUT")
                    config.atomic_json(config.RUNTIME / "supervisor.json", {
                        "run_id": run_id, "phase": "running", "heartbeat": time.time()})
                    time.sleep(0.5)
        except Exception as error:  # noqa: BLE001 -- all supervisor failures must tear down both children
            reason = str(error)
            code = 1
            print(f"SERVICE_FAILED: {reason}", flush=True)
        finally:
            config.atomic_json(config.RUNTIME / "supervisor.json", {
                "run_id": run_id, "phase": "stopped", "heartbeat": time.time(), "reason": reason})
            terminate(children)
            store.fail_pending(reason)
            config.atomic_json(config.RUNTIME / "pid.json", {})
        return code


def start() -> int:
    if running():
        print("Service already running; use './deploy.sh restart' to apply changes.")
        return status()
    key_path = config.RUNTIME / "api.key"
    try:
        with key_path.open("x") as stream:
            stream.write(secrets.token_hex(32) + "\n")
        key_path.chmod(0o600)
    except FileExistsError:
        pass
    with (config.RUNTIME / "service.log").open("ab", buffering=0) as log:
        child = subprocess.Popen([sys.executable, "-m", "navid.service", "serve"], cwd=config.ROOT,
                                 stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    deadline = time.monotonic() + int(os.environ.get("READY_TIMEOUT", "7200")) + 30
    last_notice = 0
    try:
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError("Startup failed; see .runtime/worker.log, .runtime/api.log and .runtime/service.log")
            info = running()
            worker = config.read_json(config.RUNTIME / "worker.json")
            if (info and worker.get("run_id") == info.get("run_id")
                    and worker.get("phase") == "ready" and status() == 0):
                print(f"READY: {endpoint(info)}  |  docs: /docs")
                print(f"API key file: {key_path}")
                print(f"Startup sample: {config.RUNTIME / 'warmup.mp4'}")
                return 0
            if time.monotonic() - last_notice >= 30:
                print(f"Waiting for eight-GPU Ref2VA validation: {worker.get('phase', 'starting')} (./deploy.sh logs)", flush=True)
                last_notice = time.monotonic()
            time.sleep(1)
        raise RuntimeError("Timed out waiting for readiness")
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 -- cancel every incomplete startup
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=25)
        print(str(error), file=sys.stderr)
        for name in ("worker.log", "api.log", "service.log"):
            path = config.RUNTIME / name
            if path.exists():
                print(f"--- {name} (last 30 lines) ---", file=sys.stderr)
                print("\n".join(path.read_text(errors="replace").splitlines()[-30:]), file=sys.stderr)
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "status", "serve", "is-running"])
    args = parser.parse_args()
    config.initialize_dirs()
    if args.action == "is-running":
        raise SystemExit(0 if running() else 1)
    raise SystemExit(globals()[args.action]())
