"""Live, read-only GPU owner diagnostics. Uses the standard library, not CUDA."""
from __future__ import annotations

import csv
import io
import os
import re
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROC = Path("/proc")
SECRET = re.compile(r"token|password|passwd|secret|api[-_]?key|access[-_]?key", re.IGNORECASE)


def command_text(raw: bytes) -> str:
    # Do not print credentials that launchers may have received as arguments.
    args = raw.decode(errors="replace").strip("\0").split("\0")
    hidden_next = False
    result = []
    for arg in args:
        if hidden_next:
            result.append("<redacted>")
            hidden_next = False
        elif arg in {"-c", "-lc", "-ic"}:
            result.append(arg)
            hidden_next = True
        elif (arg.startswith("-") or "=" in arg) and SECRET.search(arg.split("=", 1)[0]):
            if "=" in arg:
                result.append(arg.split("=", 1)[0] + "=<redacted>")
            else:
                result.append(arg)
                hidden_next = True
        else:
            result.append(arg)
    return shlex.join(result)[:3000]


def owner_context(csv_text: str) -> str:
    pids = set()
    for row in csv.reader(io.StringIO(csv_text)):
        if len(row) > 1 and row[1].strip().isdigit():
            pids.add(int(row[1].strip()))
    lines = ["\nLive process ancestry (GPU worker -> launcher; shared parents shown once):"]
    shown = set()
    for worker in sorted(pids):
        lines.append(f"GPU PID {worker}:")
        pid = worker
        for _ in range(16):
            if pid in shown:
                lines.append(f"  -> PID {pid} (already shown)")
                break
            path = PROC / str(pid)
            try:
                status = dict(line.split(":", 1) for line in (path / "status").read_text().splitlines() if ":" in line)
                parent = int(status["PPid"].strip())
                name = status.get("Name", "?").strip()
                cmd = command_text((path / "cmdline").read_bytes())
                lines.append(f"  PID={pid} PPID={parent} name={name} command={cmd}")
                try:
                    lines.append("    cgroup: " + (path / "cgroup").read_text().strip().replace("\n", " | "))
                except OSError:
                    pass
                try:
                    lines.append(f"    cwd: {os.readlink(path / 'cwd')}")
                except OSError:
                    pass
            except (OSError, KeyError, ValueError) as error:
                lines.append(f"  PID {pid} cannot be read in this /proc: {error}.")
                lines.append("  It may have exited, be hidden by permissions, or belong to another PID namespace; this does not prove GPU memory is free.")
                break
            shown.add(pid)
            if parent <= 0:
                break
            pid = parent
    return "\n".join(lines)


def run_readonly(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        return result.stdout.strip() or result.stderr.strip() or "No entries reported."
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"Could not query GPU processes / runtime: {error}"


def inventory() -> str:
    raw = run_readonly(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv"])
    return "GPU process inventory (read-only; no processes will be stopped):\n" + raw + owner_context(raw)


def report() -> str:
    lines = [f"GPU diagnostics at {datetime.now(timezone.utc).isoformat()}; local PID={os.getpid()}",
             "Read-only: no CUDA context, model loading, process termination or container changes.",
             run_readonly(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free", "--format=csv"]),
             inventory()]
    # Explicitly selected metadata only; never dump container environment/config.
    if shutil.which("docker"):
        lines.append("\nRunning Docker containers: ID | name | image | Compose project | Compose service | status")
        lines.append(run_readonly(["docker", "ps", "--no-trunc", "--format",
                                  '{{.ID}} | {{.Names}} | {{.Image}} | {{.Label "com.docker.compose.project"}} | {{.Label "com.docker.compose.service"}} | {{.Status}}']))
    else:
        lines.append("Docker CLI is not available in this shell.")
    lines.append("\nUse the owning service/container launcher to stop an intended old workload. Do not rely on PIDs from earlier logs.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(), flush=True)
