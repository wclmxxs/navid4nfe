from types import SimpleNamespace

from navid import gpu_diagnostics as diagnostics


def process(root, pid, parent, name, args, cgroup="0::/system.slice/old-model.service"):
    path = root / str(pid)
    path.mkdir()
    (path / "status").write_text(f"Name:\t{name}\nPPid:\t{parent}\n")
    (path / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in args) + b"\0")
    (path / "cgroup").write_text(cgroup)
    (path / "cwd").symlink_to(root)


def test_current_workers_resolve_to_shared_launcher_and_service(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "PROC", tmp_path)
    process(tmp_path, 80, 40, "sgl_diffusion", ["sgl_diffusion::scheduler_U0"])
    process(tmp_path, 81, 40, "sgl_diffusion", ["sgl_diffusion::scheduler_U1"])
    process(tmp_path, 40, 1, "python", ["python", "-m", "sglang", "serve"])
    process(tmp_path, 1, 0, "systemd", ["/sbin/init"])
    output = diagnostics.owner_context("GPU-a, 80, sgl_diffusion, 63000 MiB\nGPU-b, 81, sgl_diffusion, 63000 MiB")
    assert "PID=80 PPID=40" in output
    assert "PID=81 PPID=40" in output
    assert output.count("PID=40 PPID=1") == 1
    assert "old-model.service" in output
    assert "python -m sglang serve" in output


def test_missing_pid_does_not_claim_memory_is_free(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "PROC", tmp_path)
    output = diagnostics.owner_context("GPU-a, 230560, sgl_diffusion, 63000 MiB")
    assert "230560 cannot be read" in output
    assert "another PID namespace" in output
    assert "does not prove GPU memory is free" in output


def test_diagnostics_query_current_pids_each_time(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "PROC", tmp_path)
    pids = iter([224828, 230560])

    def run(command, **kwargs):
        assert command[0] == "nvidia-smi"
        return SimpleNamespace(stdout=f"GPU-a, {next(pids)}, sgl_diffusion, 63000 MiB", stderr="")

    monkeypatch.setattr(diagnostics.subprocess, "run", run)
    assert "GPU PID 224828" in diagnostics.inventory()
    assert "GPU PID 230560" in diagnostics.inventory()


def test_launcher_credentials_and_inline_commands_are_omitted():
    output = diagnostics.command_text(b"python\0-m\0sglang\0--api-key\0private-key\0--token=private-token\0HF_TOKEN=private-env\0")
    assert "private-" not in output
    assert "python -m sglang" in output
    assert "private" not in diagnostics.command_text(b"bash\0-lc\0HF_TOKEN=private python server.py\0")
