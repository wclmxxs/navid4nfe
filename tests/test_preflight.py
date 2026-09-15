import subprocess
from types import SimpleNamespace

import pytest

from navid import prepare


class GPUs:
    def __init__(self, free_gib):
        self.free_gib = free_gib
        self.inspected = []

    def device_count(self):
        return len(self.free_gib)

    def get_device_properties(self, _index):
        return SimpleNamespace(name="NVIDIA H200", major=9, minor=0)

    def mem_get_info(self, index):
        self.inspected.append(index)
        return int(self.free_gib[index] * 1024**3), 140 * 1024**3


def test_low_memory_reports_all_gpus_and_processes(monkeypatch, capsys):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="GPU-abc, 1234, python, 85000 MiB", stderr="")

    monkeypatch.setattr(prepare.subprocess, "run", run)
    cuda = GPUs([54.3, 130, 130, 130, 130, 130, 130, 60])
    with pytest.raises(RuntimeError, match="eight GPUs do not divide") as error:
        prepare.check_gpus(cuda)
    assert cuda.inspected == list(range(8))
    assert "GPU 0 has only 54.3" in str(error.value)
    assert "GPU 7 has only 60.0" in str(error.value)
    assert "1234, python" in capsys.readouterr().out
    assert len(commands) == 1
    assert commands[0][0] == "nvidia-smi"
    assert commands[0][1].startswith("--query-compute-apps=")


def test_failed_process_query_preserves_memory_error(monkeypatch, capsys):
    def fail(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 10)

    monkeypatch.setattr(prepare.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="GPU 0 has only 54.3"):
        prepare.check_gpus(GPUs([54.3] * 8))
    assert "Could not query GPU processes" in capsys.readouterr().out


def test_available_gpus_do_not_query_or_stop_processes(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("No process commands should run on successful preflight")

    monkeypatch.setattr(prepare.subprocess, "run", unexpected)
    prepare.check_gpus(GPUs([130] * 8))


def test_late_check_detects_memory_taken_after_initial_preflight(monkeypatch):
    inventories = []
    monkeypatch.setattr(prepare, "print_gpu_processes", lambda: inventories.append(True))
    cuda = GPUs([139.3] * 8)
    prepare.check_gpus(cuda)
    cuda.free_gib[0] = 96.2  # Another process used 43.1 GiB during the download.
    with pytest.raises(RuntimeError, match="GPU 0 has only 96.2"):
        prepare.check_gpus(cuda)
    assert inventories == [True]


def test_serial_load_check_ignores_weights_on_already_loaded_ranks(monkeypatch):
    def unexpected():
        raise AssertionError("Already loaded ranks should not fail the next rank's check")

    monkeypatch.setattr(prepare, "print_gpu_processes", unexpected)
    cuda = GPUs([5, 5, 139.3, 139.3, 139.3, 139.3, 139.3, 139.3])
    prepare.check_gpu_memory(cuda, [2])
    assert cuda.inspected == [2]
