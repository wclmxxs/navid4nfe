import os
from types import SimpleNamespace

from navid import config, service
from navid.store import Store


def test_dead_worker_stops_api_and_fails_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setattr(config, "DATA", tmp_path / "data")
    config.initialize_dirs()
    monkeypatch.setattr(service, "process_stamp", lambda _pid: "test-stamp")
    monkeypatch.setattr(service.signal, "signal", lambda *_: None)
    children = []
    cleaned = []
    jobs = []

    class Child:
        def __init__(self, command, **kwargs):
            self.pid = 100 + len(children)
            self.worker = "torch.distributed.run" in command
            children.append(self)
            if self.worker:
                store = Store(config.DATA)
                jobs.append(store.enqueue({"duration": 5, "seed": 1}, 1))

        def poll(self):
            return 7 if self.worker else None

    monkeypatch.setattr(service.subprocess, "Popen", Child)
    monkeypatch.setattr(service, "terminate", lambda processes: cleaned.extend(processes))
    assert service.serve() == 1
    assert cleaned == children and len(children) == 2
    assert Store(config.DATA).get(jobs[0])["status"] == "failed"
    assert config.read_json(config.RUNTIME / "supervisor.json")["phase"] == "stopped"
    assert not config.read_json(config.RUNTIME / "pid.json")


def test_pid_reuse_is_not_a_running_service(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNTIME", tmp_path)
    config.atomic_json(tmp_path / "pid.json", {"pid": os.getpid(), "stamp": "old"})
    monkeypatch.setattr(service, "process_stamp", lambda _pid: "new")
    assert service.running() is None


def test_probe_uses_configured_bind_host():
    assert service.endpoint({"host": "0.0.0.0", "port": 8000}) == "http://127.0.0.1:8000"
    assert service.endpoint({"host": "10.0.0.1", "port": 8000}) == "http://10.0.0.1:8000"
    assert service.endpoint({"host": "::", "port": 8000}) == "http://[::1]:8000"


def test_failed_launch_memory_check_does_not_start_any_process(monkeypatch):
    monkeypatch.setattr(service, "running", lambda: None)
    commands = []

    def check(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1)

    def unexpected(*args, **kwargs):
        raise AssertionError("No background service should start when GPU memory is occupied")

    monkeypatch.setattr(service.subprocess, "run", check)
    monkeypatch.setattr(service.subprocess, "Popen", unexpected)
    assert service.start() == 1
    assert commands[0][-2:] == ["navid.prepare", "gpucheck"]


def test_start_existing_service_does_not_check_its_occupied_gpus(monkeypatch):
    monkeypatch.setattr(service, "running", lambda: {"pid": 123})
    monkeypatch.setattr(service, "status", lambda: 0)

    def unexpected(*args, **kwargs):
        raise AssertionError("A running service already owns the GPUs")

    monkeypatch.setattr(service.subprocess, "run", unexpected)
    assert service.start() == 0
