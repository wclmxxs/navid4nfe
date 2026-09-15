import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from navid import env_migration


def test_migration_preserves_shell_content_and_backs_up_secrets(tmp_path, capsys):
    original = (b"# custom deployment\nPORT=8123\nDATA_DIR='/data/videos with spaces'\n"
                b"HF_TOKEN='fixture-secret\nsecond-line'\nexport REF2VA_NFE=4\n"
                b"ADAPTER_PATH='/models/old-four-step.safetensors'\nDIT_COMPILE=0\nCACHE_DIT_ENABLED=0")
    path = tmp_path / ".env"
    path.write_bytes(original)
    assert env_migration.migrate(tmp_path)
    assert path.read_bytes().startswith(original)
    backups = list((tmp_path / ".runtime/env-backups").glob("*.bak"))
    assert len(backups) == 1 and backups[0].read_bytes() == original
    assert stat.S_IMODE(backups[0].stat().st_mode) == 0o600
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "fixture-secret" not in capsys.readouterr().out
    # Execute the same shell syntax as deploy.sh; preservation is semantic as
    # well as byte-for-byte, including multiline credentials and quoted paths.
    keys = [*env_migration.DEFAULTS, "PORT", "DATA_DIR", "HF_TOKEN", "ADAPTER_PATH"]
    program = f"import json,os; print(json.dumps({{k:os.environ[k] for k in {keys!r} if k in os.environ}}))"
    result = subprocess.run(["bash", "-c", 'set -a; source "$1"; exec "$2" -c "$3"',
        "load", str(path), sys.executable, program],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    values = json.loads(result.stdout)
    assert {k: values[k] for k in env_migration.DEFAULTS} == env_migration.DEFAULTS
    assert values["PORT"] == "8123" and values["DATA_DIR"] == "/data/videos with spaces"
    assert values["HF_TOKEN"] == "fixture-secret\nsecond-line"
    assert "ADAPTER_PATH" not in values


def test_restarts_preserve_later_tuning_and_do_not_create_more_backups(tmp_path):
    path = tmp_path / ".env"
    path.write_text("DIT_COMPILE=0\n")
    assert env_migration.migrate(tmp_path)
    path.write_text(path.read_text() + "REF2VA_NFE=4\nCACHE_DIT_ENABLED=0\n")
    before = path.read_bytes(), path.stat().st_mtime_ns
    assert not env_migration.migrate(tmp_path)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert len(list((tmp_path / ".runtime/env-backups").glob("*.bak"))) == 1


def test_first_install_creates_env_without_backup(tmp_path):
    assert env_migration.migrate(tmp_path)
    assert "REF2VA_NFE=8" in (tmp_path / ".env").read_text()
    assert not (tmp_path / ".runtime/env-backups").exists()


def test_failed_atomic_replace_leaves_original_and_backup(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_bytes(b"PORT=8123\n")

    def fail(*args):
        raise OSError("fixture write failure")

    monkeypatch.setattr(env_migration.os, "replace", fail)
    with pytest.raises(OSError, match="fixture write failure"):
        env_migration.migrate(tmp_path)
    assert path.read_bytes() == b"PORT=8123\n"
    assert next((tmp_path / ".runtime/env-backups").glob("*.bak")).read_bytes() == path.read_bytes()
    assert not list(tmp_path.glob(".env-migration-*"))


@pytest.fixture
def launcher(tmp_path):
    """Run the actual Bash launcher with CPU-only subprocess stand-ins."""
    root = Path(__file__).resolve().parents[1]
    shutil.copy(root / "deploy.sh", tmp_path)
    (tmp_path / "navid").mkdir()
    for name in ("__init__.py", "env_migration.py"):
        shutil.copy(root / "navid" / name, tmp_path / "navid")
    (tmp_path / ".venv/bin").mkdir(parents=True)
    python = tmp_path / ".venv/bin/python"
    keys = [*env_migration.DEFAULTS, "PORT", "ADAPTER_PATH", "ENV_LOAD_COUNT"]
    python.write_text(f"#!{sys.executable}\nkeys = {keys!r}\n" + '''import json, os, pathlib, subprocess, sys
args = sys.argv[1:]
if args and args[0].endswith("/navid/env_migration.py"):
    raise SystemExit(subprocess.call([sys.executable, *args]))
if args[:2] == ["-m", "navid.service"]:
    with pathlib.Path("observed.jsonl").open("a") as stream:
        stream.write(json.dumps({"action": args[2], "env": {k: os.environ[k] for k in keys if k in os.environ}}) + "\\n")
elif args[:1] == ["-c"]:
    print("/fixture/ptxas")
else:
    raise SystemExit("Unexpected command: " + repr(args))
''')
    python.chmod(0o700)
    (tmp_path / "bin").mkdir()
    uname = tmp_path / "bin/uname"
    uname.write_text("#!/bin/sh\nprintf '%s\\n' Linux\n")
    uname.chmod(0o700)
    (tmp_path / ".env").write_text("PORT=8123\nREF2VA_NFE=4\nDIT_COMPILE=0\n"
        "CACHE_DIT_ENABLED=0\nADAPTER_PATH=/models/old-four-step.safetensors\n"
        "ENV_LOAD_COUNT=$(( ${ENV_LOAD_COUNT:-0} + 1 ))\n")
    env = dict(os.environ)
    for key in (*env_migration.DEFAULTS, "ADAPTER_PATH", "PORT", "ENV_LOAD_COUNT"):
        env.pop(key, None)
    env["PATH"] = str(tmp_path / "bin") + os.pathsep + env["PATH"]
    return tmp_path, env


@pytest.mark.parametrize("action,expected", [("start", ["start"]), ("restart", ["stop", "start"]),
                                           ("deploy", ["is-running", "start"])])
def test_launcher_migrates_before_config_imports_and_reloads_env(launcher, action, expected):
    root, env = launcher
    result = subprocess.run(["bash", "deploy.sh", action], cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in (root / "observed.jsonl").read_text().splitlines()]
    assert [r["action"] for r in records] == expected
    for record in records:
        values = record["env"]
        assert {k: values[k] for k in env_migration.DEFAULTS} == env_migration.DEFAULTS
        assert values["PORT"] == "8123" and "ADAPTER_PATH" not in values
        assert values["ENV_LOAD_COUNT"] == "1"


def test_explicit_shell_overrides_still_win(launcher):
    root, env = launcher
    env.update(REF2VA_NFE="4", CACHE_DIT_ENABLED="0", ADAPTER_PATH="/models/explicit-four-step.safetensors")
    result = subprocess.run(["bash", "deploy.sh", "start"], cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    values = json.loads((root / "observed.jsonl").read_text())["env"]
    assert values["REF2VA_NFE"] == "4" and values["CACHE_DIT_ENABLED"] == "0"
    assert values["ADAPTER_PATH"] == env["ADAPTER_PATH"]


@pytest.mark.parametrize("action", ["stop", "status", "errors"])
def test_management_commands_do_not_migrate_env(launcher, action):
    root, env = launcher
    before = (root / ".env").read_bytes()
    result = subprocess.run(["bash", "deploy.sh", action], cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (root / ".env").read_bytes() == before
    assert not (root / ".runtime/env-backups").exists()
