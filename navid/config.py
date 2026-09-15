from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
DATA = Path(os.environ.get("DATA_DIR", ROOT / "data")).expanduser().resolve()
CHECKPOINTS = Path(os.environ.get("CHECKPOINT_DIR", ROOT / "checkpoints")).expanduser().resolve()
MODEL_REVISION = "42ed227ee7df40d41602854ae760620d6eb651fe"
ADAPTER_REVISION = "3ec17a324ced54151364f24f8b5fb6bf7e26414f"
ADAPTER_NAME = "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"
MODEL = Path(os.environ.get("MODEL_DIR", CHECKPOINTS / "MiniMax-H3")).expanduser().resolve()
ADAPTER = Path(os.environ.get("ADAPTER_PATH", CHECKPOINTS / "Minimax-h3-Turbo" / ADAPTER_NAME)).expanduser().resolve()
RUN_ID = os.environ.get("NAVID_RUN_ID", "")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".json-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def initialize_dirs() -> None:
    for path in (RUNTIME, DATA / "references", DATA / "outputs"):
        path.mkdir(parents=True, exist_ok=True)
