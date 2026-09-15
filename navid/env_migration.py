"""Versioned startup defaults, independent of CUDA and runtime configuration."""
from __future__ import annotations

import argparse
import fcntl
import os
import tempfile
import time
from pathlib import Path


# Bump this when changing shipped defaults so existing deployments upgrade too.
VERSION = "ref2va8-sol15-cache-compile-v1"
MARKER = f"# Navid configuration migration: {VERSION}"
DEFAULTS = {
    "REF2VA_NFE": "8",
    "SOL_ATTN_ENABLED": "1",
    "SOL_ATTN_TAU": "1.5",
    "SOL_ATTN_DENSE_STEPS": "1",
    "CACHE_DIT_ENABLED": "1",
    "CACHE_DIT_WARMUP": "1",
    "CACHE_DIT_RDT": "0.08",
    "CACHE_DIT_MAX_CONTINUOUS": "1",
    "DIT_COMPILE": "1",
    "VAE_COMPILE": "0",
}


def migrate(root: Path) -> bool:
    runtime = root / ".runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / "env-migration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = (root / ".env").resolve()
        original = path.read_bytes() if path.exists() else b""
        if MARKER.encode() in original.splitlines():
            return False

        backup = None
        if path.exists():
            backups = runtime / "env-backups"
            backups.mkdir(mode=0o700, exist_ok=True)
            backup = backups / f"env-{VERSION}-{time.time_ns()}.bak"
            fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(original)
                stream.flush()
                os.fsync(stream.fileno())

        # .env is shell syntax, including quoted multiline values. Keep it
        # byte-for-byte and append a final override block instead of parsing or
        # reconstructing credentials, paths, comments or unrelated settings.
        block = ["", "# Managed startup defaults; this final block overrides earlier values.",
                 "# Migration runs once per version. Later edits to this block are preserved."]
        block.extend(f"{key}={value}" for key, value in DEFAULTS.items())
        block.extend(["# Select the pinned LoRA automatically from REF2VA_NFE / CHECKPOINT_DIR.",
                      "unset ADAPTER_PATH", MARKER, ""])
        content = original + (b"\n" if original and not original.endswith(b"\n") else b"")
        content += "\n".join(block).encode()
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".env-migration-")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        print(f"Configuration upgraded: {VERSION}; defaults saved to .env."
              + (f" Previous file backed up to {backup}." if backup else ""), flush=True)
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    migrate(parser.parse_args().root)
