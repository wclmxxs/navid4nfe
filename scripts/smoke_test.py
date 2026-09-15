#!/usr/bin/env python3
"""Exercise the deployed HTTP API using real reference files. Standard library only."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=os.environ.get("API_KEY"))
    parser.add_argument("--reference", action="append", required=True, help="image:PATH, video:PATH or audio:PATH; repeat in order")
    parser.add_argument("--prompt", default="A continuous cinematic shot featuring the subject in Picture 1. Natural motion and ambient sound.")
    parser.add_argument("--duration", type=int, choices=range(4, 16), default=5)
    parser.add_argument("--nfe", type=int, choices=(4, 8), help="Assert the server's resident profile")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--reference-short-edge", type=int)
    parser.add_argument("--sol", action=argparse.BooleanOptionalAction, default=None,
                        help="Override Sol; omitted options inherit the server defaults")
    parser.add_argument("--tau", type=float)
    parser.add_argument("--cache-dit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--rdt", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output", type=Path, default=Path("data/smoke.mp4"))
    args = parser.parse_args()
    key_path = Path(__file__).resolve().parents[1] / ".runtime/api.key"
    key = args.api_key or key_path.read_text().strip()
    base = args.url.rstrip("/")

    def call(path, data=None, content_type="application/json"):
        request = urllib.request.Request(base + path, data=data,
                                         headers={"X-API-Key": key, "Content-Type": content_type})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"HTTP {error.code}: {error.read().decode(errors='replace')}") from error

    refs = []
    for spec in args.reference:
        kind, separator, name = spec.partition(":")
        if not separator or kind not in {"image", "video", "audio"}:
            parser.error("--reference must be image:PATH, video:PATH or audio:PATH")
        ref = json.loads(call(f"/v1/references?kind={kind}", Path(name).read_bytes(), "application/octet-stream"))
        refs.append(ref["id"])
    optimization = {"sol_attn": {}, "cache_dit": {}}
    for group, field, value in (("sol_attn", "enabled", args.sol), ("sol_attn", "tau", args.tau),
                                ("cache_dit", "enabled", args.cache_dit), ("cache_dit", "rdt", args.rdt)):
        if value is not None:
            optimization[group][field] = value
    payload = {"prompt": args.prompt, "duration": args.duration, "seed": args.seed, "references": refs,
               "optimization": optimization}
    for name in ("width", "height", "reference_short_edge", "nfe"):
        if getattr(args, name) is not None:
            payload[name] = getattr(args, name)
    job = json.loads(call("/v1/videos", json.dumps(payload).encode()))
    print(f"Submitted task {job['id']}", flush=True)
    deadline = time.monotonic() + args.timeout
    previous = None
    while time.monotonic() < deadline:
        result = json.loads(call(job["status_url"]))
        if result["status"] != previous:
            print(result["status"], flush=True)
            previous = result["status"]
        if result["status"] == "failed":
            raise RuntimeError(result["error"])
        if result["status"] == "succeeded":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(call(result["content_url"]))
            args.output.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
            print(json.dumps(result.get("metrics"), ensure_ascii=False))
            print(f"Saved {args.output.resolve()} (GPU pipeline {result['inference_s']} s, NFE={result['nfe']})")
            return
        time.sleep(2)
    raise TimeoutError(f"Task {job['id']} did not complete within {args.timeout} seconds")


if __name__ == "__main__":
    main()
