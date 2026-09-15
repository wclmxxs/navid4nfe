#!/usr/bin/env python3
"""Exercise the live eight-GPU API; retains videos/metrics for human A/B review."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--url", default=f"http://127.0.0.1:{os.environ.get('PORT', '8000')}")
    parser.add_argument("--reference", type=Path, default=root / ".runtime/warmup-reference.png")
    parser.add_argument("--output", type=Path, default=root / "data/tuning-validation")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    key = os.environ.get("API_KEY") or (root / ".runtime/api.key").read_text().strip()

    def call(path, data=None, content_type="application/json"):
        request = urllib.request.Request(args.url.rstrip("/") + path, data=data,
            headers={"X-API-Key": key, "Content-Type": content_type})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"HTTP {error.code}: {error.read().decode(errors='replace')}") from error

    reference = json.loads(call("/v1/references?kind=image", args.reference.read_bytes(),
                                "application/octet-stream"))["id"]
    prompt = "A continuous realistic shot of the subject in Picture 1. The camera slowly moves closer. Natural ambient sound."
    cases = [
        ("dense-8s-720", 8, 1280, 720, False, False),
        ("sol-8s-720", 8, 1280, 720, True, False),
        ("cache-8s-720", 8, 1280, 720, False, True),
        ("both-8s-720", 8, 1280, 720, True, True),
        ("dense-after-overrides", 8, 1280, 720, False, False),
        ("portrait-4s-1080", 4, 1080, 1920, False, False),
        ("square-15s-512", 15, 512, 512, False, False),
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for name, duration, width, height, sol, cache in cases:
        payload = {"prompt": prompt, "duration": duration, "width": width, "height": height,
            "references": [reference], "seed": 42, "reference_short_edge": 512,
            "optimization": {"sol_attn": {"enabled": sol, "tau": 1.0, "dense_steps": 1},
                             "cache_dit": {"enabled": cache, "rdt": 0.08}}}
        task = json.loads(call("/v1/videos", json.dumps(payload).encode()))
        print(f"{name}: {task['id']}", flush=True)
        deadline = time.monotonic() + args.timeout
        while True:
            result = json.loads(call(task["status_url"]))
            if result["status"] == "failed":
                raise RuntimeError(f"{name}: {result['error']}")
            if result["status"] == "succeeded":
                break
            if time.monotonic() > deadline:
                raise TimeoutError(name)
            time.sleep(2)
        path = args.output / f"{name}.mp4"
        path.write_bytes(call(result["content_url"]))
        path.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        from navid.media import verify_output

        verify_output(path, expected_frames=duration * 24, expected_size=(width, height), expected_duration=duration)
        metrics = result["metrics"]
        assert (metrics["sol_attn"]["sparse_calls"] > 0) == sol, metrics
        if not cache:
            assert metrics["cache_dit"]["cached_steps"] == 0, metrics
        assert metrics["compile"]["padded_tokens"] % 4096 == 0, metrics
        row = {"case": name, "inference_s": result["inference_s"], "metrics": metrics}
        results.append(row)
        print(f"PASS {name}: {result['inference_s']}s, cached_steps={metrics['cache_dit']['cached_steps']}", flush=True)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Validation completed: {args.output}. Review the videos for identity, motion and audio quality.")


if __name__ == "__main__":
    main()
