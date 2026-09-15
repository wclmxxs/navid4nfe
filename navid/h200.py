"""Hopper-specific options; four-step sampling and Ref2VA weights stay upstream."""
from __future__ import annotations

from .startup import vae_compile_enabled


def configure() -> None:
    from h3_runtime import vae_parallel

    original_install = vae_parallel.install

    def install(vae, *args, **kwargs):
        # The upstream engine hardcodes compile_mode='default'. Override it here
        # so the H200 deployment setting actually controls compilation.
        kwargs["compile_mode"] = "default" if vae_compile_enabled() else ""
        return original_install(vae, *args, **kwargs)

    vae_parallel.install = install
