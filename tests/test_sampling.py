"""Exercise the engine's step dispatch against the pinned, real CPU schedulers."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers import MiniMaxH3Scheduler

from navid.profiles import PROFILES

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor/sol_h3"))
from h3_runtime.engine import DURATION_FRAMES, MiniMaxH3Inference


@pytest.mark.parametrize("nfe", (4, 8))
def test_engine_dispatches_complete_video_and_audio_schedules(monkeypatch, nfe):
    profile = PROFILES[nfe]
    engine = MiniMaxH3Inference.__new__(MiniMaxH3Inference)
    engine.inference_nfe = nfe
    engine.task, engine.rank, engine.world_size = "ref2va", 0, 1
    engine.device = torch.device("cpu")
    engine._adaln_checked = False
    engine.transformer = SimpleNamespace(transformer_blocks=[SimpleNamespace(
        adaln_proj=SimpleNamespace(table=torch.zeros(3, nfe, 12, 6)))])
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    calls = []

    def pipeline(**request):
        assert request["references"] == ["identity-reference"] and "image" not in request
        assert request["num_frames"] == DURATION_FRAMES[4]
        schedules = [MiniMaxH3Scheduler(shift=s) for s in (profile.video_shift, profile.audio_shift)]
        for scheduler in schedules:
            scheduler.set_timesteps(request["num_inference_steps"], device="cpu")
            assert scheduler.num_inference_steps == nfe
            assert scheduler.sigmas[0] == 1 and scheduler.sigmas[-1] == 0
            assert len(scheduler.sigmas) == nfe + 1
        # A constant data-ward velocity integrates to one over the complete
        # sigma interval for either modality. Missing a step leaves it short.
        samples = [torch.zeros(2), torch.zeros(2)]
        for step, timesteps in enumerate(zip(*(s.timesteps for s in schedules))):
            calls.append(step)
            for i, (scheduler, timestep) in enumerate(zip(schedules, timesteps)):
                samples[i] = scheduler.step(torch.ones(2), timestep, samples[i]).prev_sample
        for sample in samples:
            torch.testing.assert_close(sample, torch.ones(2))
        assert not torch.equal(schedules[0].timesteps, schedules[1].timesteps)
        return {"videos": torch.zeros(1, request["num_frames"], 3, 32, 32),
                "audio": torch.zeros(1, 2, 150000), "sampling_rate": 32000}

    engine.pipe = pipeline
    media = engine.generate("A person waves", duration=4, seed=42,
                            references=["identity-reference"], width=32, height=32)
    assert calls == list(range(nfe))
    assert media.video.shape == (96, 3, 32, 32)
    assert media.audio.shape == (2, 128000)
    assert engine._adaln_checked

    # A table from the other profile is not valid merely because its other
    # dimensions agree; do not reuse a four-step table for an eight-step run.
    engine._adaln_checked = False
    engine.transformer.transformer_blocks[0].adaln_proj.table = torch.zeros(3, 12 - nfe, 12, 6)
    with pytest.raises(RuntimeError, match="invalid AdaLN cache"):
        engine._check_adaln()


@pytest.mark.parametrize("nfe", (4, 8))
def test_runtime_refuses_to_report_incomplete_sampling(nfe):
    from navid.runtime import RequestRuntime

    runtime = RequestRuntime.__new__(RequestRuntime)
    runtime.engine = SimpleNamespace(inference_nfe=nfe)
    runtime.step = nfe - 2
    with pytest.raises(RuntimeError, match=f"Expected {nfe} DiT forwards"):
        runtime.finish()
