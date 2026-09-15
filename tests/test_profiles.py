import hashlib
from dataclasses import replace

import pytest

from navid import config, prepare
from navid.profiles import PROFILES, configured_profile


def test_profile_default_and_strict_selection(monkeypatch):
    monkeypatch.delenv("REF2VA_NFE", raising=False)
    assert configured_profile() == PROFILES[8]
    for value in ("4", "8"):
        monkeypatch.setenv("REF2VA_NFE", value)
        profile = configured_profile()
        assert profile.nfe == int(value)
        assert profile.metadata()["scheduler_points"] == int(value) + 1
        assert profile.video_shift == 12 and profile.audio_shift == 3 and profile.lora_alpha == 8
        assert len(profile.adapter_sha256) == 64
    for value in ("", "5", "8.0", "04", "true"):
        monkeypatch.setenv("REF2VA_NFE", value)
        with pytest.raises(ValueError, match="REF2VA_NFE"):
            configured_profile()


@pytest.mark.parametrize("nfe", (4, 8))
def test_start_downloads_only_missing_selected_adapter(tmp_path, monkeypatch, nfe):
    import huggingface_hub
    import torch
    from safetensors.torch import save

    data = save({"block.lora_A.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
                 "block.lora_B.weight": torch.zeros(2, 2, dtype=torch.bfloat16)})
    profile = replace(PROFILES[nfe], adapter_sha256=hashlib.sha256(data).hexdigest())
    path = tmp_path / profile.adapter_name
    monkeypatch.setattr(config, "PROFILE", profile)
    monkeypatch.setattr(config, "ADAPTER", path)
    monkeypatch.setattr(config, "ADAPTER_NAME", profile.adapter_name)
    monkeypatch.setattr(config, "RUNTIME", tmp_path / "runtime")
    monkeypatch.delenv("ADAPTER_PATH", raising=False)
    model_checks, downloads = [], []
    monkeypatch.setattr(prepare, "validate_model", lambda p: model_checks.append(p))

    def download(repo_id, **kwargs):
        downloads.append((repo_id, kwargs))
        path.write_bytes(data)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    prepare.ensure()
    prepare.ensure()
    assert len(model_checks) == 2 and len(downloads) == 1
    repo, kwargs = downloads[0]
    assert repo == "lightx2v/Minimax-h3-Turbo"
    assert kwargs["revision"] == config.ADAPTER_REVISION
    assert kwargs["allow_patterns"] == [profile.adapter_name]
    assert config.read_json(config.RUNTIME / "checkpoints.json")["profile"] == profile.metadata()

    # A custom/renamed file with the same tensor structure must not bypass the
    # weight identity check. Keep the user's file intact on a failed upgrade.
    path.write_bytes(b"wrong or corrupted adapter")
    monkeypatch.setenv("ADAPTER_PATH", str(path))
    with pytest.raises(RuntimeError, match=f"REF2VA_NFE={nfe}"):
        prepare.ensure_adapter()
    assert path.read_bytes() == b"wrong or corrupted adapter"
    assert len(downloads) == 1


def test_missing_custom_adapter_does_not_download_over_it(tmp_path, monkeypatch):
    path = tmp_path / "custom.safetensors"
    monkeypatch.setenv("ADAPTER_PATH", str(path))
    monkeypatch.setattr(config, "ADAPTER", path)
    with pytest.raises(RuntimeError, match="ADAPTER_PATH does not exist"):
        prepare.ensure_adapter()
    assert not path.exists()
