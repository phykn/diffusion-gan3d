import pytest
import torch
import yaml

from src.api import LowResolutionAPI
from src.export import FORMAT, VERSION, read_export


def test_export_has_the_spec_keys(tiny_lr_run):
    payload = read_export(tiny_lr_run, "lr")
    assert set(payload) == {"format", "version", "stage", "step", "config", "model"}
    assert (payload["format"], payload["version"], payload["stage"]) == (
        FORMAT,
        VERSION,
        "lr",
    )
    scores = yaml.safe_load((tiny_lr_run / "selection" / "scores.yaml").read_text())
    assert payload["step"] == scores["best_step"]
    assert all(value.device.type == "cpu" for value in payload["model"].values())


def test_plain_state_dict_is_not_an_export(tiny_lr_run, tmp_path):
    state = read_export(tiny_lr_run, "lr")["model"]
    path = tmp_path / "generator.pt"
    torch.save(state, path)
    with pytest.raises(ValueError, match="not a v3 LR export"):
        LowResolutionAPI(path, device="cpu")
    with pytest.raises(ValueError, match="not a v3 SR export"):
        read_export(tiny_lr_run, "sr")
