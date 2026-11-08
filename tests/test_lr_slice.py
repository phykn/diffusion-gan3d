"""The thinnest LR path, train -> export -> generate."""

import pytest
import torch
import yaml

import train
from src.api import LowResolutionAPI
from src.export import FORMAT, VERSION, read_export
from src.train.run import DONE


def test_train_cli_writes_the_lr_run(tiny_sample, tmp_path):
    train.main(
        [
            str(tiny_sample),
            "--run-dir",
            str(tmp_path / "cli"),
            "--lr-only",
            "--device",
            "cpu",
        ]
    )
    run_dir = tmp_path / "cli-lr"
    for name in ("train.yaml", "generator.pt", DONE, "tensorboard"):
        assert (run_dir / name).exists(), name
    events = list((run_dir / "tensorboard").glob("events.*"))
    assert events


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
    with pytest.raises(ValueError, match="not a v2 LR export"):
        LowResolutionAPI(path, device="cpu")
    with pytest.raises(ValueError, match="not a v2 SR export"):
        read_export(tiny_lr_run, "sr")


def test_generate_labels(tiny_lr_run):
    api = LowResolutionAPI(tiny_lr_run, device="cpu")
    volume = api.generate(size=24, seed=0)
    assert volume.shape == (24, 24, 24)
    assert volume.dtype == torch.uint8
    assert set(volume.unique().tolist()) <= {0, 1}
    assert torch.equal(volume, api.generate(shape=(24, 24, 24), seed=0))
    probs = api.generate_probs(size=16, seed=0)
    assert probs.shape == (2, 16, 16, 16)
    torch.testing.assert_close(probs.sum(0), torch.ones(16, 16, 16))


def test_generate_options(tiny_lr_run):
    api = LowResolutionAPI(tiny_lr_run, device="cpu")
    assert api.generate(blocks=(1, 1, 2), overlap=4, seed=1).shape == (16, 16, 24)
    with pytest.raises(ValueError, match="only one"):
        api.generate(size=16, shape=(16, 16, 16))
    with pytest.raises(ValueError, match="at least the patch"):
        api.generate(shape=(8, 16, 16))
    with pytest.raises(ValueError, match="storage"):
        api.generate(storage="disk")
    with pytest.raises(ValueError, match="original crop"):
        api.prepare_image(torch.zeros(16, 16, dtype=torch.long))
    assert api.prepare_image(torch.zeros(32, 32, dtype=torch.long)).shape == (2, 16, 16)
