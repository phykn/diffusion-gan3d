"""Scores and TensorBoard tags against the v1 reference."""

import re
from pathlib import Path

import pytest
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.config.schema import update
from src.export import read_export
from src.train.run import build_trainer
from src.train.selection import CheckpointSelection

PARITY = Path(__file__).parents[1] / "fixtures" / "parity"
# Tags v2 drops on purpose.
DROPPED = {
    "lr": {"sampling/domain"},
    "sr": {
        "sampling/domain",
        "loss/vf",
        "conditioning/vf_active",
        "conditioning/vf_fraction",
        "loss/real_transition_ramp",
    },
}


def normalize(tags, stage: str) -> set[str]:
    result = set()
    for tag in tags:
        tag = re.sub(r"^timestep/\d+/", "timestep/<t>/", tag)
        tag = re.sub(r"/t\d+/", "/t<t>/", tag)
        if stage == "sr":
            tag = re.sub(r"(^|/)0_", r"\1", tag)
        result.add(tag)
    return result - DROPPED[stage]


def run_tags(run_dir: Path) -> list[str]:
    events = EventAccumulator(str(run_dir / "tensorboard"))
    events.Reload()
    return events.Tags()["scalars"]


def test_selection_scores_match_v1():
    case = torch.load(PARITY / "selection.pt", weights_only=True)
    reference = {axis: {"images": case["images"]} for axis in range(3)}
    scores = CheckpointSelection(reference, 4, False).score(case["volumes"])
    assert scores.keys() == case["scores"].keys()
    for name, value in case["scores"].items():
        assert scores[name] == pytest.approx(value, abs=1e-6), name


def step_tags(config, lr=None) -> set[str]:
    """Tags of one transition-0 step with every ramped term on."""
    for term in ("group_statistics", "real_transition"):
        if hasattr(config.loss, term):
            config = update(config, f"loss.{term}.start_step", 0)
    trainer = build_trainer(config, torch.device("cpu"), lr=lr)
    return set(trainer.step(0, transition=0))


@pytest.mark.parametrize("stage", ["lr", "sr"])
def test_tags_cover_v1(stage, tiny_configs, tiny_lr_run, tiny_sr_run):
    run = {"lr": tiny_lr_run, "sr": tiny_sr_run}[stage]
    lr = read_export(tiny_lr_run, "lr") if stage == "sr" else None
    tags = set(run_tags(run)) | step_tags(tiny_configs[stage], lr)
    expected = yaml.safe_load((PARITY / f"tags_{stage}.yaml").read_text())
    missing = normalize(expected, stage) - normalize(tags, stage)
    assert not missing, sorted(missing)
