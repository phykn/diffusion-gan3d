"""Scores and TensorBoard tags against the v1 reference."""

import re
from pathlib import Path

import pytest
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.config.schema import update
from src.train.run import train_stage
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


def test_lr_tags_cover_v1(tiny_configs, tmp_path):
    config = update(tiny_configs["lr"], "train.total_steps", 20)
    run_dir = train_stage(config, "cpu", tmp_path / "lr")
    expected = normalize(yaml.safe_load((PARITY / "tags_lr.yaml").read_text()), "lr")
    missing = expected - normalize(run_tags(run_dir), "lr")
    assert not missing, sorted(missing)


def test_sr_tags_cover_v1(tiny_configs, tiny_lr_run, tmp_path):
    config = update(tiny_configs["sr"], "train.total_steps", 20)
    config = update(config, "lr_bank.samples_per_domain", 2)
    run_dir = train_stage(config, "cpu", tmp_path / "sr", lr_dir=tiny_lr_run)
    expected = normalize(yaml.safe_load((PARITY / "tags_sr.yaml").read_text()), "sr")
    missing = expected - normalize(run_tags(run_dir), "sr")
    assert not missing, sorted(missing)
