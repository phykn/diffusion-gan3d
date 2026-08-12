from pathlib import Path

import pytest

from src.config import (
    GenerationSettings,
    find_train_config,
    get_schedule_steps,
    load_generation_settings,
)
from src.utils import load_yaml, save_yaml

ROOT = Path(__file__).resolve().parents[2]


def test_schedule_accepts_a_ramp_past_the_training_horizon() -> None:
    assert get_schedule_steps(
        {"start_step": 10, "ramp_steps": 20},
        "anchor",
    ) == (10, 20)
    assert get_schedule_steps(
        {"start_step": 10, "ramp_steps": 21},
        "anchor",
    ) == (10, 21)

    with pytest.raises(ValueError, match="non-negative integer"):
        get_schedule_steps(
            {"start_step": -1, "ramp_steps": 20},
            "anchor",
        )


def test_find_train_config_walks_from_numbered_checkpoint(tmp_path: Path) -> None:
    config = tmp_path / "run" / "train.yaml"
    config.parent.mkdir()
    config.write_text("train: {}\n", encoding="utf-8")
    weight = config.parent / "checkpoints" / "step_00000100" / "generator.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"weights")

    assert find_train_config(weight) == config.resolve()


def test_generation_settings_use_yaml_values(tmp_path: Path) -> None:
    weight = tmp_path / "run" / "checkpoints" / "1" / "generator.pt"
    weight.parent.mkdir(parents=True)
    weight.touch()
    (tmp_path / "run" / "train.yaml").write_text(
        "generation:\n  guidance_scale: 1.5\n  overlap: 12\n  crop_margin: 6\n",
        encoding="utf-8",
    )

    assert load_generation_settings(weight) == GenerationSettings(1.5, 12, 6)


def test_generation_settings_support_old_configs(tmp_path: Path) -> None:
    weight = tmp_path / "run" / "generator.pt"
    weight.parent.mkdir()
    weight.touch()
    (weight.parent / "train.yaml").write_text("train: {}\n", encoding="utf-8")

    assert load_generation_settings(weight) == GenerationSettings()


@pytest.mark.parametrize(
    "generation",
    (
        "[]",
        "{overlap: -1}",
        "{crop_margin: true}",
        "{guidance_scale: .inf}",
        "{blocks: [2, 2, 2]}",
    ),
)
def test_generation_settings_reject_invalid_values(
    tmp_path: Path,
    generation: str,
) -> None:
    weight = tmp_path / "run" / "generator.pt"
    weight.parent.mkdir()
    weight.touch()
    (weight.parent / "train.yaml").write_text(
        f"generation: {generation}\n",
        encoding="utf-8",
    )

    with pytest.raises((TypeError, ValueError)):
        load_generation_settings(weight)


def test_yaml_config_remains_a_plain_mapping(tmp_path: Path) -> None:
    cfg = {
        "data": {"domains": {0: {0: ["data/0"]}}, "input_size": 64},
        "train": {"volume_batch_size": 1},
    }
    path = tmp_path / "train.yaml"

    save_yaml(path, cfg)

    assert load_yaml(path) == cfg


def test_yaml_root_must_be_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "train.yaml"
    path.write_text("- 64\n- 96\n", encoding="utf-8")

    with pytest.raises(TypeError, match="mapping"):
        load_yaml(path)


def test_invalid_yaml_reports_the_source(tmp_path: Path) -> None:
    path = tmp_path / "train.yaml"
    path.write_text("data: [", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid YAML"):
        load_yaml(path)


def test_training_runtime_has_no_3d_reference_input() -> None:
    paths = [
        ROOT / "config" / "train.yaml",
        ROOT / "run_train.py",
        *sorted((ROOT / "src" / "train").glob("*.py")),
        ROOT / "src" / "dataset.py",
    ]
    forbidden = (
        "generated/volumes",
        "generated\\volumes",
        "reference_volume",
        "bulk_vf",
        "ground_truth_volume",
        "tifffile.imread",
        "tifffile.asarray",
    )

    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        for token in forbidden:
            assert token not in text, f"{token!r} found in {path.relative_to(ROOT)}"
