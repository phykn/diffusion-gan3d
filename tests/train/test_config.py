from pathlib import Path

import pytest

from src.config import find_train_config, get_schedule_steps, require_schema
from src.utils import load_yaml, save_yaml

ROOT = Path(__file__).resolve().parents[2]


def test_schema_and_schedule_contracts() -> None:
    require_schema({"schema_version": 2})
    assert get_schedule_steps(
        {"start_step": 10, "ramp_steps": 20},
        "anchor",
        total_steps=30,
    ) == (10, 20)

    with pytest.raises(ValueError, match="schema_version must be 2"):
        require_schema({})
    with pytest.raises(ValueError, match="must not exceed total steps"):
        get_schedule_steps(
            {"start_step": 10, "ramp_steps": 21},
            "anchor",
            total_steps=30,
        )


def test_find_train_config_walks_from_numbered_checkpoint(tmp_path: Path) -> None:
    config = tmp_path / "run" / "train.yaml"
    config.parent.mkdir()
    config.write_text("schema_version: 2\n", encoding="utf-8")
    weight = config.parent / "checkpoints" / "step_00000100" / "generator.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"weights")

    assert find_train_config(weight) == config.resolve()


def test_yaml_config_remains_a_plain_mapping(tmp_path: Path) -> None:
    cfg = {
        "data": {"folders": {0: ["data/0"]}, "input_size": 64},
        "train": {"volume_sizes": [64, 96]},
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
