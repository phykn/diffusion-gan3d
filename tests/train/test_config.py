from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import src.config.generation as config_module
from src.config.files import find_saved_config, load_yaml, save_yaml
from src.config.generation import GenerationSettings, load_generation_settings
from src.config.train import get_schedule_steps, load_train_config

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


def test_find_saved_config_walks_from_numbered_checkpoint(tmp_path: Path) -> None:
    config = tmp_path / "run" / "train.yaml"
    config.parent.mkdir()
    config.write_text("train: {}\n", encoding="utf-8")
    weight = config.parent / "checkpoints" / "generator_00000100.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"weights")

    assert find_saved_config(weight) == config.resolve()


def test_saved_yaml_without_height_extents_recovers_them_from_images(
    tmp_path: Path,
) -> None:
    folders = {}
    for plane, shape in (("xy", (8, 8)), ("xz", (12, 8)), ("yz", (12, 8))):
        folder = tmp_path / plane
        folder.mkdir()
        Image.fromarray(np.zeros(shape, dtype=np.uint8)).save(folder / "0.png")
        folders[plane] = [str(folder)]
    data = {"domains": {0: folders}, "num_phases": 2, "crop_size": 8, "lo_res_size": 8}
    config = tmp_path / "train.yaml"
    save_yaml(config, {"data": data, "conditioning": {"height_enabled": True}})
    assert load_train_config(config)["data"]["height_extents"] == {0: 12}
    assert "height_extents" not in load_yaml(config)["data"]
    save_yaml(config, {"data": data})
    assert "height_extents" not in load_train_config(config)["data"]
    data["height_extents"] = {0: 10}
    save_yaml(config, {"data": data, "conditioning": {"height_enabled": True}})
    assert load_train_config(config)["data"]["height_extents"] == {0: 10}


def test_generation_settings_use_gen_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config_module,
        "load_yaml",
        lambda path: {
            "guidance": 1.5,
            "anchor_strength": 0.7,
            "overlap": 12,
        },
    )

    assert load_generation_settings() == GenerationSettings(1.5, 0.7, 12)


@pytest.mark.parametrize(
    "key,value", [("guidance", 2.3), ("anchor_strength", 0.45), ("overlap", 6)]
)
def test_partial_generation_settings_preserve_supplied_options(monkeypatch, key, value):
    monkeypatch.setattr(config_module, "load_yaml", lambda path: {key: value})
    settings = load_generation_settings()
    assert getattr(settings, key) == value


@pytest.mark.parametrize(
    "settings,match",
    (
        ({"overlap": -1}, "overlap"),
        ({"overlap": True}, "overlap"),
        ({"guidance": float("nan")}, "guidance"),
        ({"guidance": True}, "guidance"),
        ({"anchor_strength": -0.1}, "anchor_strength"),
        ({"anchor_strength": 1.1}, "anchor_strength"),
        ({"margin": 0}, "unknown generation setting: margin"),
    ),
)
def test_generation_settings_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    settings: dict,
    match: str,
) -> None:
    monkeypatch.setattr(config_module, "load_yaml", lambda path: settings)

    with pytest.raises(ValueError, match=match):
        load_generation_settings()


def test_yaml_config_remains_a_plain_mapping(tmp_path: Path) -> None:
    cfg = {
        "data": {"domains": {0: {"xy": ["data/0"]}}, "crop_size": 64},
        "train": {"volume_batch_size": 1},
    }
    path = tmp_path / "train.yaml"

    save_yaml(path, cfg)

    assert load_yaml(path) == cfg


def test_invalid_yaml_reports_the_source(tmp_path: Path) -> None:
    path = tmp_path / "train.yaml"
    path.write_text("data: [", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid YAML"):
        load_yaml(path)


def test_stage1_training_has_no_measured_3d_target() -> None:
    paths = [
        ROOT / "run_train.py",
        ROOT / "src" / "train" / "trainer.py",
        ROOT / "src" / "data" / "dataset.py",
        ROOT / "src" / "data" / "slice.py",
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


@pytest.mark.parametrize("text", ["- 64\n- 96\n", "[]", "false", "0", "''"])
def test_yaml_root_must_be_a_mapping(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(TypeError, match="mapping"):
        load_yaml(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1.5"])
def test_generation_guidance_requires_a_finite_number(monkeypatch, value):
    monkeypatch.setattr(config_module, "load_yaml", lambda path: {"guidance": value})
    with pytest.raises(ValueError, match="guidance"):
        load_generation_settings()
