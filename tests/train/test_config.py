from pathlib import Path

import pytest

import src.config as config_module
from src.config import (
    GenerationSettings,
    find_train_config,
    get_schedule_steps,
    load_generation_settings,
    load_train_config,
    load_yaml,
    save_yaml,
)

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


def test_generation_settings_support_missing_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_module, "load_yaml", lambda path: {})

    assert load_generation_settings() == GenerationSettings()


def test_repository_generation_config_has_expected_defaults() -> None:
    assert load_generation_settings() == GenerationSettings(0.9, 0.8, 8)


def test_repository_training_config_uses_soft_anchor_and_connectivity() -> None:
    cfg = load_train_config(ROOT / "config/train/low_res.yaml")
    assert set(cfg) == {
        "stage",
        "data",
        "model",
        "augmentation",
        "conditioning",
        "loss",
        "optim",
        "train",
    }
    assert cfg["conditioning"]["domain_keep_probability"] == 0.8
    assert cfg["model"]["generator"]["anchor_multiscale_input"] is True
    assert cfg["conditioning"]["anchor"]["borrowed_plane_probability"] == 0.20
    assert cfg["loss"]["anchor_pixel_weight"] == 0.05
    assert cfg["train"]["initial_weights"] is None
    assert "batch_size" not in cfg["data"]


@pytest.mark.parametrize(
    "settings",
    (
        {"overlap": -1},
        {"margin": True},
        {"anchor_strength": -0.1},
        {"anchor_strength": 1.1},
        {"anchor_spread": 0.0},
        {"anchor_spread": float("nan")},
        {"blocks": [2, 2, 2]},
    ),
)
def test_generation_settings_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    settings: dict,
) -> None:
    monkeypatch.setattr(config_module, "load_yaml", lambda path: settings)

    with pytest.raises(ValueError):
        load_generation_settings()


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


def test_stage1_training_has_no_measured_3d_target() -> None:
    paths = [
        ROOT / "config/train/low_res.yaml",
        ROOT / "run_train.py",
        ROOT / "src" / "train" / "trainer.py",
        ROOT / "src" / "data" / "real.py",
        ROOT / "src" / "data" / "resolution.py",
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
