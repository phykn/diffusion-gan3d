import copy
from pathlib import Path

import pytest

import src.config.train as config_module
from src.build.data import build_augmentation
from src.config.data import validate_sr_source
from src.config.files import load_yaml, save_yaml
from src.config.train import (
    load_train_config,
    normalize_train_config,
    validate_sr_config,
)
from src.train.run.sr import run_sr_train


@pytest.mark.parametrize("stage", ["low_res", "sr"])
@pytest.mark.parametrize("height", [False, True])
def test_default_presets_resolve_height_safe_augmentation(stage, height):
    raw = load_yaml(f"config/train/{stage}.yaml")
    raw["data"] = load_yaml(raw["data"])
    raw.setdefault("conditioning", {})["height_enabled"] = height
    cfg = normalize_train_config(raw, stage)
    augment = build_augmentation(cfg)
    assert set(augment.plane_transforms[0]) == set(range(8))
    for plane in (1, 2):
        assert set(augment.plane_transforms[plane]) == (
            {0, 4} if height else set(range(8))
        )
    assert normalize_train_config(cfg, stage) == cfg


def test_auto_augmentation_does_not_override_explicit_policy():
    with pytest.raises(ValueError, match="or explicit planes"):
        normalize_train_config(
            {"data": {}, "augmentation": {"auto_planes": True, "planes": {}}}
        )


def test_data_selection_and_snapshot_are_independent_of_cwd(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "config/data").mkdir(parents=True)
    data = {
        "domains": {0: {"xy": ["images"]}},
        "num_phases": 3,
        "crop_size": 256,
        "lo_res_size": 64,
    }
    save_yaml(root / "config/data/selected.yaml", data)
    preset = tmp_path / "experiment.yaml"
    save_yaml(preset, {"stage": "low_res", "data": "missing.yaml"})
    monkeypatch.setattr(config_module, "PROJECT_ROOT", root)
    monkeypatch.chdir(tmp_path)
    cfg = load_train_config(preset, data="config/data/selected.yaml")
    assert cfg["data"]["domains"][0]["xy"] == [str((root / "images").resolve())]
    assert "scale_factor" not in cfg["data"]
    snapshot = tmp_path / "train.yaml"
    save_yaml(snapshot, cfg)
    (root / "config/data/selected.yaml").unlink()
    assert load_train_config(snapshot) == cfg
    assert load_yaml(preset)["data"] == "missing.yaml"


@pytest.mark.parametrize(
    "old",
    [{"anchor": {}}, {"data": {"num_phase": 3}}, {"model": {"grad_checkpoint": True}}],
)
def test_old_keys_are_rejected_without_mutating_input(old):
    before = copy.deepcopy(old)
    with pytest.raises(ValueError, match="unknown training setting"):
        normalize_train_config(old)
    assert old == before


@pytest.mark.parametrize("stage", ["low_res", "sr"])
def test_omitted_critic_lr_follows_generator_but_explicit_value_wins(stage):
    raw = load_train_config(f"tests/fixtures/config/train/{stage}.yaml", stage)
    raw["optim"].pop("critic_lr", None)
    raw["optim"]["generator_lr"] = 0.0003
    cfg = normalize_train_config(raw, stage)
    assert cfg["optim"]["critic_lr"] == 0.0003
    assert "critic_lr" not in raw["optim"]
    raw["optim"]["critic_lr"] = 0.0002
    assert normalize_train_config(raw, stage)["optim"]["critic_lr"] == 0.0002


@pytest.mark.parametrize("stage", ["low_res", "sr"])
def test_resolved_snapshot_survives_default_and_learning_rate_changes(
    tmp_path, monkeypatch, stage
):
    cfg = load_train_config(f"tests/fixtures/config/train/{stage}.yaml", stage)
    snapshot = tmp_path / "saved.yaml"
    save_yaml(snapshot, cfg)
    monkeypatch.setitem(config_module.TRAIN_DEFAULTS, "optim.ema_decay", 0.5)
    key = (
        "model.generator.embedding_channels"
        if stage == "low_res"
        else "model.generator.latent_channels"
    )
    monkeypatch.setitem(config_module.STAGE_DEFAULTS[stage], key, 32)
    assert load_train_config(snapshot, stage) == cfg
    edited = load_yaml(snapshot)
    edited["optim"]["generator_lr"] *= 2
    assert (
        normalize_train_config(edited, stage)["optim"]["critic_lr"]
        == cfg["optim"]["critic_lr"]
    )


def test_explicit_disabled_options_and_defaults_are_not_shared():
    cfg = normalize_train_config(
        {
            "model": {"gradient_checkpointing": False},
            "train": {"mixed_precision": False},
            "conditioning": {"anchor": {"ramp_steps": 0}},
            "optim": {"ema_decay": 0.0},
        }
    )
    assert cfg["model"]["gradient_checkpointing"] is False
    assert cfg["train"]["mixed_precision"] is False
    assert cfg["conditioning"]["anchor"]["ramp_steps"] == 0
    assert cfg["optim"]["ema_decay"] == 0.0
    original_betas = cfg["optim"]["adam_betas"].copy()
    cfg["optim"]["adam_betas"][0] += 0.1
    assert normalize_train_config({})["optim"]["adam_betas"] == original_betas


def test_conflicting_old_and_new_keys_fail_instead_of_overriding():
    with pytest.raises(ValueError, match="unknown training setting"):
        normalize_train_config(
            {"data": {"batch_size": 8}, "train": {"real_batch_size": 4}}
        )


@pytest.mark.parametrize(
    "change, message",
    [
        ({"crop_size": 256}, "resolution"),
        ({"lo_res_size": 32}, "resolution"),
        ({"scale_factor": 1.5}, "unknown training setting"),
        ({"num_phases": 3}, "num_phases"),
        ({"domains": {0: {"xy": ["data"]}, 1: {"xy": ["data"]}}}, "domain IDs"),
    ],
)
def test_sr_rejects_incompatible_data_before_creating_a_run(tmp_path, change, message):
    base = tmp_path / "base"
    base.mkdir()
    low = load_train_config("tests/fixtures/config/train/low_res.yaml")
    save_yaml(base / "train.yaml", low)
    cfg = load_train_config("tests/fixtures/config/train/sr.yaml", "sr")
    cfg["data"].update(change)
    preset = tmp_path / "sr.yaml"
    save_yaml(preset, cfg)
    run = tmp_path / "run"
    # No weight file exists: validation must fail before loading/generating volumes.
    with pytest.raises(ValueError, match=message):
        run_sr_train(base_weights=base, config=preset, run_dir=run)
    assert not run.exists()


def test_sr_can_select_other_image_paths_with_the_same_contract():
    data = load_train_config("tests/fixtures/config/train/low_res.yaml")["data"]
    other = copy.deepcopy(data)
    other["domains"] = {0: {"xy": ["new/images"]}}
    assert validate_sr_source(other, data) is other


@pytest.mark.parametrize("field", ["data", "lr_bank"])
def test_new_sr_preset_requires_explicit_data_and_bank_guidance(tmp_path, field):
    cfg = load_yaml("tests/fixtures/config/train/sr.yaml")
    del cfg[field]
    path = tmp_path / "sr.yaml"
    save_yaml(path, cfg)
    with pytest.raises(ValueError, match="data|guidance"):
        load_train_config(path, "sr")


def test_stage_mismatch_is_rejected():
    with pytest.raises(ValueError, match="expected stage"):
        load_train_config("tests/fixtures/config/train/sr.yaml", "low_res")


@pytest.mark.parametrize("weights", [None, "", "  ", 42, True])
def test_sr_requires_source_weights_before_creating_a_run(tmp_path, weights):
    cfg = load_train_config("tests/fixtures/config/train/sr.yaml", "sr")
    cfg["source"] = {"weights": weights}
    preset = tmp_path / "sr.yaml"
    save_yaml(preset, cfg)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="source.weights.*--base-weights"):
        run_sr_train(config=preset, run_dir=output)
    assert not output.exists()


def test_sr_resume_rejects_an_explicit_lr_source_override(tmp_path):
    with pytest.raises(ValueError, match="base_weights cannot change on resume"):
        run_sr_train(
            base_weights=tmp_path / "generator.pt", resume=tmp_path / "last.pt"
        )


@pytest.mark.parametrize("stage", ["low_res", "sr"])
@pytest.mark.parametrize("nickname", [None, "", "   ", "  실험_A  "])
def test_nickname_is_optional_and_normalized(stage, nickname):
    cfg = normalize_train_config({"nickname": nickname}, stage)
    assert cfg["nickname"] == (nickname or "").strip()
    assert normalize_train_config({}, stage)["nickname"] == ""


@pytest.mark.parametrize(
    "nickname", [42, True, "../outside", "a\\b", "a:b", "x*", "x\nq", "end."]
)
def test_nickname_rejects_invalid_folder_components(nickname):
    with pytest.raises(ValueError, match="nickname"):
        normalize_train_config({"nickname": nickname})


@pytest.mark.parametrize("archive", [0, -1, True, 1.5])
def test_sr_archive_interval_rejects_invalid_values(archive):
    cfg = load_train_config("tests/fixtures/config/train/sr.yaml", "sr")
    cfg["train"]["archive_every_steps"] = archive
    with pytest.raises(ValueError, match="archive_every_steps"):
        validate_sr_config(cfg)


def test_sr_validation_returns_resolved_config_without_mutating_input():
    cfg = load_train_config("tests/fixtures/config/train/sr.yaml", "sr")
    del cfg["optim"]["critic_lr"]
    del cfg["conditioning"]["height_enabled"]
    original = copy.deepcopy(cfg)

    validated = validate_sr_config(cfg)

    assert cfg == original
    assert validated["optim"]["critic_lr"] == cfg["optim"]["generator_lr"]
    assert validated["conditioning"]["height_enabled"] is False


def test_saved_yaml_uses_inline_lists_and_separates_groups(tmp_path):
    cfg = {"model": {"channels": [4, 8]}, "data": {"paths": [Path("images")]}}
    path = tmp_path / "config.yaml"
    save_yaml(path, cfg)
    content = path.read_text()
    assert "channels: [4, 8]\n\ndata:" in content
    assert "paths: [images]" in content
    assert load_yaml(path)["model"]["channels"] == [4, 8]


@pytest.mark.parametrize(
    "section,key",
    [
        ("train", "seed"),
        ("train", "stability_version"),
        ("optim", "generatr_lr"),
        ("data", "input_size"),
    ],
)
def test_obsolete_and_unknown_config_keys_fail_with_full_path(section, key):
    with pytest.raises(ValueError, match=rf"{section}\.{key}"):
        normalize_train_config({section: {key: 1}})
