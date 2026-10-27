import copy
from pathlib import Path

import pytest
import torch

import src.config.train as config_module
from src.build.data import build_augmentation
from src.build.trainer import _build_settings
from src.config.data import validate_sr_source
from src.config.files import load_yaml, save_yaml
from src.config.train import (
    load_train_config,
    normalize_train_config,
    validate_sr_config,
    validate_training_settings,
)
from src.train.run.sr import run_sr_train


@pytest.mark.parametrize(
    "stage,path,field,default",
    [
        (
            "low_res",
            "conditioning.cfg_dropout_probability",
            "cfg_dropout_probability",
            0.1,
        ),
        ("low_res", "loss.volume_fraction_weight", "loss.vf_weight", 0.0),
        ("low_res", "loss.spatial_profile_weight", "loss.profile_weight", 1.0),
        (
            "low_res",
            "conditioning.spatial_profile.num_bins",
            "loss.profile_bins",
            16,
        ),
        ("low_res", "loss.real_transition.max_gap", "real_transition_max_gap", 1),
        (
            "low_res",
            "loss.real_transition.weight",
            "loss.real_transition_weight",
            0.1,
        ),
        ("sr", "loss.downsample_consistency_weight", "loss.consistency_weight", 0.0),
        ("sr", "loss.downsample_mse_tolerance", "loss.consistency_tolerance", 0.005),
        (
            "sr",
            "conditioning.coarse_corruption_probability",
            "coarse_corruption_probability",
            0.5,
        ),
        (
            "sr",
            "conditioning.coarse_corruption_strength",
            "coarse_corruption_strength",
            0.2,
        ),
    ],
)
def test_builder_uses_canonical_defaults_and_preserves_resolved_values(
    monkeypatch, stage, path, field, default
):
    raw = load_yaml(f"tests/fixtures/config/train/{stage}.yaml")
    section = raw
    *parents, key = path.split(".")
    for parent in parents:
        section = section.setdefault(parent, {})
    section.pop(key, None)
    before = copy.deepcopy(raw)

    def read(cfg):
        settings = _build_settings(cfg, torch.device("cpu"))
        for name in field.split("."):
            settings = getattr(settings, name)
        return settings

    resolved = normalize_train_config(raw, stage)
    assert read(resolved) == default
    changed = 3 if type(default) is int else 0.13
    monkeypatch.setitem(config_module.STAGE_DEFAULTS[stage], path, changed)
    assert read(normalize_train_config(raw, stage)) == changed
    # A saved, resolved config is unaffected by later default changes.
    assert read(normalize_train_config(resolved, stage)) == default
    section[key] = default
    assert read(normalize_train_config(raw, stage)) == default
    section.pop(key)
    assert raw == before


@pytest.mark.parametrize("stage", ("low_res", "sr"))
def test_runtime_inactive_sections_do_not_mutate_or_broaden_saved_config(stage):
    cfg = load_train_config(f"tests/fixtures/config/train/{stage}.yaml", stage)
    before = copy.deepcopy(cfg)
    settings = _build_settings(cfg, torch.device("cpu"))
    conditioning, loss = config_module.get_trainer_sections(cfg)
    assert cfg == before
    assert normalize_train_config(cfg, stage) == cfg
    if stage == "sr":
        assert "cfg_dropout_probability" not in cfg["conditioning"]
        assert "real_transition" not in cfg["loss"]
        assert settings.cfg_dropout_probability == 0
        assert settings.loss.real_transition_weight == 0
        assert (
            settings.real_transition_start_step
            == settings.real_transition_ramp_steps
            == 0
        )
        conditioning["cfg_dropout_probability"] = 0.9
        sections = config_module.get_trainer_sections(cfg)
        assert sections[0]["cfg_dropout_probability"] == 0
    else:
        assert "downsample_consistency_weight" not in cfg["loss"]
        assert settings.loss.consistency_weight == 0
        assert settings.coarse_corruption_probability == 0
        loss["downsample_consistency_weight"] = 123
        assert (
            config_module.get_trainer_sections(cfg)[1]["downsample_consistency_weight"]
            == 0
        )
    assert cfg == before


@pytest.mark.parametrize(
    "stage,path",
    [
        ("low_res", "conditioning.cfg_dropout_probability"),
        ("low_res", "loss.real_transition.weight"),
        ("sr", "conditioning.coarse_corruption_probability"),
    ],
)
def test_builder_requires_resolved_active_options(stage, path):
    cfg = load_train_config(f"tests/fixtures/config/train/{stage}.yaml", stage)
    section = cfg
    *parents, key = path.split(".")
    for parent in parents:
        section = section[parent]
    del section[key]
    with pytest.raises(KeyError, match=key):
        _build_settings(cfg, torch.device("cpu"))


@pytest.mark.parametrize("stage", ["low_res", "sr"])
@pytest.mark.parametrize("weight", [0, 0.37])
def test_explicit_group_statistics_settings_survive_normalization(stage, weight):
    settings = {
        "weight": weight,
        "max_gap": 3,
        "tolerance": 0.07,
        "start_step": 13,
        "ramp_steps": 19,
    }
    raw = {"loss": {"group_statistics": settings}}
    before = copy.deepcopy(raw)
    cfg = normalize_train_config(raw, stage)
    for key, value in settings.items():
        assert cfg["loss"]["group_statistics"][key] == value
    assert raw == before
    assert normalize_train_config(cfg, stage) == cfg


@pytest.mark.parametrize("stage", ["low_res", "sr"])
@pytest.mark.parametrize(
    "key,value",
    [
        ("weight", -1),
        ("weight", True),
        ("weight", float("nan")),
        ("tolerance", -1),
        ("tolerance", 1.1),
        ("tolerance", float("inf")),
        ("max_gap", 0),
        ("max_gap", True),
        ("max_gap", 1.5),
        ("start_step", -1),
        ("ramp_steps", -1),
        ("ramp_steps", True),
    ],
)
def test_group_statistics_settings_are_validated(stage, key, value):
    with pytest.raises(ValueError, match=f"group_statistics.{key}"):
        normalize_train_config({"loss": {"group_statistics": {key: value}}}, stage)


@pytest.mark.parametrize("gap", (0, -1, 0.5, True))
def test_transition_gap_is_validated(gap):
    with pytest.raises(ValueError, match="real_transition.max_gap"):
        normalize_train_config({"loss": {"real_transition": {"max_gap": gap}}})


@pytest.mark.parametrize(
    "key,value",
    [("weight", 0.47), ("max_gap", 2)],
)
def test_lr_transition_settings_are_selectable_and_sr_rejects_them(key, value):
    raw = {"loss": {"real_transition": {key: value}}}
    assert normalize_train_config(raw)["loss"]["real_transition"][key] == value
    with pytest.raises(ValueError, match="real_transition"):
        normalize_train_config(raw, "sr")


@pytest.mark.parametrize("value", [-1, True, float("nan")])
def test_real_transition_weight_is_validated(value):
    with pytest.raises(ValueError, match="real_transition.weight"):
        normalize_train_config({"loss": {"real_transition": {"weight": value}}})


def test_unknown_real_transition_setting_is_rejected():
    with pytest.raises(ValueError, match="loss.real_transition.unused_weight"):
        normalize_train_config({"loss": {"real_transition": {"unused_weight": 0}}})


@pytest.mark.parametrize("stage", ["low_res", "sr"])
@pytest.mark.parametrize(
    ("height", "groups", "side"),
    [
        (False, [["xy", "xz", "yz"]], set(range(8))),
        # Separate xy and side critics mean z differs; z stays unrotated.
        (False, [["xy"], ["xz", "yz"]], {0, 2, 4, 6}),
        (True, [["xy", "xz", "yz"]], {0, 4}),
    ],
)
def test_auto_augmentation_keeps_z_where_it_is_distinct(stage, height, groups, side):
    raw = {
        "data": {"domains": {0: {plane: ["images"] for plane in ("xy", "xz", "yz")}}},
        "model": {"critic": {"plane_groups": groups}},
        "augmentation": {"probability": 0.7, "auto_planes": True},
        "conditioning": {"height_enabled": height},
    }
    cfg = normalize_train_config(raw, stage)
    augment = build_augmentation(cfg)
    assert set(augment.plane_transforms[0]) == set(range(8))
    for plane in (1, 2):
        assert set(augment.plane_transforms[plane]) == side
    assert normalize_train_config(cfg, stage) == cfg


def test_real_only_augmentation_reaches_the_critic_augment():
    raw = {
        "data": {"domains": {0: {plane: ["images"] for plane in ("xy", "xz", "yz")}}},
        "augmentation": {"auto_planes": True, "real_only": True},
        "conditioning": {"height_enabled": True},
    }
    cfg = normalize_train_config(raw, "low_res")
    assert build_augmentation(cfg).real_only
    assert normalize_train_config(cfg, "low_res") == cfg


def test_auto_augmentation_does_not_override_explicit_policy():
    with pytest.raises(ValueError, match="or explicit planes"):
        normalize_train_config(
            {"data": {}, "augmentation": {"auto_planes": True, "planes": {}}}
        )


def test_embedded_data_folders_resolve_against_the_project_root(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    preset = tmp_path / "experiment.yaml"
    data = {
        "domains": {0: {"xy": ["images"]}},
        "num_phases": 3,
        "crop_size": 256,
        "lo_res_size": 64,
    }
    save_yaml(preset, {"stage": "low_res", "data": data})
    monkeypatch.setattr("src.config.files.PROJECT_ROOT", root)
    monkeypatch.chdir(tmp_path)
    cfg = load_train_config(preset)
    assert cfg["data"]["domains"][0]["xy"] == [str((root / "images").resolve())]


@pytest.mark.parametrize(
    "unknown",
    [{"unused": {}}, {"data": {"num_phase": 3}}, {"model": {"grad_checkpoint": True}}],
)
def test_unknown_keys_are_rejected_without_mutating_input(unknown):
    before = copy.deepcopy(unknown)
    with pytest.raises(ValueError, match="unknown training setting"):
        normalize_train_config(unknown)
    assert unknown == before


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


def test_explicit_disabled_options_are_preserved_without_sharing_input():
    raw = {
        "model": {"gradient_checkpointing": False},
        "train": {"mixed_precision": False},
        "loss": {"real_transition": {"ramp_steps": 0}},
        "optim": {"ema_decay": 0.0, "adam_betas": [0.12, 0.78]},
    }
    before = copy.deepcopy(raw)
    cfg = normalize_train_config(raw)
    other = normalize_train_config(raw)
    assert cfg["model"]["gradient_checkpointing"] is False
    assert cfg["train"]["mixed_precision"] is False
    assert cfg["loss"]["real_transition"]["ramp_steps"] == 0
    assert cfg["optim"]["ema_decay"] == 0.0
    cfg["optim"]["adam_betas"][0] += 0.1
    assert raw == before
    assert other["optim"]["adam_betas"] == raw["optim"]["adam_betas"]


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
    low["conditioning"]["height_enabled"] = False
    save_yaml(base / "train.yaml", low)
    (base / "generator.pt").write_bytes(b"not loaded")
    cfg = load_train_config("tests/fixtures/config/train/sr.yaml", "sr")
    cfg["conditioning"]["height_enabled"] = False
    cfg["data"].update(change)
    preset = tmp_path / "sr.yaml"
    save_yaml(preset, cfg)
    run = tmp_path / "run"
    # The weights are unreadable: validation must fail before loading them.
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
    with pytest.raises(ValueError, match="source.weights.*base_weights"):
        run_sr_train(config=preset, run_dir=output)
    assert not output.exists()


def test_sr_resume_rejects_an_explicit_lr_source_override(tmp_path):
    with pytest.raises(ValueError, match="base_weights cannot change on resume"):
        run_sr_train(
            base_weights=tmp_path / "generator.pt", resume=tmp_path / "training.pt"
        )


@pytest.mark.parametrize("stage", ["low_res", "sr"])
@pytest.mark.parametrize("nickname", [None, "", "   ", "  실험_A  "])
def test_nickname_is_optional_and_normalized(stage, nickname):
    cfg = normalize_train_config({"nickname": nickname}, stage)
    assert cfg["nickname"] == (nickname or "").strip()


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


@pytest.mark.parametrize(
    "name",
    [
        "total_steps",
        "volume_batch_size",
        "real_batch_size",
        "slice_pairs_per_plane",
        "weights_every_steps",
    ],
)
@pytest.mark.parametrize("value", [None, 0, True, 1.5])
def test_sr_requires_valid_training_values(name, value):
    cfg = load_train_config("tests/fixtures/config/train/sr.yaml", "sr")
    if value is None:
        del cfg["train"][name]
    else:
        cfg["train"][name] = value
    with pytest.raises(ValueError, match=rf"train\.{name}"):
        validate_sr_config(cfg)


def test_sr_validation_returns_resolved_config_without_mutating_input():
    cfg = load_train_config("tests/fixtures/config/train/sr.yaml", "sr")
    del cfg["optim"]["critic_lr"]
    cfg["conditioning"]["height_enabled"] = True
    original = copy.deepcopy(cfg)

    validated = validate_sr_config(cfg)

    assert cfg == original
    assert validated["optim"]["critic_lr"] == cfg["optim"]["generator_lr"]
    assert (
        validated["conditioning"]["height_enabled"]
        is cfg["conditioning"]["height_enabled"]
    )


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
        ("loss", "unused_weight"),
        ("data", "input_size"),
    ],
)
def test_obsolete_and_unknown_config_keys_fail_with_full_path(section, key):
    with pytest.raises(ValueError, match=rf"{section}\.{key}"):
        normalize_train_config({section: {key: 1}})


@pytest.mark.parametrize("stage", ["low_res", "sr"])
@pytest.mark.parametrize(
    "path,value",
    [
        ("data.num_phases", 257),
        ("data.num_phases", 0),
        ("data.num_phases", True),
        ("data.num_phases", 2.5),
        ("optim.ema_decay", 1.5),
        ("optim.ema_decay", 1),
        ("optim.ema_decay", -0.1),
        ("optim.ema_decay", float("nan")),
        ("optim.ema_decay", True),
        ("optim.generator_lr", float("inf")),
        ("optim.critic_lr", -0.1),
        ("optim.adam_betas", [0.5, 1]),
        ("optim.adam_betas", [0.5]),
        ("optim.adam_betas", [True, 0.9]),
        ("conditioning.domain_keep_probability", 1.1),
        ("conditioning.domain_keep_probability", float("nan")),
        ("loss.r1_weight", float("inf")),
        ("loss.r2_weight", True),
        ("augmentation.probability", -0.1),
        ("train.volume_batch_size", 0),
        ("train.real_batch_size", True),
        ("train.total_steps", 1.5),
        ("train.weights_every_steps", 0),
        ("train.archive_every_steps", -1),
        ("train.num_workers", -1),
        ("loss.r1_every_steps", 0),
        ("loss.r1_every_steps", 2.0),
        ("model.diffusion.beta_min", "1e-1"),
        ("model.diffusion.beta_max", 0.05),
    ],
)
def test_shared_training_values_fail_during_normalization(stage, path, value):
    cfg = {}
    target = cfg
    *sections, key = path.split(".")
    for section in sections:
        target = target.setdefault(section, {})
    target[key] = value
    with pytest.raises(ValueError, match=key):
        normalize_train_config(cfg, stage)


@pytest.mark.parametrize(
    "cfg,message",
    [
        ({"loss": {"volume_fraction_weight": -1}}, "volume_fraction_weight"),
        (
            {"loss": {"real_transition": {"weight": -1}}},
            "real_transition.weight",
        ),
        (
            {"conditioning": {"cfg_dropout_probability": 1.5}},
            "cfg_dropout_probability",
        ),
    ],
)
def test_lr_rejects_invalid_loss_and_sampling_probabilities(cfg, message):
    with pytest.raises(ValueError, match=message):
        normalize_train_config(cfg)


@pytest.mark.parametrize("phases", [1, 256])
def test_supported_phase_and_probability_boundaries(phases):
    cfg = normalize_train_config(
        {
            "data": {"num_phases": phases},
            "optim": {"ema_decay": 0, "generator_lr": 0},
            "conditioning": {
                "domain_keep_probability": 1,
                "cfg_dropout_probability": 1,
            },
            "loss": {"volume_fraction_weight": 0},
            "train": {"num_workers": 0, "archive_every_steps": None},
        }
    )
    assert cfg["data"]["num_phases"] == phases
    assert normalize_train_config(cfg) == cfg


@pytest.mark.parametrize("stage", ["low_res", "sr"])
@pytest.mark.parametrize("path", ["loss.r1_weight", "train.total_steps"])
def test_missing_training_setting_names_its_key(tmp_path, stage, path):
    cfg = load_yaml(f"tests/fixtures/config/train/{stage}.yaml")
    section, key = path.split(".")
    cfg[section].pop(key, None)
    preset = tmp_path / "train.yaml"
    save_yaml(preset, cfg)
    cfg = load_train_config(preset, stage)
    with pytest.raises(ValueError, match=f"missing required setting.*{path}"):
        if stage == "sr":
            validate_sr_config(cfg)
        else:
            validate_training_settings(cfg)


@pytest.mark.parametrize("stage", ["low_res", "sr"])
def test_existing_run_directory_is_rejected_before_setup(tmp_path, monkeypatch, stage):
    import src.train.run.low_res as low_res_module

    def forbidden(*args, **kwargs):
        pytest.fail("setup ran before the run directory was checked")

    monkeypatch.setattr(low_res_module, "build_trainer", forbidden)
    monkeypatch.setattr(config_module, "load_yaml", forbidden)
    run = tmp_path / "run"
    run.mkdir()
    runner = run_sr_train if stage == "sr" else low_res_module.run_low_res_train
    with pytest.raises(FileExistsError, match="already exists"):
        runner(config=tmp_path / "unused.yaml", run_dir=run)


def test_numbers_read_as_text_explain_the_yaml_spelling():
    with pytest.raises(ValueError, match=r"got str '1e-4' \(write 1\.0e-4"):
        normalize_train_config({"optim": {"generator_lr": "1e-4"}})


@pytest.mark.parametrize(
    "section, key",
    [
        ("train", "mixed_precision"),
        ("model", "gradient_checkpointing"),
        ("augmentation", "real_only"),
    ],
)
def test_quoted_booleans_are_rejected(section, key):
    raw = {
        "data": {"domains": {0: {"xy": ["images"]}}},
        section: {key: "false"},
    }
    with pytest.raises(ValueError, match=f"{section}.{key} must be a boolean"):
        normalize_train_config(raw, "low_res")


@pytest.mark.parametrize("channels", [[], [8, 0], [8, "16"], 8])
def test_critic_channels_must_list_positive_integers(channels):
    raw = {
        "data": {"domains": {0: {"xy": ["images"]}}},
        "model": {"critic": {"channels": channels}},
    }
    with pytest.raises(ValueError, match="model.critic.channels"):
        normalize_train_config(raw, "low_res")


@pytest.mark.parametrize(
    ("height", "groups", "expected"),
    [
        (False, [["xy", "xz", "yz"]], (True, True)),
        (False, [["xy"], ["xz", "yz"]], (False, True)),
        (True, [["xy", "xz", "yz"]], (False, False)),
    ],
)
def test_sr_bank_augmentation_follows_the_critic_z_policy(height, groups, expected):
    cfg = load_train_config("tests/fixtures/config/train/sr.yaml", "sr")
    cfg["conditioning"]["height_enabled"] = height
    cfg["model"]["critic"]["plane_groups"] = groups
    cfg["augmentation"] = {"probability": 0.5, "auto_planes": True}
    settings = _build_settings(normalize_train_config(cfg, "sr"), torch.device("cpu"))
    assert (settings.volume_permute_z, settings.volume_flip_z) == expected


def test_sr_bank_guidance_is_required():
    cfg = load_train_config("tests/fixtures/config/train/sr.yaml", "sr")
    del cfg["lr_bank"]["guidance"]
    with pytest.raises(ValueError, match="lr_bank.guidance"):
        validate_sr_config(cfg)
