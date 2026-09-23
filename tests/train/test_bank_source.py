from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.build.model import build_denoiser
from src.config.files import load_yaml, save_yaml
from src.config.train import load_train_config, validate_sr_config
from src.train.run.bank import create_bank, refresh_bank
from src.train.run.source import (
    file_hash,
    load_frozen_source,
    read_source_config,
)
from src.train.run.sr import run_sr_train


@pytest.mark.parametrize(
    ("changed", "message"),
    [("weights", "weights changed"), ("configuration", "configuration changed")],
)
def test_create_bank_checks_frozen_source_hash_before_loading_generator(
    tmp_path, monkeypatch, changed, message
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    weights = source_dir / "generator.pt"
    weights.write_bytes(b"weights")
    config = source_dir / "train.yaml"
    config.write_text("stage: low_res\n", encoding="utf-8")
    cfg = {
        "source": {
            "weights": str(weights),
            "weights_sha256": file_hash(weights),
            "config_sha256": file_hash(config),
        },
        "conditioning": {"height_enabled": False},
    }
    if changed == "weights":
        weights.write_bytes(b"changed weights")
    else:
        config.write_text("stage: low_res\n# changed\n", encoding="utf-8")
    loaded = []
    monkeypatch.setattr(
        "src.train.run.bank.build_generator",
        lambda *args: loaded.append(args) or SimpleNamespace(),
    )

    with pytest.raises(ValueError, match=message):
        create_bank(cfg, torch.device("cpu"))
    assert loaded == []


def test_source_and_training_loaders_match_for_external_data_yaml(tmp_path):
    train_config = load_yaml("tests/fixtures/config/train/low_res.yaml")
    data_file = tmp_path / "data.yaml"
    save_yaml(data_file, train_config["data"])
    train_config["data"] = str(data_file)
    config_file = tmp_path / "train.yaml"
    save_yaml(config_file, train_config)

    assert read_source_config(config_file)[0] == load_train_config(config_file)


def test_source_loader_remaps_external_data_reference_before_reading(tmp_path):
    train_config = load_yaml("tests/fixtures/config/train/low_res.yaml")
    old_root = tmp_path / "old"
    moved_root = tmp_path / "moved"
    old_root.mkdir()
    moved_root.mkdir()
    old_data_file = old_root / "data.yaml"
    source_data = train_config["data"]
    source_data["domains"] = {
        0: {
            plane: [str(old_root / "images")]
            for plane in ("xy", "xz", "yz")
        }
    }
    save_yaml(old_data_file, source_data)
    train_config["data"] = str(old_data_file)
    config_file = tmp_path / "train.yaml"
    save_yaml(config_file, train_config)
    config_hash = file_hash(config_file)
    old_data_file.replace(moved_root / "data.yaml")

    cfg = read_source_config(
        config_file,
        [[(str(old_root), str(moved_root))]],
    )[0]

    assert cfg["data"]["domains"][0]["xy"] == [str(moved_root / "images")]
    assert file_hash(config_file) == config_hash


@pytest.mark.parametrize("old_root", ["C:/old", "/old"])
def test_source_loader_maps_foreign_external_data_paths(tmp_path, old_root):
    train_config = load_yaml("tests/fixtures/config/train/low_res.yaml")
    source_data = train_config["data"]
    source_data["domains"] = {
        0: {
            plane: [f"{old_root}/images"]
            for plane in ("xy", "xz", "yz")
        }
    }
    data_file = tmp_path / "data.yaml"
    save_yaml(data_file, source_data)
    train_config["data"] = f"{old_root}/data.yaml"
    config_file = tmp_path / "train.yaml"
    save_yaml(config_file, train_config)

    cfg = read_source_config(
        config_file,
        [[(old_root, str(tmp_path))]],
    )[0]

    assert cfg["data"]["domains"][0]["xy"] == [str(tmp_path / "images")]


def test_refresh_bank_loads_external_source_data_before_path_remapping(
    tmp_path, monkeypatch
):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    weights = source_dir / "generator.pt"
    weights.write_bytes(b"weights")
    source_data = load_yaml("tests/fixtures/config/train/low_res.yaml")["data"]
    old_root = "C:/old/images"
    source_data["domains"] = {
        0: {plane: [old_root] for plane in ("xy", "xz", "yz")}
    }
    source_data["split"] = {
        "validation_files": [f"{old_root}/heldout.png"],
        "validation_regions": {},
    }
    data_file = source_dir / "data.yaml"
    save_yaml(data_file, source_data)
    train_config = load_yaml("tests/fixtures/config/train/low_res.yaml")
    train_config["conditioning"]["height_enabled"] = True
    train_config["data"] = str(data_file)
    config_file = source_dir / "train.yaml"
    save_yaml(config_file, train_config)
    cfg = {
        "data": {"domains": {0: {"xy": [], "xz": [], "yz": []}}},
        "conditioning": {"height_enabled": True},
        "source": {
            "weights": str(weights),
            "weights_sha256": file_hash(weights),
            "config_sha256": file_hash(config_file),
        },
        "lr_bank": {"refresh_every_steps": 1, "guidance": 1},
    }
    bank = {
        "volumes": {0: torch.full((1, 2, 8, 8, 8), 0.5)},
        "height_origins": {0: torch.zeros(1)},
        "height_extents": {0: torch.ones(1)},
        "conditions": {0: [{}]},
    }
    captured = {}
    mapped_root = tmp_path / "mapped"
    monkeypatch.setattr(
        "src.train.run.bank.build_datasets",
        lambda base_cfg: captured.setdefault("cfg", base_cfg) or {},
    )
    monkeypatch.setattr(
        "src.train.run.bank.sample_bank_condition",
        lambda *args: {"height_origin": 0.0, "height_extent": 1.0},
    )
    monkeypatch.setattr(
        "src.train.run.bank.build_generator",
        lambda *args: SimpleNamespace(
            generate_probs=lambda **kwargs: torch.full((2, 8, 8, 8), 0.5)
        ),
    )

    refresh_bank(
        bank,
        cfg,
        1,
        torch.device("cpu"),
        [[("C:/old", str(mapped_root))]],
    )

    base_cfg = captured["cfg"]
    assert base_cfg["data"]["domains"][0]["xy"] == [str(mapped_root / "images")]
    assert base_cfg["data"]["split"]["validation_files"] == [
        str(mapped_root / "images/heldout.png")
    ]


def _write_external_source(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    cfg = load_train_config("tests/fixtures/config/train/low_res.yaml")
    cfg["data"].update(crop_size=8, lo_res_size=8, num_phases=2)
    cfg["model"]["generator"].update(
        channels=[4, 8], embedding_channels=8, latent_channels=4
    )
    cfg["model"]["diffusion"]["num_steps"] = 2
    cfg["model"]["gradient_checkpointing"] = False
    cfg["conditioning"]["height_enabled"] = False
    source_data = cfg["data"]
    data_file = source_dir / "data.yaml"
    save_yaml(data_file, source_data)
    cfg["data"] = str(data_file)
    config_file = source_dir / "train.yaml"
    save_yaml(config_file, cfg)
    weights = source_dir / "generator.pt"
    denoiser = build_denoiser(load_train_config(config_file), checkpointing=False)
    torch.save(denoiser.state_dict(), weights)
    source = {
        "weights": str(weights),
        "weights_sha256": file_hash(weights),
        "config_sha256": file_hash(config_file),
        "data_sha256": file_hash(data_file),
    }
    sr_cfg = {
        "data": source_data,
        "conditioning": {"height_enabled": False},
        "source": source,
        "lr_bank": {
            "samples_per_domain": 1,
            "refresh_every_steps": 1,
            "guidance": 1,
        },
    }
    return source_dir, data_file, sr_cfg


def test_initial_and_refresh_use_verified_moved_external_source(tmp_path):
    source_dir, data_file, cfg = _write_external_source(tmp_path)
    torch.set_num_threads(1)

    bank = create_bank(cfg, torch.device("cpu"))
    assert bank["volumes"][0].shape == (1, 2, 8, 8, 8)

    moved_dir = tmp_path / "moved"
    moved_dir.mkdir()
    data_file.replace(moved_dir / "data.yaml")
    path_maps = [[(str(source_dir), str(moved_dir))]]
    refresh_bank(bank, cfg, 1, torch.device("cpu"), path_maps)
    assert torch.isfinite(bank["volumes"][0]).all()

    moved_data = moved_dir / "data.yaml"
    data_cfg = load_yaml(moved_data)
    data_cfg["crop_size"] = 9
    save_yaml(moved_data, data_cfg)
    with pytest.raises(ValueError, match="data configuration changed"):
        refresh_bank(bank, cfg, 2, torch.device("cpu"), path_maps)


def test_legacy_external_source_without_data_digest_still_loads(tmp_path):
    _, _, cfg = _write_external_source(tmp_path)
    cfg["source"].pop("data_sha256")
    base_cfg = load_frozen_source(cfg["source"])
    assert base_cfg["data"]["crop_size"] == 8


def test_source_data_digest_requires_sha256_format():
    cfg = load_train_config("tests/fixtures/config/train/sr.yaml", "sr")
    cfg["source"]["data_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="source.data_sha256"):
        validate_sr_config(cfg)


def _write_resume_checkpoint(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    moved_data_dir = tmp_path / "moved_data"
    moved_data_dir.mkdir()
    data_cfg = load_yaml("tests/fixtures/config/train/low_res.yaml")["data"]
    data_file = moved_data_dir / "data.yaml"
    save_yaml(data_file, data_cfg)
    weights = source_dir / "generator.pt"
    weights.write_bytes(b"source weights")
    train_config = load_yaml("tests/fixtures/config/train/low_res.yaml")
    train_config["data"] = "/old/data.yaml"
    config_file = source_dir / "train.yaml"
    save_yaml(config_file, train_config)
    bank_file = tmp_path / "bank.pt"
    bank_file.write_bytes(b"bank")
    sr_cfg = load_train_config("tests/fixtures/config/train/sr.yaml", "sr")
    sr_cfg["source"] = {
        "weights": str(weights),
        "weights_sha256": file_hash(weights),
        "config_sha256": file_hash(config_file),
        "data_sha256": file_hash(data_file),
        "bank": str(bank_file),
        "bank_sha256": file_hash(bank_file),
    }
    payload = {
        "format": "diffusion-gan3d.sr.train",
        "config": sr_cfg,
        "step": 0,
        "data_fingerprint": {},
        "path_maps": [[("/old", str(moved_data_dir))]],
    }
    checkpoint = tmp_path / "resume.pt"
    torch.save(payload, checkpoint)
    return checkpoint, weights, config_file, data_file


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ("weights", "weights changed"),
        ("configuration", "configuration changed"),
        ("data", "data configuration changed"),
    ],
)
def test_resume_rejects_changed_frozen_source_before_loading_bank(
    tmp_path, monkeypatch, changed, message
):
    checkpoint, weights, config_file, data_file = _write_resume_checkpoint(tmp_path)
    if changed == "weights":
        weights.write_bytes(b"changed source weights")
    elif changed == "configuration":
        config_file.write_text(
            config_file.read_text(encoding="utf-8") + "\n# changed\n",
            encoding="utf-8",
        )
    else:
        data_cfg = load_yaml(data_file)
        data_cfg["crop_size"] = 64
        save_yaml(data_file, data_cfg)
    reached_bank = []
    monkeypatch.setattr(
        "src.train.run.sr.load_bank",
        lambda path: reached_bank.append(path) or {},
    )

    with pytest.raises(ValueError, match=message):
        run_sr_train(resume=checkpoint, device="cpu")
    assert reached_bank == []


def test_resume_passes_cumulative_path_maps_to_source_validation(
    tmp_path, monkeypatch
):
    checkpoint, _, _, _ = _write_resume_checkpoint(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    old_weight = Path("/middle/source/generator.pt")
    old_bank = Path("/middle/sr/bank.pt")
    payload["config"]["source"].update(
        weights=str(old_weight), bank=str(old_bank)
    )
    remapped_bank = tmp_path / "current/sr/bank.pt"
    remapped_bank.parent.mkdir(parents=True)
    remapped_bank.write_bytes(b"bank")
    history = [[("/old", "/middle")]]
    payload["path_maps"] = history
    torch.save(payload, checkpoint)
    new_map = [("/middle", str(tmp_path / "current"))]
    seen = {}

    def validate(weights, source, path_maps):
        seen["weights"] = weights
        seen["path_maps"] = path_maps
        raise RuntimeError("source validation reached")

    monkeypatch.setattr("src.train.run.sr.validate_frozen_source", validate)
    with pytest.raises(RuntimeError, match="source validation reached"):
        run_sr_train(resume=checkpoint, path_map=new_map, device="cpu")

    assert seen["weights"] == (tmp_path / "current/source/generator.pt").resolve()
    assert seen["path_maps"] == [
        [("/old", "/middle")],
        [["/middle", (tmp_path / "current").resolve().as_posix()]],
    ]
