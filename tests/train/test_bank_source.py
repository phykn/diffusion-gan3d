from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.config.files import load_yaml, save_yaml
from src.config.train import load_train_config
from src.train.checkpoint import complete_checkpoint
from src.train.run.bank import create_bank, refresh_bank
from src.train.run.source import (
    file_hash,
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


def test_source_and_training_loaders_match(tmp_path):
    assert read_source_config(
        Path("tests/fixtures/config/train/low_res.yaml")
    ) == load_train_config("tests/fixtures/config/train/low_res.yaml")


def test_refresh_bank_remaps_source_data_paths(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    weights = source_dir / "generator.pt"
    weights.write_bytes(b"weights")
    train_config = load_yaml("tests/fixtures/config/train/low_res.yaml")
    old_root = "C:/old/images"
    train_config["data"]["domains"] = {
        0: {plane: [old_root] for plane in ("xy", "xz", "yz")}
    }
    train_config["data"]["split"] = {
        "validation_files": [f"{old_root}/heldout.png"],
        "validation_regions": {},
    }
    train_config["conditioning"]["height_enabled"] = True
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


def _write_resume_checkpoint(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    weights = source_dir / "generator.pt"
    weights.write_bytes(b"source weights")
    config_file = source_dir / "train.yaml"
    save_yaml(config_file, load_yaml("tests/fixtures/config/train/low_res.yaml"))
    bank_file = tmp_path / "bank.pt"
    bank_file.write_bytes(b"bank")
    sr_cfg = load_train_config("tests/fixtures/config/train/sr.yaml", "sr")
    sr_cfg["source"] = {
        "weights": str(weights),
        "weights_sha256": file_hash(weights),
        "config_sha256": file_hash(config_file),
        "bank": str(bank_file),
        "bank_sha256": file_hash(bank_file),
    }
    payload = {
        "format": "diffusion-gan3d.sr.train",
        "config": sr_cfg,
        "step": 0,
        "data_fingerprint": {},
        "path_maps": [],
    }
    checkpoint = tmp_path / "resume.pt"
    torch.save(payload, checkpoint)
    complete_checkpoint(checkpoint)
    return checkpoint, weights, config_file


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ("weights", "weights changed"),
        ("configuration", "configuration changed"),
    ],
)
def test_resume_rejects_changed_frozen_source_before_loading_bank(
    tmp_path, monkeypatch, changed, message
):
    checkpoint, weights, config_file = _write_resume_checkpoint(tmp_path)
    if changed == "weights":
        weights.write_bytes(b"changed source weights")
    else:
        config_file.write_text(
            config_file.read_text(encoding="utf-8") + "\n# changed\n",
            encoding="utf-8",
        )
    reached_bank = []
    monkeypatch.setattr(
        "src.train.run.sr.load_bank",
        lambda path: reached_bank.append(path) or {},
    )

    with pytest.raises(ValueError, match=message):
        run_sr_train(resume=checkpoint, device="cpu")
    assert reached_bank == []


def test_resume_passes_cumulative_path_maps_to_source_validation(tmp_path, monkeypatch):
    checkpoint, _, _ = _write_resume_checkpoint(tmp_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    old_weight = Path("/middle/source/generator.pt")
    old_bank = Path("/middle/sr/bank.pt")
    payload["config"]["source"].update(weights=str(old_weight), bank=str(old_bank))
    remapped_bank = tmp_path / "current/sr/bank.pt"
    remapped_bank.parent.mkdir(parents=True)
    remapped_bank.write_bytes(b"bank")
    history = [[("/old", "/middle")]]
    payload["path_maps"] = history
    checkpoint.with_suffix(".complete").unlink()
    torch.save(payload, checkpoint)
    complete_checkpoint(checkpoint)
    new_map = [("/middle", str(tmp_path / "current"))]
    seen = {}

    def validate(source, path_maps):
        seen["weights"] = Path(source["weights"])
        seen["path_maps"] = path_maps
        raise RuntimeError("source validation reached")

    monkeypatch.setattr("src.train.run.sr.load_frozen_source", validate)
    with pytest.raises(RuntimeError, match="source validation reached"):
        run_sr_train(resume=checkpoint, path_map=new_map, device="cpu")

    assert seen["weights"] == (tmp_path / "current/source/generator.pt").resolve()
    assert seen["path_maps"] == [
        [("/old", "/middle")],
        [["/middle", (tmp_path / "current").resolve().as_posix()]],
    ]
