from pathlib import Path

import pytest
import torch

import run_train_1st
import run_train_2nd
from src.train.relocate import PathRemapper, relocate_checkpoint
from src.train.run.low_res import run_low_res_train
from src.train.run.sr import run_sr_train


@pytest.mark.parametrize("old", ["/previous/project", "C:/previous/project"])
def test_relocation_maps_path_fields_without_copying_tensors_or_changing_hashes(
    tmp_path, old
):
    image = f"{old}/images/a.png"
    tensor = torch.ones(2, 3)
    payload = {
        "config": {
            "data": {
                "domains": {0: {"xz": [f"{old}/images"]}},
                "split": {
                    "validation_files": [f"{old}/images/heldout.png"],
                    "validation_regions": {image: [0, 0, 1, 1]},
                },
            },
            "train": {"initial_weights": f"{old}/initial.pt", "total_steps": 5},
            "source": {
                "weights": f"{old}/lr/generator.pt",
                "bank": f"{old}/sr/lr_bank/step_00000000.pt",
                "weights_sha256": "weights hash",
                "config_sha256": "config hash",
                "bank_sha256": "bank hash",
            },
        },
        "data_fingerprint": {image: "image hash"},
        "model": {"weight": tensor},
        "anchor_bank": {0: [{"volume": tensor, "geometry": {"image_id": image}}]},
    }
    moved = relocate_checkpoint(payload, [(old, str(tmp_path))])
    cfg = moved["config"]
    assert cfg["data"]["domains"][0]["xz"] == [str(tmp_path / "images")]
    assert cfg["data"]["split"] == {
        "validation_files": [str(tmp_path / "images/heldout.png")],
        "validation_regions": {str(tmp_path / "images/a.png"): [0, 0, 1, 1]},
    }
    assert cfg["train"] == {
        "initial_weights": str(tmp_path / "initial.pt"),
        "total_steps": 5,
    }
    assert cfg["source"]["weights"] == str(tmp_path / "lr/generator.pt")
    assert cfg["source"]["bank"] == str(tmp_path / "sr/lr_bank/step_00000000.pt")
    for key in ("weights_sha256", "config_sha256", "bank_sha256"):
        assert cfg["source"][key] == payload["config"]["source"][key]
    assert moved["data_fingerprint"] == {str(tmp_path / "images/a.png"): "image hash"}
    assert moved["model"] is payload["model"]
    entry = moved["anchor_bank"][0][0]
    assert entry["volume"] is tensor
    assert entry["geometry"]["image_id"] == str(tmp_path / "images/a.png")
    assert payload["anchor_bank"][0][0]["geometry"]["image_id"] == image
    assert payload["config"]["data"]["domains"][0]["xz"] == [f"{old}/images"]
    assert payload["data_fingerprint"] == {image: "image hash"}


def test_prefix_maps_use_path_boundaries_longest_match_and_windows_case(tmp_path):
    mapper = PathRemapper(
        [
            [
                ["C:/old", str(tmp_path / "all")],
                ["C:/old/images", str(tmp_path / "data")],
            ]
        ]
    )
    assert mapper(r"c:\OLD\images\a.png") == str(tmp_path / "data/a.png")
    assert mapper("C:/old/run/file.pt") == str(tmp_path / "all/run/file.pt")
    assert mapper("C:/older/images/a.png") == "C:/older/images/a.png"


def test_repeated_relocation_history_maps_original_frozen_config(tmp_path):
    original = {
        "config": {"data": {"domains": {0: {"xy": ["/old/images"]}}}},
        "data_fingerprint": {"/old/images/a.png": "unchanged"},
    }
    first = relocate_checkpoint(original, [("/old", str(tmp_path / "first"))])
    second = relocate_checkpoint(
        first, [(str(tmp_path / "first"), str(tmp_path / "next"))]
    )
    mapper = PathRemapper(second["path_maps"])
    assert mapper.config(original["config"]) == second["config"]
    assert second["data_fingerprint"] == {
        str(tmp_path / "next/images/a.png"): "unchanged"
    }
    assert relocate_checkpoint(second, None) is second


def test_relocation_rejects_merged_image_paths(tmp_path):
    payload = {"config": {}, "data_fingerprint": {"/a/img.png": "x", "/b/img.png": "y"}}
    with pytest.raises(ValueError, match="merges distinct"):
        relocate_checkpoint(payload, [("/a", str(tmp_path)), ("/b", str(tmp_path))])


@pytest.mark.parametrize("prefixes", [[""], ["/a/../b"], ["C:/old", "c:/OLD"]])
def test_relocation_rejects_ambiguous_prefixes(tmp_path, prefixes):
    with pytest.raises(ValueError, match="path-map"):
        relocate_checkpoint({}, [(old, str(tmp_path)) for old in prefixes])


@pytest.mark.parametrize("runner", [run_low_res_train, run_sr_train])
def test_path_maps_require_resume(runner, tmp_path):
    with pytest.raises(ValueError, match="requires --resume"):
        runner(path_map=[("/old", str(tmp_path))])


@pytest.mark.parametrize("cli", [run_train_1st, run_train_2nd])
def test_cli_accepts_multiple_path_maps(cli):
    args = cli.parse_args(
        [
            "--resume",
            "last.pt",
            "--path-map",
            "C:/old",
            "/new",
            "--path-map",
            "D:/images",
            "/images",
        ]
    )
    assert args.resume == Path("last.pt")
    assert args.path_map == [["C:/old", "/new"], ["D:/images", "/images"]]
