"""Smoke tests: each check script on tiny runs writes its output files."""

import importlib.util
import json
import os
from pathlib import Path

import matplotlib
import pytest
import torch

from src.export import VERSION, read_export

ROOT = Path(__file__).parents[2]
CHECK = ROOT / "scripts" / "check"
IMAGE = ROOT / "tests" / "fixtures" / "tiny" / "default" / "sample.png"
# 04_hr report.json keys.
HR_REPORT = {
    "sr_weights",
    "lr_weights",
    "input",
    "lr_source_verified",
    "seed",
    "height_origin",
    "height_extent",
    "crop_size",
    "lo_res_size",
    "hi_res_size",
    "source_pixels_per_hr_voxel",
    "lr_shape",
    "hr_shape",
    "tile_size",
    "overlap",
    "margin",
    "lr_guidance",
    "lr_phase_fractions",
    "hr_label_phase_fractions",
    "hr_label_coarse_mae",
    "has_measured_3d_reference",
}


@pytest.fixture(autouse=True)
def no_window():
    matplotlib.use("Agg")


def load_check(script):
    spec = importlib.util.spec_from_file_location(script, CHECK / f"{script}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(script, *args):
    load_check(script).main([str(value) for value in args])


def files(folder):
    return {path.name for path in folder.iterdir()}


def test_dataset(tiny_sample, tiny_lr_run, tmp_path):
    run("01_dataset", "--sample", tiny_sample, "--no-view", "--out", tmp_path / "a")
    assert files(tmp_path / "a") == {"crops.png"}
    run("01_dataset", "--weight", tiny_lr_run, "--no-view", "--out", tmp_path / "b")
    assert files(tmp_path / "b") == {"crops.png"}


def test_generate(tiny_lr_run, tmp_path):
    common = ("--weight", tiny_lr_run, "--device", "cpu", "--no-view")
    run("02_generate", *common, "--out", tmp_path / "one")
    assert files(tmp_path / "one") == {"volume.tiff", "volume.png", "volume.json"}
    run("02_generate", *common, "--blocks", 2, 1, 2, "--out", tmp_path / "tiled")
    assert "seams.json" in files(tmp_path / "tiled")
    seams = json.loads((tmp_path / "tiled" / "seams.json").read_text())
    assert seams["transition_tv"][1] is None
    assert seams["transition_tv"][0] is not None


def test_anchor(tiny_lr_run, tmp_path):
    common = ("--weight", tiny_lr_run, "--device", "cpu", "--no-view")
    run("03_anchor", *common, "--count", 2, "--out", tmp_path / "inside")
    assert files(tmp_path / "inside") == {
        "volume.tiff",
        "volume.png",
        "volume.json",
        "anchor.json",
    }
    assert json.loads((tmp_path / "inside" / "anchor.json").read_text())["indices"] == [
        4,
        12,
    ]
    run(
        "03_anchor",
        *common,
        "--anchor",
        IMAGE,
        "--side",
        "end",
        "--axis",
        1,
        "--out",
        tmp_path / "outside",
    )
    assert "volume_continuation.png" in files(tmp_path / "outside")
    report = json.loads((tmp_path / "outside" / "anchor.json").read_text())
    assert report["indices"] == [15] and report["boundary"] is not None


def test_hr(tiny_sr_run, tmp_path):
    run(
        "04_hr",
        "--weight",
        tiny_sr_run,
        "--device",
        "cpu",
        "--no-view",
        "--out",
        tmp_path,
    )
    assert files(tmp_path) == {
        "lr.tiff",
        "hr.tiff",
        "lr_probs.pt",
        "comparison.png",
        "report.json",
    }
    report = json.loads((tmp_path / "report.json").read_text())
    assert report.keys() == HR_REPORT
    assert report["hr_shape"] == [32, 32, 32]


def test_hr_auto_selects_current_version(tiny_sr_run, tmp_path, monkeypatch):
    module = load_check("04_hr")
    monkeypatch.setattr(module, "PROJECT_ROOT", tmp_path)
    payload = read_export(tiny_sr_run, "sr")
    for name, version, modified in (
        ("valid-sr", VERSION, 10),
        ("old-sr", VERSION - 1, 20),
    ):
        folder = tmp_path / "run" / name
        folder.mkdir(parents=True)
        (folder / "train.yaml").write_text("stage: sr\n")
        export = folder / "generator.pt"
        torch.save(payload | {"version": version}, export)
        os.utime(export, (modified, modified))
    assert module.latest_sr().parent.name == "valid-sr"


def test_tilt(tiny_lr_run, tiny_anode, tiny_anode_lr_run, tmp_path):
    common = ("--device", "cpu", "--samples", 2)
    run("05_tilt", "--weight", tiny_lr_run, *common, "--out", tmp_path / "default")
    assert files(tmp_path / "default") == {"report.json"}
    real = tiny_anode.parent / "data" / "volumes" / "volume_000.tiff"
    for origin in (0, 32):
        out = tmp_path / f"anode-{origin}"
        run(
            "05_tilt",
            "--weight",
            tiny_anode_lr_run,
            *common,
            "--real",
            real,
            "--real-stride",
            2,
            "--height-origin",
            origin,
            "--out",
            out,
        )
        report = json.loads((out / "report.json").read_text())
        assert all("real" in entry for entry in report["phases"].values())
