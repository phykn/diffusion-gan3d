import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.storage import save_volume

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check/07_tilt.py"


def test_tilt_check_cuts_real_volumes_to_the_generated_depths():
    depths = runpy.run_path(str(SCRIPT))["source_depths"]
    height = SimpleNamespace(height_data={"crop_size": 128, "lo_res_size": 64})
    assert depths(height, 128.0, 64) == slice(128, 256)
    assert depths(SimpleNamespace(height_data=None), 128.0, 64) == slice(None)


def test_tilt_check_keeps_whole_real_volumes_without_a_model(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    volume = (rng.random((12, 16, 16)) < 0.4).astype(np.uint8)
    save_volume(torch.from_numpy(volume), tmp_path / "volume.tiff")
    out = tmp_path / "report.json"
    argv = ["07_tilt.py", "--volumes", str(tmp_path / "volume.tiff")]
    argv += ["--real", str(tmp_path / "volume.tiff"), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(SCRIPT))["main"]()
    phase = json.loads(out.read_text())["phases"]["1"]
    assert phase["real"] == phase["generated"]
