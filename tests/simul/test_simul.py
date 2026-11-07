"""scripts/simul against the v1 reference bytes (SPEC Parity checks: simul)."""

import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from scripts.simul.src.export import generate

ROOT = Path(__file__).parents[2]
REFERENCE = torch.load(
    ROOT / "tests" / "fixtures" / "parity" / "simul.pt", weights_only=True
)


def simul_files(config_path: Path) -> dict[str, bytes]:
    """Output files of config_anode.yaml with count 1 and size 64."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["output"]["count"] = 1
    config["geometry"]["size"] = 64
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder) / "simul"
        config["output"]["data_dir"] = str(root)
        generate(config)
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }


def test_simul_matches_v1_bytes():
    files = simul_files(ROOT / "scripts" / "simul" / "config_anode.yaml")
    assert files.keys() == REFERENCE.keys()
    for name, data in REFERENCE.items():
        assert files[name] == data, name


def test_simul_refuses_a_non_empty_output(tmp_path):
    config = yaml.safe_load(
        (ROOT / "scripts" / "simul" / "config_anode.yaml").read_text()
    )
    config["output"]["count"] = 1
    config["geometry"]["size"] = 64
    (tmp_path / "volumes").mkdir()
    (tmp_path / "volumes" / "volume_000.tiff").write_text("x")
    config["output"]["data_dir"] = str(tmp_path)
    with pytest.raises(FileExistsError, match="not empty"):
        generate(config)
