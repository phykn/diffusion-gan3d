from pathlib import Path

import pytest
import yaml

from scripts.simul.src.export import generate

ROOT = Path(__file__).parents[2]


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
