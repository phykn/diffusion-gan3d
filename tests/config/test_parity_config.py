"""v2 settings and plane policies against the v1 reference."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.config.io import PROJECT_ROOT
from src.config.sample import build_stage_configs, load_sample
from src.config.schema import to_dict
from src.data.orientation import plane_policies

REFERENCE = yaml.safe_load(
    (Path(__file__).parents[1] / "fixtures" / "parity" / "config.yaml").read_text()
)
# sha256 of the parsed sample files (comments ignored) at v1 commit 0608ab8,
# which v2 trains unchanged.
V1_SAMPLES = {
    "default": "5e9a136ad40564ba954f89604179fe319da58adbf7b54d37d35a00ca1d915ecb",
    "simul_anode": "bfdfe0dbf5174e1a2ab3d0a3f016ecdf7cee3fde7a13bc1bcd50c683124e6b04",
}


def _configs(sample: str) -> dict:
    if (
        sample == "simul_anode"
        and not (PROJECT_ROOT / "data/generated/simul_anode/slices/0").is_dir()
    ):
        pytest.skip("data/generated/simul_anode is not generated here.")
    return build_stage_configs(load_sample(sample))


@pytest.mark.parametrize("sample", ["default", "simul_anode"])
@pytest.mark.parametrize("stage", ["lr", "sr"])
def test_config_matches_v1(sample, stage):
    config = _configs(sample)[stage]
    values = yaml.safe_load(yaml.safe_dump(to_dict(config)))
    values.pop("source", None)
    assert values == REFERENCE["configs"][sample][stage]


@pytest.mark.parametrize("sample", ["default", "simul_anode"])
@pytest.mark.parametrize("stage", ["lr", "sr"])
def test_plane_policies_match_v1(sample, stage):
    config = _configs(sample)[stage]
    policies = {
        plane: {"flip_axes": list(policy.flip_axes), "rotate_90": policy.rotate_90}
        for plane, policy in plane_policies(config).items()
    }
    assert policies == REFERENCE["planes"][sample][stage]


@pytest.mark.parametrize("sample", sorted(V1_SAMPLES))
def test_sample_files_match_v1(sample):
    text = (PROJECT_ROOT / "config" / "samples" / f"{sample}.yaml").read_text()
    data = json.dumps(yaml.safe_load(text), sort_keys=True).encode()
    assert hashlib.sha256(data).hexdigest() == V1_SAMPLES[sample]


def test_show_prints_both_stages(capsys):
    import train

    train.main(["default", "--show"])
    out = capsys.readouterr().out
    lr, sr = out.split("# sr\n")
    assert lr.startswith("# lr\n")
    lr_config = yaml.safe_load(lr.removeprefix("# lr\n"))
    sr_config = yaml.safe_load(sr)
    assert lr_config["loss"]["mirror_balance"]["weight"] == 10.0
    assert lr_config["augmentation"]["probability"] == 1.0
    assert sr_config["stage"] == "sr"


@pytest.mark.parametrize("value", ["2", True, 0, 300])
def test_sample_rejects_a_bad_phase_count(tmp_path, value):
    sample = yaml.safe_load(
        (PROJECT_ROOT / "tests/fixtures/tiny/tiny_default.yaml").read_text()
    )
    sample["num_phases"] = value
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(sample), encoding="utf-8")
    with pytest.raises(ValueError, match="num_phases must be an integer"):
        load_sample(path)
