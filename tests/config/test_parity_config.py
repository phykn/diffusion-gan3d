"""v2 settings and plane policies against the v1 reference (SPEC Parity checks)."""

import hashlib
from pathlib import Path

import pytest
import yaml

from src.config.io import PROJECT_ROOT
from src.config.sample import build_stage_configs, load_sample
from src.config.schema import to_dict
from src.data.orientation import plane_policies, policy_dict

REFERENCE = yaml.safe_load(
    (Path(__file__).parents[1] / "fixtures" / "parity" / "config.yaml").read_text()
)
# sha256 of the sample files at v1 commit 0608ab8, which v2 trains unchanged.
V1_SAMPLES = {
    "default": "c47f098fbc7d23072727fc71ac6ad6b4dd9a6853acc1828a959584ade07c78de",
    "simul_anode": "a38348953fa5c2294cf7e135ed893d568ede80ff0b46df13bcacaace1c8501a1",
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
    assert policy_dict(plane_policies(config)) == REFERENCE["planes"][sample][stage]


@pytest.mark.parametrize("sample", sorted(V1_SAMPLES))
def test_sample_files_match_v1(sample):
    data = (PROJECT_ROOT / "config" / "samples" / f"{sample}.yaml").read_bytes()
    assert hashlib.sha256(data).hexdigest() == V1_SAMPLES[sample]


def test_defaults_live_in_the_schema_only():
    assert not (PROJECT_ROOT / "config" / "base").exists()
    found = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in (PROJECT_ROOT / "src").rglob("*.py")
        if "DEFAULTS" in path.read_text(encoding="utf-8")
    ]
    assert found == ["src/config/schema.py"]


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
