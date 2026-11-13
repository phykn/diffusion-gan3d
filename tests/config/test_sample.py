import pytest
import yaml

from src.config.io import PROJECT_ROOT
from src.config.sample import load_sample


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
