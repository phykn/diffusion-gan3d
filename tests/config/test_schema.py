import pytest

from src.config.schema import LRConfig, build_config, plane_groups, to_dict, update


def _values(**data):
    return {
        "stage": "lr",
        "data": {
            "images": {"xy": ["data"], "xz": ["data"], "yz": ["data"]},
            "num_phases": 2,
            "crop_size": 32,
            "lo_res_size": 16,
            "hi_res_size": 32,
        }
        | data,
    }


def test_defaults_and_round_trip():
    config = build_config(_values())
    assert isinstance(config, LRConfig)
    assert config.train.total_steps == 10000
    assert build_config(to_dict(config)) == config


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"trian": {}}, "trian"),
        ({"train": {"total_steps": 0}}, "train.total_steps"),
        ({"train": {"total_steps": "many"}}, "train.total_steps"),
        ({"augmentation": {"probability": 1.5}}, "augmentation.probability"),
        ({"optim": {"ema_decay": 1.0}}, "optim.ema_decay"),
        ({"loss": {"mirror_balance": {"wieght": 1.0}}}, "loss.mirror_balance.wieght"),
    ],
)
def test_errors_name_the_key(change, message):
    with pytest.raises(ValueError, match=message):
        build_config(_values() | change)


def test_missing_data_key():
    values = _values()
    del values["data"]["crop_size"]
    with pytest.raises(ValueError, match="data.crop_size"):
        build_config(values)


def test_update_and_groups():
    config = update(build_config(_values()), "train.total_steps", 3)
    assert config.train.total_steps == 3
    assert plane_groups(config) == {"xy_xz_yz": (0, 1, 2)}
    split = update(config, "model.critic.plane_groups", [["xy"], ["xz", "yz"]])
    assert plane_groups(split) == {"xy": (0,), "xz_yz": (1, 2)}
