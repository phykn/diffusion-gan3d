from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from backend.src.config import load_config
from backend.src.schema import GenerateRequest

SERVER_CONFIG = Path(__file__).resolve().parents[1] / "fixtures/config/backend.yaml"


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1024"])
def test_limits_require_positive_integers(tmp_path, value):
    values = load_config(SERVER_CONFIG).model_dump()
    values["max_size"] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValidationError, match="max_size"):
        load_config(path)


def test_config_rejects_unknown_settings(tmp_path):
    values = load_config(SERVER_CONFIG).model_dump()
    values["max_szie"] = 2048
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValidationError, match="max_szie"):
        load_config(path)


def test_config_can_raise_limits_without_changing_request_classes(tmp_path):
    values = load_config(SERVER_CONFIG).model_dump()
    values.update(max_size=2048, max_voxels=2048**3)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    request = GenerateRequest(size=1025)
    with pytest.raises(ValueError, match="server limit"):
        request.check_limits(load_config(SERVER_CONFIG))
    assert request.check_limits(load_config(path)) is request


def test_phase_ids_keep_the_uint8_contract():
    with pytest.raises(ValidationError):
        GenerateRequest(anchors=[{"image": [[256]], "axis": 0, "index": 0}])
