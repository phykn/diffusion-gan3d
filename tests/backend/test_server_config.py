from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from backend.src.config import load_server_config
from backend.src.schema import GenerateRequest, PrepareRequest

SERVER_CONFIG = Path(__file__).resolve().parents[1] / "fixtures/config/backend.yaml"


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1024"])
def test_limits_require_positive_integers(tmp_path, value):
    values = load_server_config(SERVER_CONFIG).model_dump()
    values["max_size"] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValidationError, match="max_size"):
        load_server_config(path)


def test_config_rejects_unknown_settings(tmp_path):
    values = load_server_config(SERVER_CONFIG).model_dump()
    values["max_szie"] = 2048
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    with pytest.raises(ValidationError, match="max_szie"):
        load_server_config(path)


def test_config_can_raise_limits_without_changing_request_classes(tmp_path):
    values = load_server_config(SERVER_CONFIG).model_dump()
    values.update(max_size=2048, max_voxels=2048**3)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(values), encoding="utf-8")
    request = GenerateRequest(size=1025)
    with pytest.raises(ValueError, match="server limit"):
        request.check_limits(load_server_config(SERVER_CONFIG))
    request.check_limits(load_server_config(path))


def test_phase_ids_keep_the_uint8_contract():
    with pytest.raises(ValidationError):
        GenerateRequest(anchors=[{"image": [[256]], "axis": 0, "index": 0}])


@pytest.mark.parametrize(
    "values",
    [
        {"domain": True},
        {"domain": "1"},
        {"overlap": 2.0},
        {"progress": True},
        {"include_metrics": True},
        {"anchors": [{"image": [[0]], "axis": True, "index": 0}]},
        {"anchors": [{"image": [[0]], "axis": 0, "index": "0"}]},
    ],
)
def test_generate_request_rejects_coerced_or_unknown_fields(values):
    with pytest.raises(ValidationError):
        GenerateRequest(**values)


def test_prepare_request_rejects_ragged_rows():
    with pytest.raises(ValidationError, match="equal length"):
        PrepareRequest(image=[[0, 1], [0]])
