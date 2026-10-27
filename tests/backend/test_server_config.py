from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.backend.src.config import load_server_config
from app.backend.src.schema import GenerateRequest, PrepareRequest

CONFIG = load_server_config(
    Path(__file__).resolve().parents[1] / "fixtures/config/backend.yaml"
)


def write_config(tmp_path, **values) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(CONFIG.model_dump() | values), encoding="utf-8")
    return path


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1024"])
def test_limits_require_positive_integers(tmp_path, value):
    with pytest.raises(ValidationError, match="max_size"):
        load_server_config(write_config(tmp_path, max_size=value))


def test_config_rejects_unknown_settings(tmp_path):
    with pytest.raises(ValidationError, match="max_szie"):
        load_server_config(write_config(tmp_path, max_szie=2048))


def test_config_can_raise_limits_without_changing_request_classes(tmp_path):
    path = write_config(tmp_path, max_size=2048, max_voxels=2048**3)
    request = GenerateRequest(size=1025)
    with pytest.raises(ValueError, match="server limit"):
        request.check_limits(CONFIG)
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
