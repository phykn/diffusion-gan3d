from pathlib import Path

import pytest

import src.config.generation as generation_config


@pytest.fixture(autouse=True)
def isolated_generation_presets(monkeypatch):
    monkeypatch.setattr(
        generation_config, "PROJECT_ROOT", Path(__file__).parent / "fixtures"
    )
