from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

import src.config.generation as generation_config


@pytest.fixture(autouse=True)
def predictable_available_memory(monkeypatch):
    # Small inference fixtures should not depend on other applications' RAM use.
    # Memory-limit tests override this probe to exercise rejection explicitly.
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: SimpleNamespace(available=8 * 1024**3)
    )


@pytest.fixture(autouse=True)
def isolated_generation_presets(monkeypatch):
    monkeypatch.setattr(
        generation_config, "PROJECT_ROOT", Path(__file__).parent / "fixtures"
    )
