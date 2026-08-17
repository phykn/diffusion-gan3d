import numpy as np
import pytest
import torch

from src.api import metrics as metrics_module
from src.api.metrics import measure_volume


def test_measure_volume_uses_phase_zero_and_axis_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume = torch.tensor(
        [[[0, 1], [0, 1]], [[1, 1], [0, 0]]],
        dtype=torch.uint8,
    )
    calls = []

    def fake_tortuosity(values, *, phase, axis, device):
        calls.append((values, phase, axis, device))
        return 1.75

    monkeypatch.setattr(metrics_module, "tortuosity", fake_tortuosity)

    result = measure_volume(volume, device=torch.device("cpu"))

    assert result.porosity == pytest.approx(0.5)
    assert result.tortuosity == pytest.approx(1.75)
    assert calls == [(volume, 0, 0, torch.device("cpu"))]


def test_measure_volume_marks_failed_tortuosity_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        metrics_module,
        "tortuosity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("no path")),
    )

    result = measure_volume(
        np.zeros((2, 2, 2), dtype=np.uint8), device=torch.device("cpu")
    )

    assert result.porosity == pytest.approx(1.0)
    assert result.tortuosity is None
