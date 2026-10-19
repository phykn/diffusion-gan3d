import numpy as np
import pytest
import torch

import src.evaluate.volume as metrics_module
from src.evaluate.tortuosity import tau
from src.evaluate.volume import pore_tortuosity


def test_pore_tortuosity_uses_phase_zero_and_thickness_axis(monkeypatch):
    volume = torch.zeros(2, 2, 2, dtype=torch.uint8)
    calls = []

    def fake_tortuosity(values, *, phase, axis, device):
        calls.append((values, phase, axis, device))
        return 1.75

    monkeypatch.setattr(metrics_module, "tortuosity", fake_tortuosity)
    assert pore_tortuosity(volume, torch.device("cpu")) == pytest.approx(1.75)
    assert calls == [(volume, 0, 0, torch.device("cpu"))]


def test_unconverged_solver_reports_unavailable(monkeypatch):
    class Solver:
        converged = False

        def __init__(self, *args, **kwargs):
            pass

        def solve(self, **kwargs):
            self.tau = np.array([2.75])

    monkeypatch.setattr(tau, "Solver", Solver)
    volume = torch.zeros(2, 2, 2, dtype=torch.uint8)
    assert pore_tortuosity(volume, torch.device("cpu")) is None


@pytest.mark.parametrize(
    "error",
    [
        torch.OutOfMemoryError("CUDA out of memory"),
        RuntimeError("MPS backend out of memory (MPS allocated: 1 GB)"),
    ],
)
def test_pore_tortuosity_propagates_memory_exhaustion(monkeypatch, error):
    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(metrics_module, "tortuosity", fail)
    with pytest.raises(RuntimeError, match="out of memory"):
        pore_tortuosity(torch.zeros(2, 2, 2, dtype=torch.uint8), torch.device("cpu"))
