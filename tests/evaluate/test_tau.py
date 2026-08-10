import numpy as np
import pytest

from src.evaluate import tau as tau_module
from src.evaluate import tortuosity


def test_tortuosity_orients_selected_phase_for_taufactor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeSolver:
        def __init__(self, conductive, *, device):
            calls.append((conductive.copy(), device))

        def solve(self, *, verbose, conv_crit):
            calls.append((verbose, conv_crit))
            return np.asarray((1.75,))

    monkeypatch.setattr(tau_module.tau, "Solver", FakeSolver)
    volume = np.asarray(
        (
            ((0, 1), (1, 0)),
            ((1, 0), (1, 1)),
        ),
        dtype=np.uint8,
    )

    result = tortuosity(volume, phase=1, axis=1, device="cpu", convergence=1e-4)

    assert result == pytest.approx(1.75)
    assert np.array_equal(calls[0][0], np.moveaxis(volume == 1, 1, 0))
    assert calls[0][1] == "cpu"
    assert calls[1] == (False, 1e-4)
