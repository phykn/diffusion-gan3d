import numpy as np
import pytest

from src.evaluate.tortuosity import tau, tortuosity


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

    monkeypatch.setattr(tau, "Solver", FakeSolver)
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


def test_tortuosity_preserves_an_explicit_cuda_device_index(monkeypatch):
    devices = []

    class FakeSolver:
        def __init__(self, values, *, device):
            devices.append(device)

        def solve(self, **kwargs):
            return np.asarray([1.0])

    monkeypatch.setattr(tau, "Solver", FakeSolver)
    assert tortuosity(np.zeros((2, 2, 2)), device="cuda:3") == 1
    assert devices == ["cuda:3"]
