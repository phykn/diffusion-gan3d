import numpy as np
import pytest

from src.evaluate.tortuosity import tau, tortuosity


def test_tortuosity_orients_selected_phase_for_taufactor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeSolver:
        converged = True

        def __init__(self, conductive, *, device):
            calls.append((conductive.copy(), device))

        def solve(self, *, verbose, conv_crit):
            calls.append((verbose, conv_crit))
            self.tau = np.asarray((1.75,))
            return np.asarray((9.0,))

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
        converged = True

        def __init__(self, values, *, device):
            devices.append(device)

        def solve(self, **kwargs):
            self.tau = np.asarray([1.0])

    monkeypatch.setattr(tau, "Solver", FakeSolver)
    assert tortuosity(np.zeros((2, 2, 2)), device="cuda:3") == 1
    assert devices == ["cuda:3"]


@pytest.mark.parametrize(
    "converged,value,message",
    [
        (False, 2.75, "did not converge"),
        (True, float("nan"), "invalid result"),
        (True, -1, "invalid result"),
        (True, 0, "invalid result"),
    ],
)
def test_tortuosity_rejects_unconverged_or_invalid_results(
    monkeypatch, converged, value, message
):
    class Solver:
        def __init__(self, *args, **kwargs):
            self.converged = converged

        def solve(self, **kwargs):
            self.tau = np.array([value])

    monkeypatch.setattr(tau, "Solver", Solver)
    with pytest.raises(RuntimeError, match=message):
        tortuosity(np.zeros((2, 2, 2)), device="cpu")


def test_tortuosity_preserves_infinity_for_no_percolating_path(monkeypatch):
    class Solver:
        converged = True

        def __init__(self, *args, **kwargs):
            pass

        def solve(self, **kwargs):
            self.tau = np.array([np.inf])

    monkeypatch.setattr(tau, "Solver", Solver)
    assert tortuosity(np.ones((2, 2, 2)), device="cpu") == np.inf


def test_tortuosity_reads_the_solver_tau_after_inf_replacement(monkeypatch):
    class Solver:
        converged = True

        def __init__(self, *args, **kwargs):
            self.tau = None

        def solve(self, **kwargs):
            self.tau = np.array([np.inf])
            return np.array([0.0])

    monkeypatch.setattr(tau, "Solver", Solver)
    assert tortuosity(np.ones((2, 2, 2)), device="cpu") == np.inf


def test_tortuosity_measures_a_straight_channel_along_the_thickness_axis():
    volume = np.ones((8, 8, 8), dtype=np.uint8)
    volume[:, 2:6, 2:6] = 0

    assert tortuosity(volume, phase=0, axis=0, device="cpu") == pytest.approx(1.0)
    assert tortuosity(volume, phase=0, axis=1, device="cpu") == np.inf
