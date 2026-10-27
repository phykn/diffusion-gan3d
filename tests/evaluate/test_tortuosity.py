import numpy as np
import pytest
import torch

import src.evaluate.tortuosity as tortuosity_module
from src.evaluate.tortuosity import pore_tortuosity, tau, tortuosity


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


def test_tortuosity_reads_the_solver_tau_after_inf_replacement(monkeypatch):
    class Solver:
        converged = True

        def __init__(self, *args, **kwargs):
            self.tau = None

        def solve(self, **kwargs):
            self.tau = np.array([np.inf])
            return np.array([0.0])

    monkeypatch.setattr(tau, "Solver", Solver)
    # Phase 0 spans the volume, so the solver runs rather than the span check.
    assert tortuosity(np.zeros((2, 2, 2)), device="cpu") == np.inf


def test_tortuosity_measures_a_straight_channel_along_the_thickness_axis():
    volume = np.ones((8, 8, 8), dtype=np.uint8)
    volume[:, 2:6, 2:6] = 0

    assert tortuosity(volume, phase=0, axis=0, device="cpu") == pytest.approx(1.0)
    assert tortuosity(volume, phase=0, axis=1, device="cpu") == np.inf


@pytest.mark.parametrize("blocked", [True, False])
def test_tortuosity_is_infinite_without_a_spanning_path(monkeypatch, blocked):
    class Solver:
        def __init__(self, *args, **kwargs):
            raise AssertionError("the solver cannot converge without a path.")

    monkeypatch.setattr(tau, "Solver", Solver)
    volume = np.zeros((6, 4, 4), dtype=np.uint8)
    if blocked:
        volume[3] = 1
    else:
        volume[:] = 1
    assert tortuosity(volume, phase=0, axis=0, device="cpu") == float("inf")


def test_pore_tortuosity_uses_phase_zero_and_thickness_axis(monkeypatch):
    volume = torch.zeros(2, 2, 2, dtype=torch.uint8)
    calls = []

    def fake_tortuosity(values, *, phase, axis, device):
        calls.append((values, phase, axis, device))
        return 1.75

    monkeypatch.setattr(tortuosity_module, "tortuosity", fake_tortuosity)
    assert pore_tortuosity(volume, torch.device("cpu")) == pytest.approx(1.75)
    assert calls == [(volume, 0, 0, torch.device("cpu"))]


def test_unconverged_solver_reports_unavailable(monkeypatch, caplog):
    class Solver:
        converged = False

        def __init__(self, *args, **kwargs):
            pass

        def solve(self, **kwargs):
            self.tau = np.array([2.75])

    monkeypatch.setattr(tau, "Solver", Solver)
    volume = torch.zeros(2, 2, 2, dtype=torch.uint8)
    assert pore_tortuosity(volume, torch.device("cpu")) is None
    assert "did not converge" in caplog.text


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

    monkeypatch.setattr(tortuosity_module, "tortuosity", fail)
    with pytest.raises(RuntimeError, match="out of memory"):
        pore_tortuosity(torch.zeros(2, 2, 2, dtype=torch.uint8), torch.device("cpu"))
