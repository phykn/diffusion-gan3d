from pathlib import Path

import torch
from torch import nn

from src.train.ema import build_ema, update_ema
from src.train.weights import (
    CRITIC_NAMES,
    load_weights,
    save_training_weights,
    save_weights,
)


class _BufferedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((2,), 2.0))
        self.register_buffer("counter", torch.tensor(3))


def test_model_file_contains_only_ema_weights(tmp_path: Path) -> None:
    online = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4))
    average = build_ema(online)
    with torch.no_grad():
        for parameter in online.parameters():
            parameter.add_(1.0)
    update_ema(average, online, 0.9)

    first = save_weights(tmp_path, average)
    path = save_weights(tmp_path, average)

    assert first == path == tmp_path / "model.pt"
    assert path.is_file()
    assert not list(tmp_path.glob(".model.pt.*.tmp"))
    values = torch.load(path, map_location="cpu", weights_only=True)
    expected = average.state_dict()
    assert values.keys() == expected.keys()
    assert all(isinstance(value, torch.Tensor) for value in values.values())
    for name, value in values.items():
        assert torch.equal(value, expected[name])
    assert not tuple(tmp_path.glob("critic_*.pt"))


def test_training_weights_load_independently(
    tmp_path: Path,
) -> None:
    source = nn.Linear(3, 2)
    source_critics = nn.ModuleDict({str(axis): nn.Linear(2, 1) for axis in range(3)})
    with torch.no_grad():
        source.weight.fill_(4.0)
        source.bias.fill_(5.0)
        for axis, critic in enumerate(source_critics.values()):
            critic.weight.fill_(axis + 1.0)
            critic.bias.fill_(axis + 2.0)

    path = save_training_weights(tmp_path, source, source_critics)

    assert path == tmp_path / "model.pt"
    assert all((tmp_path / name).is_file() for name in CRITIC_NAMES)
    assert not tuple(tmp_path.glob(".*.tmp"))

    denoiser = nn.Linear(3, 2)
    critics = nn.ModuleDict({str(axis): nn.Linear(2, 1) for axis in range(3)})
    load_weights(tmp_path / "model.pt", denoiser)
    for axis, name in enumerate(CRITIC_NAMES):
        load_weights(tmp_path / name, critics[str(axis)])

    _assert_same_state(denoiser, source)
    for axis in range(3):
        _assert_same_state(critics[str(axis)], source_critics[str(axis)])


def test_ema_is_frozen_and_updates_parameters_and_buffers() -> None:
    online = _BufferedModel()
    average = build_ema(online)
    with torch.no_grad():
        online.weight.fill_(6.0)
        online.counter.fill_(9)

    update_ema(average, online, 0.75)

    assert not average.training
    assert all(not parameter.requires_grad for parameter in average.parameters())
    assert torch.equal(average.weight, torch.full((2,), 3.0))
    assert average.counter.item() == 9


def _assert_same_state(actual: nn.Module, expected: nn.Module) -> None:
    assert all(
        torch.equal(actual.state_dict()[name], value)
        for name, value in expected.state_dict().items()
    )
