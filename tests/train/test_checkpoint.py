from pathlib import Path

import torch
from torch import nn

from src.train import (
    build_ema,
    save_model_weights,
    update_ema,
)


class _BufferedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((2,), 2.0))
        self.register_buffer("counter", torch.tensor(3))


def test_model_file_contains_only_ema_weights(tmp_path: Path) -> None:
    online = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4))
    average = build_ema(online)
    critics = nn.ModuleDict(
        {str(axis): nn.Linear(4, 1) for axis in (0, 1, 2)}
    )
    with torch.no_grad():
        for parameter in online.parameters():
            parameter.add_(1.0)
    update_ema(average, online, 0.9)

    first = save_model_weights(tmp_path, average, critics)
    path = save_model_weights(tmp_path, average, critics)

    assert first == path == tmp_path / "model.pt"
    assert path.is_file()
    assert not list(tmp_path.glob(".model.pt.*.tmp"))
    values = torch.load(path, map_location="cpu", weights_only=True)
    expected = average.state_dict()
    assert values.keys() == expected.keys()
    assert all(isinstance(value, torch.Tensor) for value in values.values())
    for name, value in values.items():
        assert torch.equal(value, expected[name])
    for axis in (0, 1, 2):
        critic_path = tmp_path / f"critic_{axis}.pt"
        critic_values = torch.load(
            critic_path,
            map_location="cpu",
            weights_only=True,
        )
        critic_expected = critics[str(axis)].state_dict()
        assert critic_values.keys() == critic_expected.keys()
        for name, value in critic_values.items():
            assert torch.equal(value, critic_expected[name])


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
