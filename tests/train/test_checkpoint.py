import random
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.train import build_ema, load_checkpoint, save_checkpoint, update_ema


class _DisabledScaler:
    def __init__(self) -> None:
        self.loaded = False

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        assert state_dict == {}
        self.loaded = True


class _BufferedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((2,), 2.0))
        self.register_buffer("counter", torch.tensor(3))


def _make_training_state():
    denoiser = nn.Sequential(
        nn.Linear(4, 8),
        nn.SiLU(),
        nn.Linear(8, 4),
    )
    ema_denoiser = build_ema(denoiser)
    critics = tuple(nn.Linear(8, 1) for _ in range(3))
    denoiser_optimizer = torch.optim.Adam(denoiser.parameters(), lr=1.0e-3)
    critic_optimizer = torch.optim.Adam(
        [parameter for critic in critics for parameter in critic.parameters()],
        lr=2.0e-3,
    )
    return (
        denoiser,
        ema_denoiser,
        critics,
        denoiser_optimizer,
        {"all": critic_optimizer},
    )


def _take_optimizer_step(
    denoiser: nn.Module,
    critics: Sequence[nn.Module],
    denoiser_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
) -> None:
    denoiser_optimizer.zero_grad(set_to_none=True)
    denoiser(torch.randn(3, 4)).square().mean().backward()
    denoiser_optimizer.step()

    critic_optimizer.zero_grad(set_to_none=True)
    loss = sum(critic(torch.randn(3, 8)).square().mean() for critic in critics)
    loss.backward()
    critic_optimizer.step()


def _assert_nested_equal(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert torch.equal(actual, expected)
        return
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
        return
    if isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
        return
    assert actual == expected


def test_checkpoint_round_trip_restores_training_state_and_rng(
    tmp_path: Path,
) -> None:
    torch.manual_seed(11)
    np.random.seed(13)
    random.seed(17)
    (
        denoiser,
        ema_denoiser,
        critics,
        denoiser_optimizer,
        critic_optimizers,
    ) = _make_training_state()
    _take_optimizer_step(
        denoiser,
        critics,
        denoiser_optimizer,
        critic_optimizers["all"],
    )
    update_ema(ema_denoiser, denoiser, 0.9)
    signature = {
        "data": {"folder": {0: Path("slices/0")}},
        "model": {"channels": (4, 8)},
    }

    first = save_checkpoint(
        tmp_path,
        step=6,
        denoiser=denoiser,
        ema_denoiser=ema_denoiser,
        critics=critics,
        denoiser_optimizer=denoiser_optimizer,
        critic_optimizers=critic_optimizers,
        scaler=None,
        config_signature=signature,
    )
    path = save_checkpoint(
        tmp_path,
        step=7,
        denoiser=denoiser,
        ema_denoiser=ema_denoiser,
        critics=critics,
        denoiser_optimizer=denoiser_optimizer,
        critic_optimizers=critic_optimizers,
        scaler=None,
        config_signature=signature,
    )

    assert first == path == tmp_path / "last.pt"
    assert path.is_file()
    assert not list(tmp_path.glob(".last.pt.*.tmp"))
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert raw["step"] == 7
    assert raw["config_signature"]["model"]["channels"] == [4, 8]
    assert raw["config_signature"]["data"]["folder"][0] == str(Path("slices/0"))
    assert set(raw["optimizers"]["critics"]) == {"all"}

    expected_torch = torch.rand(4)
    expected_numpy = np.random.random(4)
    expected_python = random.random()
    with torch.no_grad():
        for module in (denoiser, ema_denoiser, *critics):
            for parameter in module.parameters():
                parameter.zero_()
    denoiser_optimizer.state.clear()
    critic_optimizers["all"].state.clear()
    torch.manual_seed(19)
    np.random.seed(23)
    random.seed(29)

    scaler = _DisabledScaler()
    step = load_checkpoint(
        path,
        denoiser=denoiser,
        ema_denoiser=ema_denoiser,
        critics=critics,
        denoiser_optimizer=denoiser_optimizer,
        critic_optimizers=critic_optimizers,
        scaler=scaler,
        config_signature=signature,
    )

    assert step == 7
    assert scaler.loaded
    _assert_nested_equal(denoiser.state_dict(), raw["models"]["denoiser"])
    _assert_nested_equal(
        ema_denoiser.state_dict(),
        raw["models"]["ema_denoiser"],
    )
    for critic, expected in zip(
        critics,
        raw["models"]["critics"],
        strict=True,
    ):
        _assert_nested_equal(critic.state_dict(), expected)
    _assert_nested_equal(
        denoiser_optimizer.state_dict(),
        raw["optimizers"]["denoiser"],
    )
    _assert_nested_equal(
        critic_optimizers["all"].state_dict(),
        raw["optimizers"]["critics"]["all"],
    )
    assert torch.equal(torch.rand(4), expected_torch)
    assert np.array_equal(np.random.random(4), expected_numpy)
    assert random.random() == expected_python

    before = {
        name: value.detach().clone() for name, value in denoiser.state_dict().items()
    }
    with pytest.raises(ValueError, match="config signature"):
        load_checkpoint(
            path,
            denoiser=denoiser,
            ema_denoiser=ema_denoiser,
            critics=critics,
            denoiser_optimizer=denoiser_optimizer,
            critic_optimizers=critic_optimizers,
            scaler=None,
            config_signature={"different": True},
        )
    _assert_nested_equal(denoiser.state_dict(), before)


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
