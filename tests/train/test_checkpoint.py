from pathlib import Path

import torch
from torch import nn

from src.model.denoiser import Denoiser3D
from src.model.ema import build_ema, update_ema
from src.utils import load_model, save_model


class _BufferedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((2,), 2.0))
        self.register_buffer("counter", torch.tensor(3))


def _multiscale_denoiser() -> Denoiser3D:
    return Denoiser3D(
        num_phases=2,
        base_channels=4,
        channel_multipliers=(1, 2, 4),
        embedding_channels=8,
        latent_channels=4,
        num_domains=1,
        anchor_multiscale=True,
    )


def test_model_save_and_load(tmp_path: Path) -> None:
    source = nn.Linear(4, 2)
    path = save_model(tmp_path / "nested" / "model.pt", source)
    restored = nn.Linear(4, 2)

    loaded = load_model(path, restored)

    assert path.is_file()
    assert loaded is restored
    _assert_same_state(restored, source)


def test_models_can_be_saved_separately(tmp_path: Path) -> None:
    generator = nn.Linear(4, 2)
    critic = nn.Linear(2, 1)
    generator_path = save_model(tmp_path / "generator.pt", generator)
    critic_path = save_model(tmp_path / "critic.pt", critic)
    restored_generator = nn.Linear(4, 2)
    restored_critic = nn.Linear(2, 1)

    load_model(generator_path, restored_generator)
    load_model(critic_path, restored_critic)

    _assert_same_state(restored_generator, generator)
    _assert_same_state(restored_critic, critic)


def test_multiscale_model_loads_strictly(tmp_path: Path) -> None:
    source = _multiscale_denoiser()
    with torch.no_grad():
        for index, projection in enumerate(source.anchor_pyramid, start=1):
            projection.weight.fill_(float(index))

    path = save_model(tmp_path / "generator.pt", source)
    restored = _multiscale_denoiser()
    load_model(path, restored)

    _assert_same_state(restored, source)


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
