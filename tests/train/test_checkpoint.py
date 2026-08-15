from pathlib import Path

import pytest
import torch
from torch import nn

from src.model.denoiser import Denoiser3D
from src.train.ema import build_ema, update_ema
from src.train.weights import (
    CHECKPOINT_DIR,
    CRITIC_C_FILE,
    CRITIC_FILES,
    load_weights,
    save_all_weights,
    save_checkpoint,
    save_weights,
)


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
        anchor_adapter="multiscale",
    )


def test_model_file_contains_only_ema_weights(tmp_path: Path) -> None:
    online = nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 4))
    average = build_ema(online)
    with torch.no_grad():
        for parameter in online.parameters():
            parameter.add_(1.0)
    update_ema(average, online, 0.9)

    first = save_weights(tmp_path, average)
    path = save_weights(tmp_path, average)

    assert first == path == tmp_path / "generator.pt"
    assert path.is_file()
    assert not list(tmp_path.glob(".generator.pt.*.tmp"))
    values = torch.load(path, map_location="cpu", weights_only=True)
    expected = average.state_dict()
    assert values.keys() == expected.keys()
    assert all(isinstance(value, torch.Tensor) for value in values.values())
    for name, value in values.items():
        assert torch.equal(value, expected[name])
    assert not tuple(tmp_path.glob("critic_*.pt"))


def test_multiscale_adapter_checkpoint_is_tensor_only_and_loads_strictly(
    tmp_path: Path,
) -> None:
    source = _multiscale_denoiser()
    with torch.no_grad():
        for index, projection in enumerate(source.anchor_pyramid, start=1):
            projection.weight.fill_(float(index))

    path = save_weights(tmp_path, source)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    restored = _multiscale_denoiser()
    load_weights(path, restored)

    assert saved.keys() == source.state_dict().keys()
    assert all(isinstance(value, torch.Tensor) for value in saved.values())
    _assert_same_state(restored, source)


def test_training_weights_load_independently(
    tmp_path: Path,
) -> None:
    source = nn.Linear(3, 2)
    source_critics = nn.ModuleDict({str(axis): nn.Linear(2, 1) for axis in range(3)})
    source_connectivity = nn.Linear(4, 1)
    with torch.no_grad():
        source.weight.fill_(4.0)
        source.bias.fill_(5.0)
        for axis, critic in enumerate(source_critics.values()):
            critic.weight.fill_(axis + 1.0)
            critic.bias.fill_(axis + 2.0)
        source_connectivity.weight.fill_(7.0)
        source_connectivity.bias.fill_(8.0)

    path = save_all_weights(
        tmp_path,
        source,
        source_critics,
        source_connectivity,
    )

    assert path == tmp_path / "generator.pt"
    assert all((tmp_path / name).is_file() for name in CRITIC_FILES)
    assert (tmp_path / CRITIC_C_FILE).is_file()
    assert not tuple(tmp_path.glob(".*.tmp"))

    denoiser = nn.Linear(3, 2)
    critics = nn.ModuleDict({str(axis): nn.Linear(2, 1) for axis in range(3)})
    connectivity = nn.Linear(4, 1)
    load_weights(tmp_path / "generator.pt", denoiser)
    for axis, name in enumerate(CRITIC_FILES):
        load_weights(tmp_path / name, critics[str(axis)])
    load_weights(tmp_path / CRITIC_C_FILE, connectivity)

    _assert_same_state(denoiser, source)
    for axis in range(3):
        _assert_same_state(critics[str(axis)], source_critics[str(axis)])
    _assert_same_state(connectivity, source_connectivity)


def test_training_weights_save_only_active_axis_critics(tmp_path: Path) -> None:
    source = nn.Linear(3, 2)
    critics = nn.ModuleDict({"1": nn.Linear(2, 1)})
    connectivity = nn.Linear(4, 1)

    save_all_weights(tmp_path, source, critics, connectivity)

    assert (tmp_path / "generator.pt").is_file()
    assert (tmp_path / "critic_1.pt").is_file()
    assert not (tmp_path / "critic_0.pt").exists()
    assert not (tmp_path / "critic_2.pt").exists()
    assert (tmp_path / CRITIC_C_FILE).is_file()


def test_numbered_checkpoint_preserves_complete_weight_set(tmp_path: Path) -> None:
    source = nn.Linear(3, 2)
    critics = nn.ModuleDict({str(axis): nn.Linear(2, 1) for axis in range(3)})
    connectivity = nn.Linear(4, 1)

    path = save_checkpoint(
        tmp_path,
        10_000,
        source,
        critics,
        connectivity,
    )

    root = tmp_path / CHECKPOINT_DIR / "step_00010000"
    assert path == root / "generator.pt"
    assert path.is_file()
    assert all((root / name).is_file() for name in CRITIC_FILES)
    assert (root / CRITIC_C_FILE).is_file()
    assert not tuple((tmp_path / CHECKPOINT_DIR).glob(".*.tmp"))
    expected_states = {
        "generator.pt": source.state_dict(),
        **{
            name: critics[str(axis)].state_dict()
            for axis, name in enumerate(CRITIC_FILES)
        },
        CRITIC_C_FILE: connectivity.state_dict(),
    }
    for name, expected in expected_states.items():
        saved = torch.load(root / name, map_location="cpu", weights_only=True)
        assert saved.keys() == expected.keys()
        assert all(isinstance(value, torch.Tensor) for value in saved.values())
        assert all(torch.equal(saved[key], value) for key, value in expected.items())
    with pytest.raises(FileExistsError, match="already exists"):
        save_checkpoint(
            tmp_path,
            10_000,
            source,
            critics,
            connectivity,
        )


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
