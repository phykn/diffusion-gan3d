import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch import nn

from src.data.provenance import fingerprint_data
from src.model.denoiser import Denoiser3D
from src.storage import load_model, save_model
from src.train.ema import build_ema, update_ema


class _BufferedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((2,), 2.0))
        self.register_buffer("counter", torch.tensor(3))


def _denoiser() -> Denoiser3D:
    return Denoiser3D(
        num_phases=2,
        base_channels=4,
        channel_multipliers=(1, 2, 4),
        embedding_channels=8,
        latent_channels=4,
        num_domains=1,
    )


def test_model_save_and_load(tmp_path: Path) -> None:
    source = nn.Linear(4, 2)
    path = save_model(tmp_path / "nested" / "model.pt", source)
    restored = nn.Linear(4, 2)

    loaded = load_model(path, restored)

    assert path.is_file()
    assert loaded is restored
    _assert_same_state(restored, source)


def test_denoiser_loads_strictly(tmp_path: Path) -> None:
    torch.manual_seed(0)
    source = _denoiser()
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.normal_()

    path = save_model(tmp_path / "generator.pt", source)
    restored = _denoiser()
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


def test_data_fingerprints_deduplicate_and_sort_sources(tmp_path):
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("L", (5, 3), 0).save(first)
    Image.new("L", (4, 7), 1).save(second)

    def stream(groups):
        return SimpleNamespace(
            loader=SimpleNamespace(dataset=SimpleNamespace(path_groups=groups))
        )

    streams = {
        1: {2: stream(((second, first),))},
        0: {0: stream(((first,),))},
    }
    fingerprints = fingerprint_data(streams)
    expected_hashes = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (first, second)
    }
    assert list(fingerprints) == list(expected_hashes)
    assert fingerprints == expected_hashes
