from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.api import InferenceAPI, PlaneAnchor
from src.api import inference as inference_module


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.patch_size = 8

    def generate(self, **kwargs) -> torch.Tensor:
        self.calls.append(kwargs)
        return torch.randint(0, 2, (8, 8, 8), dtype=torch.uint8)


class FakeScaledGenerator:
    def __init__(self, generator: FakeGenerator) -> None:
        self.generator = generator
        self.calls: list[dict] = []

    def generate(self, **kwargs) -> torch.Tensor:
        self.calls.append(kwargs)
        return torch.ones(12, 12, 12, dtype=torch.uint8)


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> InferenceAPI:
    weights = tmp_path / "generator.pt"
    weights.touch()
    generator = FakeGenerator()
    monkeypatch.setattr(
        inference_module,
        "load_generation_settings",
        lambda: SimpleNamespace(guidance=1.2, anchor_strength=0.9, overlap=8),
    )
    monkeypatch.setattr(
        inference_module,
        "load_generator",
        lambda _weights, device: generator,
    )
    monkeypatch.setattr(inference_module, "find_train_config", lambda _weights: weights)
    monkeypatch.setattr(
        inference_module,
        "load_yaml",
        lambda _path: {"data": {"crop_size": 6, "input_size": 8}},
    )
    monkeypatch.setattr(inference_module, "ScaledGenerator", FakeScaledGenerator)
    return InferenceAPI(weights, device="cpu")


def test_crop_and_input_sizes_are_loaded_independently(api: InferenceAPI) -> None:
    assert api.crop_size == 6
    assert api.input_size == 8


def test_generate_without_geometry_uses_direct_generator(api: InferenceAPI) -> None:
    result = api.generate(domain=0, seed=4)

    assert result.shape == (8, 8, 8)
    assert result.dtype == torch.uint8
    assert api.generator.calls == [
        {
            "anchors": (),
            "vf": None,
            "size": None,
            "anchor_strength": 0.9,
            "guidance": 1.2,
            "domain": 0,
        }
    ]
    assert api.scaled.calls == []


def test_generate_with_anchor_uses_direct_conditioning(api: InferenceAPI) -> None:
    anchor = PlaneAnchor(torch.zeros(8, 8, dtype=torch.uint8), axis=0, index=0)

    api.generate(anchors=(anchor,), anchor_strength=0.8)

    assert api.generator.calls[0]["anchors"] == (anchor,)
    assert api.generator.calls[0]["anchor_strength"] == 0.8
    assert api.scaled.calls == []


def test_generate_with_shape_uses_scaled_generator(api: InferenceAPI) -> None:
    result = api.generate(shape=(12, 12, 12), storage="cpu", progress=True)

    assert result.shape == (12, 12, 12)
    assert api.generator.calls == []
    assert api.scaled.calls[0]["blocks"] is None
    assert api.scaled.calls[0]["shape"] == (12, 12, 12)
    assert api.scaled.calls[0]["overlap"] == 8
    assert api.scaled.calls[0]["storage"] == "cpu"
    assert api.scaled.calls[0]["progress"] is True


def test_generate_with_anchor_and_blocks_builds_conditioned_base(
    api: InferenceAPI,
) -> None:
    anchor = PlaneAnchor(torch.zeros(8, 8, dtype=torch.uint8), axis=0, index=0)

    api.generate(anchors=(anchor,), blocks=2)

    assert api.generator.calls[0]["anchors"] == (anchor,)
    assert api.scaled.calls[0]["blocks"] == 2
    assert torch.equal(api.scaled.calls[0]["base"], torch.zeros(8, 8, 8)) is False
    assert api.scaled.calls[0]["base_offset"] == (0, None, None)
    assert api.scaled.calls[0]["shape"] is None


def test_explicit_partial_anchor_position_preserves_all_global_axes(
    api: InferenceAPI,
) -> None:
    anchor = PlaneAnchor(
        torch.zeros(4, 4, dtype=torch.uint8),
        axis=1,
        index=2,
        position=(1, 3),
    )

    api.generate(anchors=(anchor,), blocks=2)

    assert api.scaled.calls[0]["base_offset"] == (0, 0, 0)


def test_seed_is_reproducible_without_changing_caller_rng(api: InferenceAPI) -> None:
    first = api.generate(seed=7)
    second = api.generate(seed=7)
    assert torch.equal(first, second)

    torch.manual_seed(31)
    expected = (torch.rand(()), torch.rand(()))
    torch.manual_seed(31)
    before = torch.rand(())
    api.generate(seed=7)
    after = torch.rand(())
    assert torch.equal(before, expected[0])
    assert torch.equal(after, expected[1])


def test_cpu_seed_never_queries_the_cuda_device(
    api: InferenceAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> int:
        raise AssertionError("CPU seeded generation queried CUDA")

    monkeypatch.setattr(torch.cuda, "current_device", fail)

    api.generate(seed=7)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"blocks": 2, "shape": 16}, "blocks and shape"),
        ({"blocks": 2, "size": 8}, "size cannot"),
        ({"base": torch.zeros(8, 8, 8)}, "base requires"),
        ({"overlap": 4}, "only to scale-up"),
    ],
)
def test_generate_rejects_ambiguous_inputs(
    api: InferenceAPI,
    kwargs: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        api.generate(**kwargs)
