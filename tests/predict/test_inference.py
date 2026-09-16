import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.api import InferenceAPI, PlaneAnchor
from src.predict import inference as inference_module


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.patch_size = 8

    def generate(self, **kwargs) -> torch.Tensor:
        self.calls.append(kwargs)
        return torch.randint(0, 2, (8, 8, 8), dtype=torch.uint8)


class FakeTiledGenerator:
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
        "load_train_config",
        lambda _path: {"data": {"crop_size": 6, "lo_res_size": 8}},
    )
    monkeypatch.setattr(inference_module, "TiledGenerator", FakeTiledGenerator)
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
            "height_origin": 0.0,
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


def test_generate_with_anchor_and_blocks_passes_global_anchor_to_tiler(
    api: InferenceAPI,
) -> None:
    anchor = PlaneAnchor(torch.zeros(8, 8, dtype=torch.uint8), axis=0, index=0)

    api.generate(anchors=(anchor,), blocks=2)

    assert api.generator.calls == []
    assert api.scaled.calls[0]["blocks"] == 2
    assert api.scaled.calls[0]["anchors"] == (anchor,)
    assert api.scaled.calls[0]["base"] is None
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

    assert api.scaled.calls[0]["anchors"] == (anchor,)


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


def test_probability_api_forwards_tiled_storage_and_geometry(api, monkeypatch):
    calls = []

    def sample(**kwargs):
        calls.append(kwargs)
        return torch.ones(2, 12, 12, 12) / 2

    monkeypatch.setattr(api.scaled, "generate_probs", sample, raising=False)
    probs = api.generate_probs(shape=(12, 12, 12), overlap=2, storage="cpu", seed=3)
    assert calls[0]["storage"] == "cpu"
    assert calls[0]["shape"] == (12, 12, 12)
    assert calls[0]["overlap"] == 2
    assert probs.shape == (2, 12, 12, 12)


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


def test_public_inference_import_does_not_load_training():
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "import sys; from src.api import InferenceAPI, PlaneAnchor, create_app, SuperResolutionAPI; "
            "assert not [name for name in sys.modules if name.startswith('src.train')]",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fail_inside", [False, True])
def test_seed_scope_only_changes_selected_gpu_and_restores_on_error(
    monkeypatch, fail_inside
):
    states = {
        index: torch.Generator().manual_seed(index + 100).get_state()
        for index in (0, 1)
    }
    original = {index: state.clone() for index, state in states.items()}
    cpu_state = torch.get_rng_state().clone()
    active = [0]

    @contextmanager
    def device(index):
        previous, active[0] = active[0], index
        try:
            yield
        finally:
            active[0] = previous

    def seed_selected(seed):
        states[active[0]] = torch.Generator().manual_seed(seed).get_state()

    def forbidden(*args, **kwargs):
        pytest.fail("seeding unrelated devices")

    monkeypatch.setattr(torch, "manual_seed", forbidden)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    monkeypatch.setattr(
        torch.cuda, "get_rng_state", lambda index: states[index].clone()
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state, index: states.__setitem__(index, state),
    )
    monkeypatch.setattr(torch.cuda, "device", device)
    monkeypatch.setattr(torch.cuda, "manual_seed", seed_selected)
    try:
        with inference_module._seeded_rng(7, torch.device("cuda:1")):
            assert torch.equal(states[0], original[0])
            assert torch.equal(states[1], torch.Generator().manual_seed(7).get_state())
            assert torch.equal(
                torch.rand(3), torch.rand(3, generator=torch.Generator().manual_seed(7))
            )
            if fail_inside:
                raise ValueError("inference failed")
    except ValueError:
        assert fail_inside
    assert torch.equal(cpu_state, torch.get_rng_state())
    assert all(torch.equal(states[index], original[index]) for index in states)
    assert active[0] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_seeded_cuda_sampling_restores_cpu_and_device_rng():
    torch.cuda.init()
    cpu = torch.get_rng_state().clone()
    gpu = torch.cuda.get_rng_state(0).clone()
    samples = []
    for _ in range(2):
        with inference_module._seeded_rng(17, torch.device("cuda:0")):
            samples.append(torch.randn(16, device="cuda:0").cpu())
    torch.testing.assert_close(*samples, rtol=0, atol=0)
    assert torch.equal(cpu, torch.get_rng_state())
    assert torch.equal(gpu, torch.cuda.get_rng_state(0))
