from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

import run_train
from src.build.trainer import build_trainer
from src.config.train import load_train_config
from src.device import (
    check_device,
    default_device,
    empty_cache,
    inference_amp,
    wide_dtype,
)
from src.evaluate.connectivity import transition_counts, transition_tv
from src.evaluate.label import phase_fractions, voxel_accuracy
from src.predict import memory
from src.predict.memory import estimate_memory, select_storage, workspace_bytes
from src.prepare.profile import rebin_profile
from src.prepare.resize import resize_phases
from src.random import preserved_rng, seeded_rng
from src.train.metrics import materialize_metrics
from src.train.step import step_optimizer

MPS = torch.backends.mps.is_available()


@pytest.mark.parametrize(
    "cuda,mps,expected",
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_default_device_prefers_cuda_then_apple_gpu(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert default_device() == expected
    assert run_train.parse_args([]).device == expected
    assert run_train.parse_args(["--device", "mps"]).device == "mps"


def test_check_device_rejects_unavailable_or_unsupported_devices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert check_device("cpu") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="CUDA"):
        check_device("cuda")
    with pytest.raises(RuntimeError, match="MPS"):
        check_device("mps")
    with pytest.raises(ValueError, match="CPU, CUDA or MPS"):
        check_device("meta")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert check_device(torch.device("mps")) == torch.device("mps")


def test_wide_dtype_uses_single_precision_only_on_apple_gpu():
    assert wide_dtype(torch.device("cpu")) == torch.float64
    assert wide_dtype(torch.device("cuda")) == torch.float64
    assert wide_dtype(torch.device("mps")) == torch.float32


@pytest.mark.parametrize(
    "mixed_precision,device,expected",
    [
        (True, "cuda", True),
        (True, "mps", True),
        (True, "cpu", False),
        (False, "cuda", False),
        (False, "mps", False),
    ],
)
def test_prediction_amp_runs_on_cuda_and_apple_gpu(mixed_precision, device, expected):
    assert inference_amp(mixed_precision, torch.device(device)) is expected


def test_empty_cache_releases_the_selected_gpu(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("mps"))
    for device in ("cpu", "cuda", "mps"):
        empty_cache(torch.device(device))
    assert calls == ["cuda", "mps"]


@pytest.mark.parametrize("fail_inside", [False, True])
def test_seed_scope_seeds_and_restores_the_apple_gpu(monkeypatch, fail_inside):
    state = {"value": torch.Generator().manual_seed(100).get_state()}
    original = state["value"].clone()
    cpu = torch.get_rng_state().clone()

    def seed(value):
        state["value"] = torch.Generator().manual_seed(value).get_state()

    monkeypatch.setattr(
        torch.mps, "get_rng_state", lambda device="mps": state["value"].clone()
    )
    monkeypatch.setattr(
        torch.mps,
        "set_rng_state",
        lambda new_state, device="mps": state.__setitem__("value", new_state),
    )
    monkeypatch.setattr(torch.mps, "manual_seed", seed)
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda *_: pytest.fail("CUDA"))
    try:
        with seeded_rng(7, torch.device("mps")):
            assert torch.equal(
                state["value"], torch.Generator().manual_seed(7).get_state()
            )
            if fail_inside:
                raise ValueError("inference failed")
    except ValueError:
        assert fail_inside
    assert torch.equal(state["value"], original)
    assert torch.equal(cpu, torch.get_rng_state())


def test_apple_gpu_storage_counts_its_workspace_in_unified_memory(monkeypatch):
    generator = SimpleNamespace(device=torch.device("mps"), model=None, num_phases=3)
    estimate = estimate_memory(16, 3)
    reserve = workspace_bytes(generator, estimate.tile_shape)
    required = (
        estimate.output_bytes + estimate.buffer_bytes + estimate.storage_bytes + reserve
    )
    monkeypatch.setattr(
        memory.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=2 * required),
    )
    assert select_storage("auto", estimate, generator) == "cpu"
    with pytest.raises(ValueError, match="CUDA generator"):
        select_storage("cuda", estimate, generator)
    monkeypatch.setattr(
        memory.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=(required - reserve // 2) / 0.8),
    )
    with pytest.raises(MemoryError, match="RAM"):
        select_storage("cpu", estimate, generator)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps", marks=pytest.mark.skipif(not MPS, reason="MPS unavailable")
        ),
    ],
)
def test_preserved_rng_replays_noise_drawn_inside_the_block(device):
    device = torch.device(device)
    with preserved_rng(device):
        first = (torch.rand(4), torch.rand(4, device=device).cpu())
    second = (torch.rand(4), torch.rand(4, device=device).cpu())
    assert all(torch.equal(a, b) for a, b in zip(first, second))


@pytest.mark.skipif(not MPS, reason="MPS unavailable")
def test_seeded_apple_gpu_sampling_is_repeatable_and_restores_state():
    before = torch.mps.get_rng_state().clone()
    samples = []
    for _ in range(2):
        with seeded_rng(17, torch.device("mps")):
            samples.append(torch.randn(16, device="mps").cpu())
    torch.testing.assert_close(*samples, rtol=0, atol=0)
    assert torch.equal(before, torch.mps.get_rng_state())


@pytest.mark.skipif(not MPS, reason="MPS unavailable")
def test_single_precision_paths_match_the_cpu_on_apple_gpu():
    torch.manual_seed(0)
    probs = torch.rand(2, 3, 6, 5, 7).softmax(1)
    for shape in ((4, 4, 4), (9, 10, 11)):
        torch.testing.assert_close(
            resize_phases(probs.to("mps"), shape).cpu(),
            resize_phases(probs, shape),
            rtol=1e-5,
            atol=1e-6,
        )
    values = torch.rand(2, 3, 20)
    torch.testing.assert_close(
        rebin_profile(values.to("mps"), 6).cpu(),
        rebin_profile(values, 6),
        atol=1e-5,
        rtol=1e-5,
    )
    labels = torch.randint(3, (6, 6, 6), dtype=torch.uint8)
    gpu = labels.to("mps")
    torch.testing.assert_close(
        phase_fractions(gpu, 3).cpu().double(), phase_fractions(labels, 3)
    )
    assert voxel_accuracy(gpu, gpu) == 1.0
    assert transition_tv(
        transition_counts(gpu[0], gpu[1], 3), transition_counts(gpu[2], gpu[3], 3)
    ) == pytest.approx(
        transition_tv(
            transition_counts(labels[0], labels[1], 3),
            transition_counts(labels[2], labels[3], 3),
        )
    )
    metrics = materialize_metrics(
        {
            "loss": torch.tensor(0.5, device="mps"),
            "count": torch.tensor(3, device="mps"),
        }
    )
    assert metrics == {"loss": 0.5, "count": 3}


@pytest.mark.skipif(not MPS, reason="MPS unavailable")
def test_optimizer_step_reports_gradient_norm_on_apple_gpu():
    model = torch.nn.Linear(3, 2).to("mps")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model(torch.ones(4, 3, device="mps")).sum().backward()
    expected = float(
        torch.cat([p.grad.flatten().cpu() for p in model.parameters()]).norm()
    )
    diagnostics = {}
    assert step_optimizer(
        optimizer, torch.amp.GradScaler(enabled=False), diagnostics, "model"
    )
    assert diagnostics["gradient/model"] == pytest.approx(expected)


@pytest.mark.skipif(not MPS, reason="MPS unavailable")
def test_lr_training_step_runs_on_apple_gpu(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for index in range(2):
        labels = ((np.indices((8, 8)).sum(0) + index) % 3).astype(np.uint8)
        Image.fromarray(labels).save(images / f"{index}.png")
    cfg = load_train_config("tests/fixtures/config/train/low_res.yaml")
    cfg["data"].update(
        domains={0: {p: [str(images)] for p in ("xy", "xz", "yz")}},
        num_phases=3,
        crop_size=8,
        lo_res_size=8,
    )
    cfg["model"]["generator"].update(
        channels=[4, 8], embedding_channels=8, latent_channels=4
    )
    cfg["model"]["critic"].update(channels=[4, 8])
    cfg["model"]["gradient_checkpointing"] = False
    cfg["model"]["diffusion"]["num_steps"] = 2
    cfg["conditioning"]["cfg_dropout_probability"] = 0.5
    cfg["loss"].update(r1_weight=0.01, r1_every_steps=1)
    cfg["train"].update(real_batch_size=2, slice_pairs_per_plane=2)
    trainer = build_trainer(cfg, torch.device("mps"))
    assert not trainer.settings.amp_enabled
    assert trainer.settings.rollout_amp
    optimizers = [trainer.denoiser_optim, *trainer.critic_optims.values()]
    assert all(optimizer.defaults["fused"] for optimizer in optimizers)
    for step in range(2):
        metrics = trainer.step(step, transition=0)
        assert np.isfinite(metrics.generator_total)
    assert trainer.completed_steps == 2
