import importlib
import math
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.anchor import PlaneAnchor
from src.build import (
    build_models,
    build_trainer,
    load_generator,
    validate_anchor_capacity,
)
from src.diffusion import Diffusion
from src.generate import Generator
from src.model.denoiser import Denoiser3D
from src.scale import DEFAULT_SCALE_OVERLAP, ScaledGenerator, TileBuffer, VolumeState
from src.train.ema import build_ema
from src.train.weights import save_checkpoint, save_weights
from src.utils import save_yaml


def test_ema_weights_generate_categorical_volume(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    run_dir = tmp_path / "run" / "sample"
    run_dir.mkdir(parents=True)
    save_yaml(run_dir / "train.yaml", cfg)
    denoiser, _, _ = build_models(cfg)
    ema = build_ema(denoiser)
    weights = save_weights(run_dir, ema)

    generator = load_generator(weights, device=torch.device("cpu"))
    probs = generator.generate_probs(vf=None)
    vol = generator.generate(vf=None)
    conditioned = generator.generate(vf=(0.5, 0.1, 0.4))

    assert probs.shape == (3, 8, 8, 8)
    assert torch.allclose(
        probs.sum(dim=0),
        torch.ones(8, 8, 8),
    )
    assert vol.shape == (8, 8, 8)
    assert vol.dtype == torch.uint8
    assert int(vol.max()) < cfg["data"]["num_phases"]
    assert conditioned.shape == (8, 8, 8)
    assert conditioned.dtype == torch.uint8
    assert int(conditioned.max()) < cfg["data"]["num_phases"]


def test_generator_loads_numbered_checkpoint_with_run_config(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    run_dir = tmp_path / "run" / "sample"
    run_dir.mkdir(parents=True)
    save_yaml(run_dir / "train.yaml", cfg)
    denoiser, critics, connectivity = build_models(cfg)
    weights = save_checkpoint(
        run_dir,
        10,
        build_ema(denoiser),
        critics,
        connectivity,
    )

    generator = load_generator(weights, device=torch.device("cpu"))

    assert generator.generate().shape == (8, 8, 8)


def test_anchor_aware_weights_accept_soft_plane_condition(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    cfg["anchor"]["training_probability"] = 0.5
    run_dir = tmp_path / "run" / "anchored"
    run_dir.mkdir(parents=True)
    save_yaml(run_dir / "train.yaml", cfg)
    denoiser, _, _ = build_models(cfg)
    ema = build_ema(denoiser)
    with torch.no_grad():
        ema.anchor_input.weight.fill_(0.01)
    weights = save_weights(run_dir, ema)
    generator = load_generator(weights, device=torch.device("cpu"))
    anchor = PlaneAnchor(
        image=torch.randint(
            0,
            cfg["data"]["num_phases"],
            (8, 8),
        ),
        axis=1,
        index=4,
    )

    vol = generator.generate(
        anchors=(anchor,),
        vf=(0.5, 0.1, 0.4),
    )

    assert vol.shape == (8, 8, 8)
    assert vol.dtype == torch.uint8
    assert int(vol.max()) < cfg["data"]["num_phases"]


def test_generator_accepts_anchors_when_training_never_reaches_start(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    cfg["anchor"]["start_step"] = cfg["train"]["total_steps"]
    run_dir = tmp_path / "run" / "unanchored"
    run_dir.mkdir(parents=True)
    save_yaml(run_dir / "train.yaml", cfg)
    denoiser, _, _ = build_models(cfg)
    weights = save_weights(run_dir, build_ema(denoiser))

    generator = load_generator(weights, device=torch.device("cpu"))
    anchor = PlaneAnchor(
        image=torch.zeros(8, 8, dtype=torch.uint8),
        axis=0,
        index=4,
    )

    volume = generator.generate(anchors=(anchor,))

    assert torch.all(volume[4] == 0)


def test_build_trainer_rejects_anchor_batch_larger_than_real_batch(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    cfg["anchor"]["training_probability"] = 1.0
    cfg["train"]["volume_batch_size"] = 3

    with pytest.raises(ValueError, match="volume_batch_size.*data.batch_size"):
        build_trainer(cfg, torch.device("cpu"))


def test_build_trainer_rejects_teacher_bank_smaller_than_one_largest_entry(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    cfg["anchor"]["training_probability"] = 1.0
    cfg["train"]["volume_sizes"] = (128,)

    with pytest.raises(ValueError, match="one largest teacher volume"):
        build_trainer(cfg, torch.device("cpu"))


def test_anchor_capacity_accepts_one_largest_teacher_entry(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg["anchor"]["training_probability"] = 1.0
    cfg["anchor"]["teacher_bank_size_mib"] = 2.01
    cfg["train"]["volume_sizes"] = (128,)

    validate_anchor_capacity(
        data=cfg["data"],
        train=cfg["train"],
        anchor=cfg["anchor"],
        anchor_start_step=0,
    )


def test_generator_prepares_vf_on_its_device() -> None:
    generator = _generator(_TraceModel(), Diffusion(1))

    vf = generator.prepare_vf((5.0, 1.0, 4.0))

    assert vf is not None
    assert vf.shape == (1, 3)
    assert vf.dtype == torch.float32
    assert vf.device == generator.device
    assert torch.allclose(vf, torch.tensor(((0.5, 0.1, 0.4),)))
    assert generator.prepare_vf(None) is None


@pytest.mark.parametrize(
    "vf",
    (
        (0.5, 0.5),
        (0.4, 0.3, 0.2, 0.1),
        ((0.5,), (0.1,), (0.4,)),
    ),
)
def test_prepare_vf_rejects_wrong_shape(
    vf: tuple[object, ...],
) -> None:
    generator = _generator(_TraceModel(), Diffusion(1))

    with pytest.raises(ValueError, match="shape"):
        generator.prepare_vf(vf)


def test_prepare_vf_rejects_zero_sum() -> None:
    generator = _generator(_TraceModel(), Diffusion(1))

    with pytest.raises(ValueError, match="sum"):
        generator.prepare_vf((0.0, 0.0, 0.0))


@pytest.mark.parametrize(
    "vf",
    (
        (-1.0, 2.0, 0.0),
        (float("nan"), 0.5, 0.5),
        (float("inf"), 0.5, 0.5),
    ),
)
def test_prepare_vf_rejects_invalid_values(vf: tuple[float, ...]) -> None:
    generator = _generator(_TraceModel(), Diffusion(1))

    with pytest.raises(ValueError):
        generator.prepare_vf(vf)


def test_regular_sampling_reuses_vf_for_every_reverse_step() -> None:
    model = _TraceModel()
    generator = _generator(model, Diffusion(3))

    generator.generate_probs(vf=(0.5, 0.1, 0.4))

    assert [call.transition for call in model.calls] == [2, 1, 0]
    vf = model.calls[0].vf
    assert vf is not None
    assert vf.shape == (1, 3)
    assert vf.dtype == torch.float32
    assert torch.allclose(vf, torch.tensor(((0.5, 0.1, 0.4),)))
    assert all(call.vf is vf for call in model.calls)


def test_regular_sampling_keeps_unconditional_reverse_steps_unconditioned() -> None:
    model = _TraceModel()
    generator = _generator(model, Diffusion(3))

    generator.generate_probs(vf=None)

    assert [call.transition for call in model.calls] == [2, 1, 0]
    assert all(call.vf is None for call in model.calls)


def test_guidance_scale_one_preserves_default_rng_path() -> None:
    generator = _generator(_ControlledModel(), Diffusion(3))

    torch.manual_seed(71)
    baseline = generator.generate_probs(vf=(0.5, 0.1, 0.4))
    torch.manual_seed(71)
    explicit = generator.generate_probs(
        vf=(0.5, 0.1, 0.4),
        guidance_scale=1.0,
    )

    assert torch.equal(explicit, baseline)


def test_volume_fraction_normalization_handles_large_finite_values() -> None:
    generator = _generator(_ControlledModel(), Diffusion(1))

    vf = generator.prepare_vf((3e38, 3e38, 3e38))

    assert vf is not None
    assert torch.all(vf > 0)
    torch.testing.assert_close(vf.sum(), torch.tensor(1.0))


def test_label_fast_path_matches_probability_sampling() -> None:
    generator = _generator(_ControlledModel(), Diffusion(3))

    torch.manual_seed(72)
    expected = generator.generate_probs(vf=(0.5, 0.1, 0.4)).argmax(dim=0)
    torch.manual_seed(72)
    observed = generator.generate(vf=(0.5, 0.1, 0.4))

    assert observed.dtype == torch.uint8
    assert torch.equal(observed, expected.to(torch.uint8))


def test_guidance_scale_routes_direct_and_scaled_predictions() -> None:
    direct_model = _GuidanceTraceModel()
    direct = _generator(direct_model, Diffusion(2))
    direct.generate_probs(
        vf=(0.5, 0.1, 0.4),
        guidance_scale=1.75,
    )

    scaled_model = _GuidanceTraceModel()
    scaled = ScaledGenerator(_generator(scaled_model, Diffusion(2)))
    baseline_plan = scaled.plan((6, 4, 4), overlap=0)
    scaled.generate(
        shape=(6, 4, 4),
        vf=(0.5, 0.1, 0.4),
        overlap=0,
        progress=False,
        guidance_scale=1.75,
    )
    guided_plan = scaled.stats

    assert direct_model.guidance_scales == [1.75, 1.75]
    assert scaled_model.guidance_scales == [1.75] * 4
    assert guided_plan is not None
    expected_guidance = 3 * guided_plan.tile_size**3 * 16
    assert guided_plan.guidance_bytes == expected_guidance
    assert guided_plan.workspace_bytes == (
        baseline_plan.workspace_bytes + expected_guidance
    )
    assert guided_plan.cuda_bytes == baseline_plan.cuda_bytes + expected_guidance
    assert guided_plan.cpu_bytes == baseline_plan.cpu_bytes + expected_guidance

    for offset in range(0, len(scaled_model.guidance_inputs), 2):
        left, right = scaled_model.guidance_inputs[offset : offset + 2]
        assert left[1] is right[1]
        assert left[2] is right[2]
    vf = scaled_model.guidance_inputs[0][3]
    assert vf is not None
    assert all(call[3] is vf for call in scaled_model.guidance_inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_amp_guidance_runs_direct_and_scaled_with_float32_diffusion_state() -> None:
    device = torch.device("cuda")
    model = Denoiser3D(
        num_phases=3,
        base_channels=4,
        channel_multipliers=(1, 2),
        embedding_channels=8,
        latent_channels=4,
    ).eval().to(device)
    generator = _generator(
        model,
        Diffusion(2).to(device),
        patch_size=4,
        device=device,
        use_amp=True,
    )

    direct = generator.generate(
        vf=(0.5, 0.1, 0.4),
        guidance_scale=1.5,
    )
    scaled = ScaledGenerator(generator).generate(
        shape=4,
        overlap=0,
        vf=(0.5, 0.1, 0.4),
        storage="cuda",
        progress=False,
        guidance_scale=1.5,
    )

    assert direct.shape == scaled.shape == (4, 4, 4)
    assert direct.dtype == scaled.dtype == torch.uint8


def test_guided_sampling_preserves_hard_anchor() -> None:
    model = _GuidanceTraceModel()
    generator = _generator(model, Diffusion(2))
    anchor = PlaneAnchor(
        image=torch.zeros(4, 4, dtype=torch.long),
        axis=0,
        index=1,
    )

    volume = generator.generate(
        anchors=(anchor,),
        guidance_scale=1.5,
    )

    assert torch.all(volume[1] == 0)
    assert model.guidance_scales == [1.5, 1.5]


@pytest.mark.parametrize(
    "guidance_scale",
    (-0.1, float("nan"), float("inf"), True),
)
def test_guidance_scale_rejects_invalid_values(guidance_scale: object) -> None:
    generator = _generator(_ControlledModel(), Diffusion(1))

    with pytest.raises(ValueError, match="guidance_scale"):
        generator.generate_probs(guidance_scale=guidance_scale)
    with pytest.raises(ValueError, match="guidance_scale"):
        ScaledGenerator(generator).generate(
            shape=4,
            overlap=0,
            progress=False,
            guidance_scale=guidance_scale,
        )


def test_scaled_guidance_one_preserves_default_rng_path() -> None:
    scaled = ScaledGenerator(_generator(_ControlledModel(), Diffusion(2)))

    torch.manual_seed(73)
    baseline = scaled.generate(
        shape=(6, 4, 4),
        vf=(0.5, 0.1, 0.4),
        overlap=0,
        progress=False,
    )
    torch.manual_seed(73)
    explicit = scaled.generate(
        shape=(6, 4, 4),
        vf=(0.5, 0.1, 0.4),
        overlap=0,
        progress=False,
        guidance_scale=1.0,
    )

    assert torch.equal(explicit, baseline)
    assert scaled.stats is not None
    assert scaled.stats.guidance_bytes == 0


def test_base_only_guidance_is_a_no_op() -> None:
    model = Denoiser3D(
        num_phases=3,
        base_channels=4,
        channel_multipliers=(1, 2),
        embedding_channels=8,
        latent_channels=4,
    ).eval()
    scaled = ScaledGenerator(_generator(model, Diffusion(2)))
    base = torch.randint(0, 3, (4, 4, 4), dtype=torch.uint8)

    torch.manual_seed(79)
    baseline = scaled.generate(
        shape=(4, 4, 4),
        base=base,
        overlap=0,
        progress=False,
    )
    torch.manual_seed(79)
    guided = scaled.generate(
        shape=(4, 4, 4),
        base=base,
        overlap=0,
        progress=False,
        guidance_scale=1.5,
    )

    assert torch.equal(guided, baseline)
    assert scaled.stats is not None
    assert scaled.stats.guidance_bytes == 0


def test_regular_sampling_accepts_direct_volume_size() -> None:
    generator = _generator(_ControlledModel(), Diffusion(1))

    vol = generator.generate(size=8)

    assert vol.shape == (8, 8, 8)


@pytest.mark.parametrize("size", (0, -1, 1.5, True))
def test_regular_sampling_rejects_invalid_volume_size(size: object) -> None:
    generator = _generator(_ControlledModel(), Diffusion(1))

    with pytest.raises(ValueError, match="size must be a positive integer"):
        generator.generate(size=size)


def test_scaled_generation_shares_time_and_latent_before_each_state_update() -> None:
    events: list[tuple[str, int]] = []
    model = _TraceModel(events)
    diffusion = _TraceDiffusion(timesteps=3, events=events)
    generator = _generator(model, diffusion)

    scaled = ScaledGenerator(generator)
    vol = scaled.generate(
        shape=(6, 4, 4),
        overlap=0,
        progress=False,
    )
    stats = scaled.stats

    assert vol.shape == (6, 4, 4)
    assert vol.dtype == torch.uint8
    assert stats is not None
    assert stats.tile_count == 2
    assert diffusion.sample_calls == 0
    assert [call.transition for call in diffusion.calls] == [2, 2, 1, 1]
    assert [call.current_shape for call in diffusion.calls] == [
        (1, 3, 4, 4, 4),
        (1, 3, 2, 4, 4),
    ] * 2
    assert all(call.current_dtype == torch.float32 for call in diffusion.calls)
    assert events == [
        ("model", 2),
        ("model", 2),
        ("posterior", 2),
        ("posterior", 2),
        ("model", 1),
        ("model", 1),
        ("posterior", 1),
        ("posterior", 1),
        ("model", 0),
        ("model", 0),
    ]

    assert len(model.calls) == 6
    assert all(call.vf is None for call in model.calls)
    for offset, transition in enumerate((2, 1, 0)):
        left, right = model.calls[offset * 2 : offset * 2 + 2]
        assert left.transition == right.transition == transition
        assert left.timestep is right.timestep
        assert left.latent is right.latent
        assert left.current.shape == right.current.shape == (1, 3, 4, 4, 4)


def test_parallel_anchors_use_one_combined_prediction_per_step() -> None:
    model = _AnchorTraceModel()
    diffusion = _TraceDiffusion(timesteps=3)
    generator = _generator(model, diffusion)
    anchors = (
        PlaneAnchor(
            image=torch.zeros(4, 4, dtype=torch.long),
            axis=0,
            index=1,
        ),
        PlaneAnchor(
            image=torch.ones(4, 4, dtype=torch.long),
            axis=0,
            index=3,
        ),
    )

    probs = generator.generate_probs(
        anchors=anchors,
        vf=(0.5, 0.1, 0.4),
    )

    assert probs.shape == (3, 4, 4, 4)
    assert diffusion.sample_calls == 1
    assert [call.transition for call in diffusion.calls] == [2, 1, 0]
    assert len(model.calls) == 3
    assert all(int(call.anchor_mask.sum()) == 32 for call in model.calls)

    vf = model.calls[0].vf
    assert vf is not None
    assert [call.transition for call in model.calls] == [2, 1, 0]
    assert all(call.vf is vf for call in model.calls)
    assert len({call.anchor_image_id for call in model.calls}) == 1
    assert len({call.anchor_mask_id for call in model.calls}) == 1


def test_anchor_strength_scales_combined_mask_for_every_step() -> None:
    model = _AnchorTraceModel()
    generator = _generator(model, Diffusion(2))
    anchor = PlaneAnchor(
        image=torch.zeros(4, 4, dtype=torch.long),
        axis=0,
        index=1,
    )

    generator.generate_probs(
        anchors=(anchor,),
        anchor_strength=0.75,
    )

    assert len(model.calls) == 2
    expected = torch.full((1, 1, 4, 4, 4), 0.75)
    expected[:, :, (0, 2, 3)] = 0.0
    assert all(torch.equal(call.anchor_mask, expected) for call in model.calls)


def test_hard_anchor_is_exact_even_when_model_predicts_another_phase() -> None:
    generator = _generator(
        _OptionalAnchorPhaseModel(phase=2),
        Diffusion(2),
    )
    anchor = PlaneAnchor(
        image=torch.zeros(4, 4, dtype=torch.long),
        axis=0,
        index=1,
    )

    volume = generator.generate(anchors=(anchor,))

    assert torch.all(volume[1] == 0)
    assert torch.all(volume[(0, 2, 3), :, :] == 2)


def test_zero_anchor_strength_matches_unconditioned_rng_path() -> None:
    generator = _generator(
        _OptionalAnchorPhaseModel(phase=2),
        Diffusion(3),
    )
    anchor = PlaneAnchor(
        image=torch.zeros(4, 4, dtype=torch.long),
        axis=0,
        index=1,
    )

    torch.manual_seed(23)
    baseline = generator.generate()
    torch.manual_seed(23)
    conditioned = generator.generate(
        anchors=(anchor,),
        anchor_strength=0.0,
    )

    assert torch.equal(conditioned, baseline)


@pytest.mark.parametrize("strength", (-0.1, 1.1, float("nan"), True))
def test_anchor_strength_rejects_invalid_values(strength: object) -> None:
    generator = _generator(_AnchorTraceModel(), Diffusion(1))

    with pytest.raises(ValueError, match="anchor_strength"):
        generator.generate_probs(anchor_strength=strength)


def test_mixed_axis_anchors_keep_single_combined_prediction() -> None:
    model = _AnchorTraceModel()
    diffusion = _TraceDiffusion(timesteps=2)
    generator = _generator(model, diffusion)
    anchors = (
        PlaneAnchor(
            image=torch.zeros(4, 4, dtype=torch.long),
            axis=0,
            index=1,
        ),
        PlaneAnchor(
            image=torch.zeros(4, 4, dtype=torch.long),
            axis=1,
            index=2,
        ),
    )

    generator.generate_probs(anchors=anchors)

    assert diffusion.sample_calls == 1
    assert len(diffusion.calls) == 2
    assert len(model.calls) == 2
    assert all(int(call.anchor_mask.sum()) == 28 for call in model.calls)


def test_scaled_generation_reuses_vf_for_every_tile_and_transition() -> None:
    model = _TraceModel()
    generator = _generator(model, _TraceDiffusion(timesteps=3))

    ScaledGenerator(generator).generate(
        shape=(6, 4, 4),
        vf=(0.5, 0.1, 0.4),
        overlap=0,
        progress=False,
    )

    assert [call.transition for call in model.calls] == [2, 2, 1, 1, 0, 0]
    vf = model.calls[0].vf
    assert vf is not None
    assert vf.shape == (1, 3)
    assert vf.dtype == torch.float32
    assert torch.allclose(vf, torch.tensor(((0.5, 0.1, 0.4),)))
    assert all(call.vf is vf for call in model.calls)


def test_scaled_generator_returns_probabilities_and_categorical_volume() -> None:
    scaled = ScaledGenerator(_generator(_ControlledModel(), Diffusion(1)))

    probs = scaled.generate_probs(shape=(6, 4, 4), overlap=0, progress=False)
    vol = scaled.generate(shape=(6, 4, 4), overlap=0, progress=False)

    assert probs.shape == (3, 6, 4, 4)
    assert torch.allclose(probs.sum(dim=0), torch.ones(6, 4, 4))
    assert vol.shape == (6, 4, 4)
    assert vol.dtype == torch.uint8
    assert int(vol.max()) < 3
    assert scaled.stats is not None


def test_scaled_generator_uses_default_overlap() -> None:
    scaled = ScaledGenerator(_generator(_ControlledModel(), Diffusion(1)))

    plan = scaled.plan(shape=(6, 4, 4))
    probs = scaled.generate_probs(shape=(6, 4, 4), progress=False)
    vol = scaled.generate(shape=(6, 4, 4), progress=False)

    assert plan.overlap == DEFAULT_SCALE_OVERLAP == 8
    assert probs.shape == (3, 6, 4, 4)
    assert vol.shape == (6, 4, 4)
    assert scaled.stats is not None
    assert scaled.stats.overlap == DEFAULT_SCALE_OVERLAP


def test_overlapping_tiles_read_the_same_unchanged_global_state() -> None:
    model = _OverlapTraceModel()

    ScaledGenerator(_generator(model, _TraceDiffusion(timesteps=2))).generate(
        shape=(6, 4, 4),
        overlap=2,
        progress=False,
    )

    assert len(model.calls) == 4
    first, second = model.calls[:2]
    assert first.current.shape == second.current.shape == (1, 3, 8, 8, 8)
    assert torch.equal(
        first.current[:, :, 4:8, 2:6, 2:6],
        second.current[:, :, :4, 2:6, 2:6],
    )


def test_posterior_receives_clean_prediction_before_state_quantization() -> None:
    diffusion = _TraceDiffusion(timesteps=2)

    ScaledGenerator(_generator(_PreciseModel(), diffusion)).generate(
        shape=4,
        overlap=0,
        storage="cpu",
        progress=False,
    )

    clean = diffusion.calls[0].clean
    expected = torch.tensor(_PreciseModel.value, dtype=torch.float32)
    quantized = expected.half().float()
    assert torch.all(clean == expected)
    assert not torch.equal(clean, torch.full_like(clean, quantized))


def test_overlap_fuses_clean_predictions_before_posterior() -> None:
    model = _TileProbabilityModel(
        ((0.8, 0.2, 0.0), (0.2, 0.8, 0.0)),
    )
    diffusion = _TraceDiffusion(timesteps=2)

    ScaledGenerator(_generator(model, diffusion)).generate_probs(
        shape=(8, 4, 4),
        overlap=2,
        progress=False,
    )

    calls = [call for call in diffusion.calls if call.transition == 1]
    clean = torch.cat([call.clean for call in calls], dim=2)
    probs = (clean + 1.0) * 0.5
    expected = torch.tensor(
        (
            (0.8, 0.8, 0.8, 0.6, 0.4, 0.2, 0.2, 0.2),
            (0.2, 0.2, 0.2, 0.4, 0.6, 0.8, 0.8, 0.8),
            (0.0,) * 8,
        ),
    ).view(1, 3, 8, 1, 1)

    torch.testing.assert_close(probs, expected.expand_as(probs))
    assert sum(math.prod(call.current_shape) for call in calls) == 3 * 8 * 4 * 4
    assert sum(math.prod(call.noise_shape or ()) for call in calls) == 3 * 8 * 4 * 4


def test_final_labels_use_the_fused_prediction() -> None:
    predictions = ((0.55, 0.45, 0.0), (0.05, 0.95, 0.0))
    probs = ScaledGenerator(
        _generator(_TileProbabilityModel(predictions), Diffusion(1))
    ).generate_probs(
        shape=(8, 4, 4),
        overlap=2,
        progress=False,
    )
    vol = ScaledGenerator(
        _generator(_TileProbabilityModel(predictions), Diffusion(1))
    ).generate(
        shape=(8, 4, 4),
        overlap=2,
        progress=False,
    )

    assert torch.equal(vol, probs.argmax(dim=0).to(torch.uint8))
    assert torch.all(vol[2] == 0)
    assert torch.all(vol[3] == 1)


def test_padded_edges_only_use_valid_predictions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaled = ScaledGenerator(_generator(_PaddingModel(), Diffusion(1)))

    def fill_ones(state: VolumeState, tiles: object) -> None:
        del tiles
        state.values.fill_(1.0)

    monkeypatch.setattr(scaled, "fill_noise", fill_ones)
    probs = scaled.generate_probs(
        shape=(9, 7, 5),
        overlap=2,
        progress=False,
    )

    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=0), torch.ones(9, 7, 5))
    assert torch.all(probs[0] == 1.0)
    assert torch.all(probs[1:] == 0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_and_cuda_storage_use_the_same_fusion() -> None:
    device = torch.device("cuda")
    predictions = ((0.8, 0.2, 0.0), (0.2, 0.8, 0.0))
    cpu_diffusion = _TraceDiffusion(timesteps=2)
    cuda_diffusion = _TraceDiffusion(timesteps=2)
    cpu_scaled = ScaledGenerator(
        _generator(
            _TileProbabilityModel(predictions).to(device),
            cpu_diffusion,
            device=device,
        )
    )
    cuda_scaled = ScaledGenerator(
        _generator(
            _TileProbabilityModel(predictions).to(device),
            cuda_diffusion,
            device=device,
        )
    )
    required = cuda_scaled.plan((8, 4, 4), overlap=2).cuda_bytes
    free, _ = torch.cuda.mem_get_info(device)
    if required > free:
        pytest.skip("CUDA does not have enough free memory for the planned workspace")

    cpu_vol = cpu_scaled.generate(
        shape=(8, 4, 4),
        overlap=2,
        storage="cpu",
        progress=False,
    )
    cuda_vol = cuda_scaled.generate(
        shape=(8, 4, 4),
        overlap=2,
        storage="cuda",
        progress=False,
    )
    cpu_clean = torch.cat(
        [call.clean for call in cpu_diffusion.calls if call.transition == 1],
        dim=2,
    )
    cuda_clean = torch.cat(
        [call.clean.cpu() for call in cuda_diffusion.calls if call.transition == 1],
        dim=2,
    )

    assert torch.equal(cpu_vol, cuda_vol)
    torch.testing.assert_close(cpu_clean, cuda_clean)


def test_scale_plan_calculates_2048_layout_without_allocating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _ControlledModel()
    model.downsample_factor = 8
    scaled = ScaledGenerator(_generator(model, Diffusion(1), patch_size=224))

    def reject_allocation(*args, **kwargs):
        del args, kwargs
        raise AssertionError("plan must not allocate a tensor")

    monkeypatch.setattr(torch, "empty", reject_allocation)
    plan = scaled.plan(2048, overlap=16)

    assert plan.shape == (2048, 2048, 2048)
    assert plan.overlap == 16
    assert plan.core_size == 224
    assert plan.grid == (10, 10, 10)
    assert plan.tile_count == 1000
    assert plan.states_bytes == 96 * 1024**3
    assert plan.fusion_bytes == 128 * 1024**3
    assert plan.tile_bytes == 192 * 1024**2
    assert plan.cuda_bytes == (
        plan.states_bytes + plan.fusion_bytes + plan.workspace_bytes
    )
    assert plan.output_bytes == 8 * 1024**3
    assert plan.cpu_bytes == (
        plan.states_bytes
        + plan.fusion_bytes
        + plan.output_bytes
        + 2 * plan.tile_bytes
        + plan.workspace_bytes
    )


def test_auto_storage_uses_cuda_only_with_workspace_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _generator(_ControlledModel(), Diffusion(1))
    generator.device = torch.device("cuda")
    scaled = ScaledGenerator(generator)
    plan = scaled.plan(64, overlap=0)
    monkeypatch.setattr(
        ScaledGenerator,
        "get_available_memory",
        staticmethod(lambda: 2 * plan.cpu_bytes),
    )
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)

    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device: (2 * plan.cuda_bytes, 2 * plan.cuda_bytes),
    )
    assert scaled.select_storage(plan, "auto") == "cuda"

    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device: (plan.cuda_bytes, plan.cuda_bytes),
    )
    assert scaled.select_storage(plan, "auto") == "cpu"


def test_auto_storage_counts_reclaimable_cuda_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _generator(_ControlledModel(), Diffusion(1))
    generator.device = torch.device("cuda")
    scaled = ScaledGenerator(generator)
    plan = scaled.plan(64, overlap=0)
    monkeypatch.setattr(
        ScaledGenerator,
        "get_available_memory",
        staticmethod(lambda: 2 * plan.cpu_bytes),
    )
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device: (plan.cuda_bytes // 2, 2 * plan.cuda_bytes),
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_reserved",
        lambda device: plan.cuda_bytes,
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_allocated",
        lambda device: plan.cuda_bytes // 4,
    )

    assert scaled.select_storage(plan, "auto") == "cuda"


def test_probability_output_bytes_are_counted_for_cuda_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _generator(_ControlledModel(), Diffusion(1))
    generator.device = torch.device("cuda")
    scaled = ScaledGenerator(generator)
    plan = scaled.plan(64, overlap=0)
    output_bytes = 4 * generator.num_phases * math.prod(plan.shape)
    free = math.ceil(plan.cuda_bytes / 0.8)
    assert plan.cuda_bytes <= int(free * 0.8)
    assert plan.cuda_bytes + output_bytes > int(free * 0.8)
    monkeypatch.setattr(
        ScaledGenerator,
        "get_available_memory",
        staticmethod(lambda: 2 * plan.cpu_bytes),
    )
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free, free))

    assert scaled.select_storage(plan, "auto") == "cuda"
    assert scaled.select_storage(plan, "auto", output_bytes) == "cpu"


def test_default_scale_workspace_does_not_force_small_outputs_to_cpu() -> None:
    model = Denoiser3D(
        num_phases=3,
        base_channels=16,
        channel_multipliers=(1, 2, 4, 4),
        embedding_channels=32,
        latent_channels=8,
    )
    scaled = ScaledGenerator(_generator(model, Diffusion(1), patch_size=64))

    plan = scaled.plan(128, overlap=16)

    assert plan.workspace_bytes == 648 * 1024**2


def test_cpu_generation_returns_cpu_uint8_without_creating_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    diffusion = _TraceDiffusion(timesteps=2)
    scaled = ScaledGenerator(_generator(_PhaseModel(phase=2), diffusion))

    vol = scaled.generate(
        shape=(6, 5, 4),
        overlap=0,
        storage="cpu",
        progress=False,
    )

    assert vol.device.type == "cpu"
    assert vol.dtype == torch.uint8
    assert vol.shape == (6, 5, 4)
    assert torch.all(vol == 2)
    assert {call.transition for call in diffusion.calls} == {1}
    assert not tuple(tmp_path.rglob("*"))


def test_failed_generation_creates_no_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    scaled = ScaledGenerator(_generator(_FailModel(), Diffusion(1)))

    with pytest.raises(RuntimeError, match="failed prediction"):
        scaled.generate(
            shape=4,
            overlap=0,
            storage="cpu",
            progress=False,
        )

    assert not tuple(tmp_path.rglob("*"))


def test_cpu_generation_checks_memory_before_allocating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaled = ScaledGenerator(_generator(_ControlledModel(), Diffusion(1)))
    plan = scaled.plan(4, overlap=0)
    monkeypatch.setattr(
        ScaledGenerator,
        "get_available_memory",
        staticmethod(lambda: plan.cpu_bytes),
    )

    def reject_allocation(*args, **kwargs):
        del args, kwargs
        raise AssertionError("memory must be checked before allocation")

    monkeypatch.setattr(torch, "empty", reject_allocation)

    with pytest.raises(MemoryError, match="planned CPU allocation"):
        scaled.generate(
            shape=4,
            overlap=0,
            storage="cpu",
            progress=False,
        )


def test_model_input_never_exceeds_planned_tile() -> None:
    model = _OverlapTraceModel()

    ScaledGenerator(_generator(model, Diffusion(2))).generate(
        shape=(9, 7, 5),
        overlap=2,
        storage="cpu",
        progress=False,
    )

    assert model.calls
    assert all(call.current.shape == (1, 3, 8, 8, 8) for call in model.calls)


def test_generate_probs_rejects_large_streaming_volume_before_allocation() -> None:
    scaled = ScaledGenerator(_generator(_ControlledModel(), Diffusion(1)))

    with pytest.raises(ValueError, match="small in-memory"):
        scaled.generate_probs(shape=512, overlap=0, progress=False)


def test_tile_targets_cover_non_divisible_shape_exactly_once() -> None:
    scaled = ScaledGenerator(_generator(_OverlapTraceModel(), Diffusion(1)))
    plan = scaled.plan((9, 7, 5), overlap=2)
    tiles = scaled.make_tiles(plan)
    coverage = torch.zeros(plan.shape, dtype=torch.int32)

    for tile in tiles:
        coverage[tile.target].add_(1)
        source_shape = tuple(region.stop - region.start for region in tile.source)
        valid_shape = tuple(region.stop - region.start for region in tile.valid)
        assert valid_shape == source_shape

    assert plan.tile_size == 8
    assert plan.grid == (3, 2, 2)
    assert len(tiles) == plan.tile_count == 12
    assert torch.equal(coverage, torch.ones_like(coverage))


def test_scale_plan_uses_configured_overlap_and_patch_core() -> None:
    model = _ControlledModel()
    model.downsample_factor = 4

    plan = ScaledGenerator(_generator(model, Diffusion(1), patch_size=64)).plan(
        128, overlap=16
    )

    assert plan.overlap == 16
    assert plan.tile_size == 96
    assert plan.core_size == 64
    assert plan.base_shell == 8
    assert plan.grid == (2, 2, 2)
    assert plan.tile_count == 8


def test_tile_core_prediction_matches_full_periodic_prediction() -> None:
    model = _LocalModel()
    scaled = ScaledGenerator(_generator(model, Diffusion(1)))
    plan = scaled.plan((7, 6, 5), overlap=1)
    tiles = scaled.make_tiles(plan)
    current = VolumeState(3, plan.shape, torch.device("cpu"))
    next_state = VolumeState(3, plan.shape, torch.device("cpu"))
    assert current.values.dtype == next_state.values.dtype == torch.float16
    values = torch.arange(current.values.numel(), dtype=torch.float32)
    values = values.reshape_as(current.values).remainder(17).div_(8).sub_(1)
    current.values.copy_(values)
    time = torch.zeros(1, dtype=torch.long)
    latent = torch.zeros(1, 4)
    fusion = scaled.make_fusion(plan, tiles, current.values.device)

    scaled.step(
        current,
        next_state,
        tiles,
        time,
        latent,
        None,
        0,
        plan,
        None,
        fusion,
    )
    periodic = F.pad(current.values.float(), (1, 1, 1, 1, 1, 1), mode="circular")
    expected = F.avg_pool3d(periodic, kernel_size=3, stride=1).half().float()

    assert torch.equal(next_state.values.float(), expected)


def test_boundary_tile_reads_periodic_context_from_opposite_faces() -> None:
    model = _OverlapTraceModel()
    scaled = ScaledGenerator(_generator(model, Diffusion(1)))
    plan = scaled.plan((6, 7, 5), overlap=2)
    tiles = scaled.make_tiles(plan)
    current = VolumeState(3, plan.shape, torch.device("cpu"))
    next_state = VolumeState(3, plan.shape, torch.device("cpu"))
    coordinates = torch.arange(math.prod(plan.shape), dtype=torch.float32).reshape(
        plan.shape
    )
    current.values.zero_()
    current.values[0, 0].copy_(coordinates)
    fusion = scaled.make_fusion(plan, tiles, current.values.device)

    scaled.step(
        current,
        next_state,
        tiles,
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, 4),
        None,
        0,
        plan,
        None,
        fusion,
    )

    indices = tuple(
        torch.arange(-plan.overlap, plan.core_size + plan.overlap).remainder(size)
        for size in plan.shape
    )
    expected = coordinates
    for axis, index in enumerate(indices):
        expected = expected.index_select(axis, index)
    observed = model.calls[0].current[0, 0]

    assert torch.equal(observed, expected)


def test_periodic_tile_reads_reuse_the_workspace() -> None:
    state = VolumeState(3, (6, 7, 5), torch.device("cpu"))
    buffer = TileBuffer(3, 8, enabled=False)

    first = buffer.read_periodic(state, (-1, -1, -1), 8, torch.device("cpu"))
    second = buffer.read_periodic(state, (1, 2, 3), 8, torch.device("cpu"))

    assert first.data_ptr() == second.data_ptr()


def test_scaled_generation_supports_anisotropic_shape() -> None:
    scaled = ScaledGenerator(_generator(_TraceModel(), _TraceDiffusion(timesteps=1)))
    vol = scaled.generate(
        shape=(6, 4, 8),
        overlap=0,
        progress=False,
    )
    stats = scaled.stats

    assert vol.shape == (6, 4, 8)
    assert vol.dtype == torch.uint8
    assert int(vol.max()) < 3
    assert stats is not None
    assert stats.tile_count == 4
    assert stats.seams == ((4,), (), (4,))


def test_scaled_generation_does_not_force_the_base_into_the_output() -> None:
    diffusion = _TraceDiffusion(timesteps=1)
    scaled = ScaledGenerator(_generator(_PhaseModel(phase=0), diffusion))
    base = torch.full((4, 4, 4), 2, dtype=torch.uint8)

    vol = scaled.generate(
        shape=(6, 6, 6),
        base=base,
        overlap=0,
        progress=False,
    )

    assert vol.shape == (6, 6, 6)
    assert torch.all(vol == 0)


def test_zero_overlap_keeps_the_complete_base_active() -> None:
    scaled = ScaledGenerator(_generator(_TraceModel(), Diffusion(1)))
    base = torch.zeros((4, 4, 4), dtype=torch.uint8)
    plan = scaled.plan((6, 4, 4), overlap=0)

    condition = scaled.prepare_base(base, plan)

    assert condition is not None
    assert condition.region == (slice(1, 5), slice(0, 4), slice(0, 4))
    assert torch.all(condition.weight == 1.0)


def test_base_uses_cosine_transition_only_on_expanded_axes() -> None:
    scaled = ScaledGenerator(_generator(_TraceModel(), Diffusion(1), patch_size=8))
    plan = scaled.plan((12, 8, 8), overlap=4)

    condition = scaled.prepare_base(
        torch.zeros((8, 8, 8), dtype=torch.uint8),
        plan,
    )

    assert condition is not None
    expected = torch.tensor((0.25, 0.75, 1.0, 1.0, 1.0, 1.0, 0.75, 0.25))
    torch.testing.assert_close(condition.weight[0, 0, :, 4, 4], expected)
    assert torch.all(condition.weight[0, 0, 4] == 1.0)


def test_base_condition_blends_the_transition() -> None:
    scaled = ScaledGenerator(_generator(_TraceModel(), Diffusion(1), patch_size=8))
    plan = scaled.plan((12, 8, 8), overlap=4)
    condition = scaled.prepare_base(
        torch.zeros((8, 8, 8), dtype=torch.uint8),
        plan,
    )
    assert condition is not None
    state = VolumeState(3, plan.shape, torch.device("cpu"))
    state.values.zero_()

    scaled.condition_base(state, condition, torch.ones_like(condition.clean))

    blended = state.read(condition.region).float()
    torch.testing.assert_close(
        blended,
        condition.weight.expand_as(blended),
    )


def test_scaled_generation_allows_the_complete_base_to_adapt() -> None:
    base = torch.full((8, 8, 8), 2, dtype=torch.uint8)
    generator = _generator(_PhaseModel(phase=0), Diffusion(1), patch_size=8)
    scaled = ScaledGenerator(generator)

    probs = scaled.generate_probs(
        shape=(12, 8, 8),
        overlap=4,
        base=base,
        progress=False,
    )
    vol = scaled.generate(
        shape=(12, 8, 8),
        overlap=4,
        base=base,
        progress=False,
    )

    prob_labels = probs.argmax(dim=0).to(torch.uint8)
    for labels in (prob_labels, vol):
        assert torch.all(labels == 0)


@pytest.mark.parametrize("shape", ((6, 6, 6), (8, 6, 4), (6, 4, 4)))
def test_scaled_generation_centers_base_in_anisotropic_shape(
    shape: tuple[int, int, int],
) -> None:
    scaled = ScaledGenerator(_generator(_TraceModel(), Diffusion(1)))
    plan = scaled.plan(shape, overlap=0)

    condition = scaled.prepare_base(
        torch.zeros((4, 4, 4), dtype=torch.uint8),
        plan,
    )

    assert condition is not None
    for size, region in zip(shape, condition.region, strict=True):
        assert region.start is not None
        assert region.stop is not None
        assert abs(region.start - (size - region.stop)) <= 1


def test_scaled_base_keeps_constant_prediction_in_every_core() -> None:
    model = _PhaseModel(phase=0)
    diffusion = _TraceDiffusion(timesteps=3)
    base = torch.zeros((4, 4, 4), dtype=torch.uint8)

    vol = ScaledGenerator(_generator(model, diffusion)).generate(
        shape=(6, 6, 6),
        base=base,
        overlap=0,
        progress=False,
    )

    assert model.call_count == 3 * 8
    assert len(diffusion.calls) == 2 * 8
    for call in diffusion.calls:
        assert torch.all(call.clean[:, 0] == 1.0)
        assert torch.all(call.clean[:, 1:] == -1.0)
    assert isinstance(vol, torch.Tensor)
    assert torch.all(vol == 0)


def test_scaled_generation_conditions_every_step_with_one_base_noise() -> None:
    model = _TraceModel()
    diffusion = _NoiseTraceDiffusion(timesteps=3)

    ScaledGenerator(_generator(model, diffusion)).generate(
        shape=(6, 4, 4),
        base=torch.zeros((4, 4, 4), dtype=torch.uint8),
        overlap=0,
        progress=False,
    )

    assert [call.state for call in diffusion.noise_calls] == [3, 2, 1]
    assert len({call.clean_ptr for call in diffusion.noise_calls}) == 1
    assert len({call.noise_ptr for call in diffusion.noise_calls}) == 1
    first = diffusion.noise_calls[0].noisy
    assert torch.equal(
        model.calls[0].current[:, :, 1:4],
        first[:, :, :3].half().float(),
    )


def test_single_block_base_does_not_replace_prediction() -> None:
    base = torch.arange(4 * 4 * 4).reshape(4, 4, 4).remainder(3).to(torch.uint8)
    scaled = ScaledGenerator(_generator(_PhaseModel(phase=0), Diffusion(1)))

    vol = scaled.generate(
        shape=(4, 4, 4),
        base=base,
        overlap=0,
        progress=False,
    )

    assert torch.all(vol == 0)


def test_zero_overlap_does_not_force_the_whole_base() -> None:
    base = torch.arange(4 * 4 * 4).reshape(4, 4, 4).remainder(3).to(torch.uint8)
    scaled = ScaledGenerator(_generator(_PhaseModel(phase=0), Diffusion(1)))

    vol = scaled.generate(
        shape=(6, 4, 4),
        base=base,
        overlap=0,
        progress=False,
    )

    assert torch.all(vol == 0)


@pytest.mark.parametrize(
    "base,error",
    (
        (torch.zeros((3, 4, 4), dtype=torch.uint8), "shape"),
        (torch.zeros((4, 4, 4), dtype=torch.float32), "uint8"),
        (torch.full((4, 4, 4), 3, dtype=torch.uint8), "outside num_phases"),
    ),
)
def test_scaled_generation_rejects_invalid_base(
    base: torch.Tensor,
    error: str,
) -> None:
    scaled = ScaledGenerator(_generator(_TraceModel(), Diffusion(1)))

    with pytest.raises(ValueError, match=error):
        scaled.generate(shape=6, overlap=0, base=base, progress=False)


@pytest.mark.parametrize("vf", (None, (0.5, 0.1, 0.4)))
def test_single_core_matches_regular_categorical_prediction(
    vf: tuple[float, ...] | None,
) -> None:
    expected = _generator(_PhaseModel(phase=2), Diffusion(1)).generate(vf=vf)
    scaled = ScaledGenerator(_generator(_PhaseModel(phase=2), Diffusion(1)))
    actual = scaled.generate(
        shape=(4, 4, 4),
        vf=vf,
        overlap=0,
        progress=False,
    )
    stats = scaled.stats

    assert torch.equal(actual, expected)
    assert stats is not None
    assert stats.tile_count == 1
    assert stats.seams == ((), (), ())


def test_scaled_generation_validates_shape_overlap_and_progress() -> None:
    generator = _generator(_TraceModel(), _TraceDiffusion(timesteps=1))

    with pytest.raises(TypeError, match="shape must be"):
        ScaledGenerator(generator).generate(shape="2", overlap=0, progress=False)
    with pytest.raises(ValueError, match="three positive integers"):
        ScaledGenerator(generator).generate(shape=(2, 0, 1), overlap=0, progress=False)
    with pytest.raises(ValueError, match="overlap"):
        ScaledGenerator(generator).generate(shape=4, overlap=-1, progress=False)
    with pytest.raises(TypeError, match="progress must be"):
        ScaledGenerator(generator).generate(shape=4, overlap=0, progress=1)
    with pytest.raises(ValueError, match="storage must be"):
        ScaledGenerator(generator).generate(
            shape=4,
            overlap=0,
            storage="mmap",
            progress=False,
        )
    vol = ScaledGenerator(generator).generate(
        shape=4,
        overlap=0,
        storage="cpu",
        progress=False,
    )
    assert vol.device.type == "cpu"
    with pytest.raises(ValueError, match="requires a CUDA"):
        ScaledGenerator(generator).generate(
            shape=4,
            overlap=0,
            storage="cuda",
            progress=False,
        )


def test_scale_quality_skips_axes_without_seams() -> None:
    module = importlib.import_module("scripts.04_check_scale_up")
    vol = torch.arange(12 * 8 * 8, dtype=torch.long).reshape(12, 8, 8) % 3

    quality = module.measure_seams(
        vol.to(torch.uint8),
        ((6,), (), ()),
        4,
        3,
    )

    assert quality.change_ratio[0] is not None
    assert quality.change_ratio[1:] == (None, None)
    assert quality.transition_tv[1:] == (None, None)
    assert quality.continuation_delta[1:] == (None, None)


@dataclass(frozen=True)
class _ModelCall:
    transition: int
    timestep: torch.Tensor
    latent: torch.Tensor
    current: torch.Tensor
    current_ptr: int
    vf: torch.Tensor | None


@dataclass(frozen=True)
class _AnchorModelCall:
    transition: int
    timestep: torch.Tensor
    latent: torch.Tensor
    current_ptr: int
    vf: torch.Tensor | None
    anchor_image_id: int
    anchor_mask_id: int
    anchor_mask: torch.Tensor


@dataclass(frozen=True)
class _PosteriorCall:
    transition: int
    current_shape: tuple[int, ...]
    current_dtype: torch.dtype
    noise_shape: tuple[int, ...] | None
    clean_pointer: int
    clean: torch.Tensor


@dataclass(frozen=True)
class _NoiseCall:
    state: int
    clean_ptr: int
    noise_ptr: int
    noisy: torch.Tensor


class _TraceModel(torch.nn.Module):
    def __init__(self, events: list[tuple[str, int]] | None = None) -> None:
        super().__init__()
        self.calls: list[_ModelCall] = []
        self.events = events

    def forward(
        self,
        current: torch.Tensor,
        timestep: torch.Tensor,
        latent: torch.Tensor,
        *,
        vf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        transition = int(timestep.item())
        if self.events is not None:
            self.events.append(("model", transition))
        self.calls.append(
            _ModelCall(
                transition=transition,
                timestep=timestep,
                latent=latent,
                current=current.detach().clone(),
                current_ptr=current.untyped_storage().data_ptr(),
                vf=vf,
            )
        )
        bias = latent.mean(dim=1).reshape(current.shape[0], 1, 1, 1, 1)
        return torch.tanh(0.25 * current + bias)


class _OverlapTraceModel(_TraceModel):
    downsample_factor = 2


class _AnchorTraceModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[_AnchorModelCall] = []

    def forward(
        self,
        current: torch.Tensor,
        timestep: torch.Tensor,
        latent: torch.Tensor,
        *,
        anchor_image: torch.Tensor,
        anchor_mask: torch.Tensor,
        vf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls.append(
            _AnchorModelCall(
                transition=int(timestep.item()),
                timestep=timestep,
                latent=latent,
                current_ptr=current.untyped_storage().data_ptr(),
                vf=vf,
                anchor_image_id=id(anchor_image),
                anchor_mask_id=id(anchor_mask),
                anchor_mask=anchor_mask.detach().clone(),
            )
        )
        return torch.tanh(0.25 * current + 0.05 * anchor_image)


class _PhaseModel(torch.nn.Module):
    def __init__(self, phase: int) -> None:
        super().__init__()
        self.phase = phase
        self.call_count = 0

    def forward(
        self,
        current: torch.Tensor,
        timestep: torch.Tensor,
        latent: torch.Tensor,
        *,
        vf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del timestep, latent, vf
        self.call_count += 1
        pred = torch.full_like(current, -1.0)
        pred[:, self.phase] = 1.0
        return pred


class _OptionalAnchorPhaseModel(_PhaseModel):
    def forward(
        self,
        current: torch.Tensor,
        timestep: torch.Tensor,
        latent: torch.Tensor,
        *,
        vf: torch.Tensor | None = None,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del anchor_image, anchor_mask
        return super().forward(current, timestep, latent, vf=vf)


class _LocalModel(torch.nn.Module):
    downsample_factor = 1

    def forward(
        self,
        current: torch.Tensor,
        timestep: torch.Tensor,
        latent: torch.Tensor,
        *,
        vf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del timestep, latent, vf
        return F.avg_pool3d(
            current,
            kernel_size=3,
            stride=1,
            padding=1,
            count_include_pad=True,
        )


class _FailModel(torch.nn.Module):
    def forward(
        self,
        current: torch.Tensor,
        timestep: torch.Tensor,
        latent: torch.Tensor,
        *,
        vf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del current, timestep, latent, vf
        raise RuntimeError("failed prediction")


class _PreciseModel(torch.nn.Module):
    value = 0.123456

    def forward(
        self,
        current: torch.Tensor,
        timestep: torch.Tensor,
        latent: torch.Tensor,
        *,
        vf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del timestep, latent, vf
        return torch.full_like(current, self.value)


class _TileProbabilityModel(torch.nn.Module):
    def __init__(self, probs: tuple[tuple[float, ...], ...]) -> None:
        super().__init__()
        self.probs = probs
        self.call_count = 0

    def forward(
        self,
        current: torch.Tensor,
        timestep: torch.Tensor,
        latent: torch.Tensor,
        *,
        vf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del timestep, latent, vf
        probs = torch.tensor(
            self.probs[self.call_count % len(self.probs)],
            device=current.device,
            dtype=current.dtype,
        )
        self.call_count += 1
        clean = probs.mul(2.0).sub(1.0).view(1, -1, 1, 1, 1)
        return clean.expand_as(current)


class _PaddingModel(torch.nn.Module):
    def forward(
        self,
        current: torch.Tensor,
        timestep: torch.Tensor,
        latent: torch.Tensor,
        *,
        vf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del timestep, latent, vf
        valid = current.abs().sum(dim=1, keepdim=True) > 0
        probs = torch.zeros_like(current)
        probs[:, :1].copy_(valid)
        probs[:, 2:].copy_(~valid)
        return probs.mul(2.0).sub(1.0)


class _ControlledModel(torch.nn.Module):
    def forward(
        self,
        current: torch.Tensor,
        timestep: torch.Tensor,
        latent: torch.Tensor,
        *,
        vf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = (current.shape[0], 1, 1, 1, 1)
        time = timestep.to(current.dtype).reshape(shape)
        style = latent.mean(dim=1).reshape(shape)
        condition = (
            torch.zeros(shape, device=current.device, dtype=current.dtype)
            if vf is None
            else vf[:, :1].to(current.dtype).reshape(shape)
        )
        return torch.tanh(0.2 * current + 0.02 * time + 0.1 * style + 0.05 * condition)


class _GuidanceTraceModel(_ControlledModel):
    def __init__(self) -> None:
        super().__init__()
        self.guidance_scales: list[float] = []
        self.guidance_inputs: list[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
            ]
        ] = []

    def predict_guided(
        self,
        current: torch.Tensor,
        timestep: torch.Tensor,
        latent: torch.Tensor,
        guidance_scale: float,
        vf: torch.Tensor | None = None,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del anchor_image, anchor_mask
        self.guidance_scales.append(guidance_scale)
        self.guidance_inputs.append((current, timestep, latent, vf))
        return self.forward(current, timestep, latent, vf=vf)


class _TraceDiffusion(Diffusion):
    def __init__(
        self,
        *,
        timesteps: int,
        events: list[tuple[str, int]] | None = None,
    ) -> None:
        super().__init__(timesteps)
        self.sample_calls = 0
        self.calls: list[_PosteriorCall] = []
        self.events = events

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        initial_noise: torch.Tensor,
        latent_channels: int,
        *,
        conditions: dict[str, object] | None = None,
        known_clean: torch.Tensor | None = None,
        known_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.sample_calls += 1
        return super().sample(
            model,
            initial_noise,
            latent_channels,
            conditions=conditions,
            known_clean=known_clean,
            known_mask=known_mask,
        )

    def sample_posterior(
        self,
        current: torch.Tensor,
        pred: torch.Tensor,
        transition: int,
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None and transition > 0:
            noise = torch.randn_like(current)
        if self.events is not None:
            self.events.append(("posterior", transition))
        self.calls.append(
            _PosteriorCall(
                transition=transition,
                current_shape=tuple(current.shape),
                current_dtype=current.dtype,
                noise_shape=None if noise is None else tuple(noise.shape),
                clean_pointer=pred.data_ptr(),
                clean=pred.detach().clone(),
            )
        )
        if transition == 0:
            return pred
        return pred.clone()


class _NoiseTraceDiffusion(_TraceDiffusion):
    def __init__(self, *, timesteps: int) -> None:
        super().__init__(timesteps=timesteps)
        self.noise_calls: list[_NoiseCall] = []

    def add_noise(
        self,
        clean: torch.Tensor,
        state: int | torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert noise is not None
        noisy = super().add_noise(clean, state, noise=noise)
        value = int(state.item()) if isinstance(state, torch.Tensor) else state
        self.noise_calls.append(
            _NoiseCall(
                state=value,
                clean_ptr=clean.data_ptr(),
                noise_ptr=noise.data_ptr(),
                noisy=noisy.detach().clone(),
            )
        )
        return noisy


def _generator(
    model: torch.nn.Module,
    diffusion: Diffusion,
    patch_size: int = 4,
    device: torch.device | None = None,
    use_amp: bool = False,
) -> Generator:
    device = torch.device("cpu") if device is None else device
    return Generator(
        model,
        diffusion,
        device=device,
        patch_size=patch_size,
        num_phases=3,
        latent_channels=4,
        use_amp=use_amp,
    )


def _config(root: Path) -> dict:
    return {
        "data": {
            "folders": {axis: root / str(axis) for axis in (0, 1, 2)},
            "crop_size": 8,
            "input_size": 8,
            "num_phases": 3,
            "batch_size": 2,
            "num_workers": 0,
        },
        "model": {
            "base_channels": 4,
            "channel_multipliers": (1, 2),
            "embedding_channels": 8,
            "latent_channels": 4,
            "critic_channels": (4, 8),
            "gradient_checkpointing": False,
        },
        "diffusion": {"timesteps": 2, "beta_min": 0.1, "beta_max": 2.0},
        "anchor": {
            "training_probability": 0.0,
            "start_step": 0,
            "ramp_steps": 0,
            "multi_anchor_prob": 0.5,
            "max_density": 0.05,
            "min_spacing": 2,
            "mixed_axis_prob": 0.5,
            "teacher_bank_size_mib": 1,
            "loss_weight": 0.0,
        },
        "conditioning": {
            "cfg_dropout": {
                "drop_each_prob": 0.0,
                "single_condition_drop_prob": 0.0,
            }
        },
        "connectivity": {
            "loss_weight": 0.0,
            "normal_transition_loss_weight": 0.0,
            "replay_triplets_per_axis": 1,
            "replay_capacity_per_axis": 2,
            "max_triplets_per_step": 1,
            "reversal_invariant": True,
        },
        "vf": {
            "loss_weight": 1.0,
        },
        "scale_consistency": {
            "overlap": 4,
            "probability": 0.0,
            "start_step": 0,
            "ramp_steps": 0,
            "loss_weight": 0.0,
        },
        "optim": {
            "denoiser_lr": 1e-3,
            "critic_lr": 1e-3,
            "beta1": 0.0,
            "beta2": 0.9,
            "r1_gamma": 0.0,
            "r1_interval": 2,
            "local_loss_weight": 0.5,
        },
        "train": {
            "total_steps": 10,
            "volume_batch_size": 1,
            "volume_sizes": (8,),
            "slice_pairs_per_axis": 2,
            "mixed_precision": False,
            "ema_decay": 0.9,
            "save_every_steps": 1,
        },
    }
