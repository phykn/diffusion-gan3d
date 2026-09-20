import math
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from src.anchor import PlaneAnchor
from src.build.model import build_models
from src.build.predict import load_generator
from src.build.trainer import build_trainer
from src.config.files import save_yaml
from src.evaluate.seam import measure_seams
from src.model.denoiser import Denoiser3D
from src.model.diffusion import Diffusion
from src.predict.generator import Generator
from src.predict.tiling.fusion import get_axis_windows, make_fusion
from src.predict.tiling.layout import crop_output, make_tiles
from src.predict.tiling.sampler import TiledGenerator
from src.predict.tiling.state import TileBuffer, VolumeState
from src.storage import save_model
from src.train.ema import build_ema

_SCALED_GENERATE = TiledGenerator.generate
_SCALED_GENERATE_PROBS = TiledGenerator.generate_probs
_GENERATOR_GENERATE = Generator.generate
_GENERATOR_GENERATE_PROBS = Generator.generate_probs


@pytest.fixture(autouse=True)
def preserve_generation_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    def generate(self, *args, **kwargs):
        kwargs.setdefault("margin", 0)
        return _SCALED_GENERATE(self, *args, **kwargs)

    def generate_probs(self, *args, **kwargs):
        kwargs.setdefault("margin", 0)
        return _SCALED_GENERATE_PROBS(self, *args, **kwargs)

    def direct_generate(self, *args, **kwargs):
        kwargs.setdefault("margin", 0)
        return _GENERATOR_GENERATE(self, *args, **kwargs)

    def direct_generate_probs(self, *args, **kwargs):
        kwargs.setdefault("margin", 0)
        return _GENERATOR_GENERATE_PROBS(self, *args, **kwargs)

    monkeypatch.setattr(TiledGenerator, "generate", generate)
    monkeypatch.setattr(TiledGenerator, "generate_probs", generate_probs)
    monkeypatch.setattr(Generator, "generate", direct_generate)
    monkeypatch.setattr(Generator, "generate_probs", direct_generate_probs)


def test_build_models_uses_boolean_anchor_multiscale(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)

    single, _, _ = build_models(cfg)
    cfg["model"]["generator"]["anchor_multiscale_input"] = True
    multiscale, _, _ = build_models(cfg)

    assert not single.anchor_multiscale
    assert len(single.anchor_pyramid) == 0
    assert multiscale.anchor_multiscale
    assert len(multiscale.anchor_pyramid) == 1


def test_build_models_rejects_non_multiple_generator_channels(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cfg["model"]["generator"]["channels"] = [4, 7]

    with pytest.raises(ValueError, match="multiples"):
        build_models(cfg)


def test_ema_weights_generate_categorical_volume(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    run_dir = tmp_path / "run" / "sample"
    run_dir.mkdir(parents=True)
    save_yaml(run_dir / "train.yaml", cfg)
    denoiser, _, _ = build_models(cfg)
    ema = build_ema(denoiser)
    weights = save_model(run_dir / "generator.pt", ema)

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
    denoiser, _, _ = build_models(cfg)
    weights = save_model(
        run_dir / "checkpoints" / "step_00000010" / "generator.pt",
        build_ema(denoiser),
    )

    generator = load_generator(weights, device=torch.device("cpu"))

    assert generator.generate().shape == (8, 8, 8)


def test_anchor_aware_weights_accept_soft_plane_condition(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    cfg["model"]["generator"]["anchor_multiscale_input"] = True
    cfg["conditioning"]["anchor"]["probability"] = 0.5
    run_dir = tmp_path / "run" / "anchored"
    run_dir.mkdir(parents=True)
    save_yaml(run_dir / "train.yaml", cfg)
    denoiser, _, _ = build_models(cfg)
    ema = build_ema(denoiser)
    with torch.no_grad():
        ema.anchor_input.weight.fill_(0.01)
        for projection in ema.anchor_pyramid:
            projection.weight.fill_(0.01)
    weights = save_model(run_dir / "generator.pt", ema)
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
    cfg["conditioning"]["anchor"]["start_step"] = cfg["train"]["total_steps"]
    run_dir = tmp_path / "run" / "unanchored"
    run_dir.mkdir(parents=True)
    save_yaml(run_dir / "train.yaml", cfg)
    denoiser, _, _ = build_models(cfg)
    weights = save_model(run_dir / "generator.pt", build_ema(denoiser))

    generator = load_generator(weights, device=torch.device("cpu"))
    anchor = PlaneAnchor(
        image=torch.zeros(8, 8, dtype=torch.uint8),
        axis=0,
        index=4,
    )

    volume = generator.generate(anchors=(anchor,))

    assert volume.shape == (8, 8, 8)
    assert volume.dtype == torch.uint8
    assert int(volume.max()) < cfg["data"]["num_phases"]


def test_build_trainer_rejects_anchor_batch_larger_than_real_batch(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    cfg["conditioning"]["anchor"]["probability"] = 1.0
    cfg["train"]["volume_batch_size"] = 3

    with pytest.raises(ValueError, match="volume_batch.*train.real_batch_size"):
        build_trainer(cfg, torch.device("cpu"))


def test_generator_prepares_vf_on_its_device() -> None:
    generator = _generator(_TraceModel(), Diffusion(1))

    vf = generator.prepare_vf((5.0, 1.0, 4.0))

    assert vf is not None
    assert vf.shape == (1, 3)
    assert vf.dtype == torch.float32
    assert vf.device == generator.device
    assert torch.allclose(vf, torch.tensor(((0.5, 0.1, 0.4),)))
    assert generator.prepare_vf(None) is None


def test_generator_requires_and_reuses_multi_domain() -> None:
    model = _TraceModel()
    model.num_domains = 2
    generator = _generator(model, Diffusion(3))

    with pytest.raises(ValueError, match="domain is required"):
        generator.generate_probs()

    generator.generate_probs(domain=1)

    assert [call.domain.tolist() for call in model.calls] == [[1], [1], [1]]
    assert all(call.domain is model.calls[0].domain for call in model.calls)


@pytest.mark.parametrize("domain", (-1, 2, True))
def test_generator_rejects_invalid_domain(domain: object) -> None:
    model = _TraceModel()
    model.num_domains = 2
    generator = _generator(model, Diffusion(1))

    with pytest.raises(ValueError, match="domain"):
        generator.generate_probs(domain=domain)


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


def test_scaled_sampling_reuses_domain_for_every_tile_and_step() -> None:
    model = _TraceModel()
    model.num_domains = 2
    scaled = TiledGenerator(_generator(model, Diffusion(2)))

    scaled.generate(
        shape=(6, 4, 4),
        overlap=0,
        progress=False,
        domain=1,
    )

    assert model.calls
    assert all(call.domain.tolist() == [1] for call in model.calls)
    assert all(call.domain is model.calls[0].domain for call in model.calls)


def test_direct_generation_uses_the_model_downsample_factor_as_margin() -> None:
    model = _TraceModel()
    model.downsample_factor = 8
    generator = _generator(model, Diffusion(1))

    volume = _GENERATOR_GENERATE(generator)

    assert volume.shape == (4, 4, 4)
    assert model.calls[0].current.shape == (1, 3, 20, 20, 20)


def test_direct_anchor_coordinates_survive_margin_crop() -> None:
    model = _AnchorTraceModel()
    model.downsample_factor = 8
    generator = _generator(model, Diffusion(1))
    anchor = PlaneAnchor(
        image=torch.ones((4, 4), dtype=torch.long),
        axis=0,
        index=1,
        position=(0, 0),
    )

    volume = _GENERATOR_GENERATE(generator, anchors=(anchor,))

    assert volume.shape == (4, 4, 4)
    assert len(model.calls) == 1
    anchor_mask = model.calls[0].anchor_mask
    assert anchor_mask is not None
    mask = anchor_mask[0, 0]
    assert mask.shape == (20, 20, 20)
    assert int(mask.sum()) == 16
    assert torch.all(mask[9, 8:12, 8:12])


def test_generator_derives_margin_from_one_coarse_3d_cell() -> None:
    model = _TraceModel()
    model.downsample_factor = 4

    generator = _generator(model, Diffusion(1))

    assert generator.default_margin == 4


def test_guidance_one_preserves_default_rng_path() -> None:
    generator = _generator(_ControlledModel(), Diffusion(3))

    torch.manual_seed(71)
    baseline = generator.generate_probs(vf=(0.5, 0.1, 0.4))
    torch.manual_seed(71)
    explicit = generator.generate_probs(
        vf=(0.5, 0.1, 0.4),
        guidance=1.0,
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


def test_guidance_routes_direct_and_scaled_predictions() -> None:
    direct_model = _GuidanceTraceModel()
    direct = _generator(direct_model, Diffusion(2))
    direct.generate_probs(
        vf=(0.5, 0.1, 0.4),
        guidance=1.75,
    )

    scaled_model = _GuidanceTraceModel()
    scaled = TiledGenerator(_generator(scaled_model, Diffusion(2)))
    scaled.generate(
        shape=(6, 4, 4),
        vf=(0.5, 0.1, 0.4),
        overlap=0,
        progress=False,
        guidance=1.75,
    )
    assert direct_model.guidances == [1.75, 1.75]
    assert scaled_model.guidances == [1.75] * 4

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
    model = (
        Denoiser3D(
            num_phases=3,
            base_channels=4,
            channel_multipliers=(1, 2),
            embedding_channels=8,
            latent_channels=4,
            num_domains=1,
        )
        .eval()
        .to(device)
    )
    generator = _generator(
        model,
        Diffusion(2).to(device),
        patch_size=4,
        device=device,
        use_amp=True,
    )

    direct = generator.generate(
        vf=(0.5, 0.1, 0.4),
        guidance=1.5,
    )
    scaled = TiledGenerator(generator).generate(
        shape=4,
        overlap=0,
        vf=(0.5, 0.1, 0.4),
        storage="cuda",
        progress=False,
        guidance=1.5,
    )

    assert direct.shape == scaled.shape == (4, 4, 4)
    assert direct.dtype == scaled.dtype == torch.uint8


def test_guided_sampling_uses_anchor_as_a_condition() -> None:
    model = _GuidanceTraceModel()
    generator = _generator(model, Diffusion(2))
    anchor = PlaneAnchor(
        image=torch.zeros(4, 4, dtype=torch.long),
        axis=0,
        index=1,
    )

    with patch.object(
        model,
        "apply_guidance_logits",
        wraps=model.apply_guidance_logits,
    ) as apply_guidance_logits:
        volume = generator.generate(
            anchors=(anchor,),
            guidance=1.5,
        )

    assert volume.shape == (4, 4, 4)
    assert model.guidances == [1.5] * 2
    for call in apply_guidance_logits.call_args_list:
        assert call.kwargs["anchor_image"] is not None
        assert int(call.kwargs["anchor_mask"].sum()) == 16


def test_scaled_guidance_one_preserves_default_rng_path() -> None:
    scaled = TiledGenerator(_generator(_ControlledModel(), Diffusion(2)))

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
        guidance=1.0,
    )

    assert torch.equal(explicit, baseline)


def test_base_only_guidance_is_a_no_op() -> None:
    model = Denoiser3D(
        num_phases=3,
        base_channels=4,
        channel_multipliers=(1, 2),
        embedding_channels=8,
        latent_channels=4,
        num_domains=1,
    ).eval()
    scaled = TiledGenerator(_generator(model, Diffusion(2)))
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
        guidance=1.5,
    )

    assert torch.equal(guided, baseline)


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

    scaled = TiledGenerator(generator)
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
    for transition in (2, 1):
        calls = [call for call in diffusion.calls if call.transition == transition]
        assert sum(math.prod(call.current_shape[-3:]) for call in calls) == 6 * 4 * 4
        assert all(max(call.current_shape[-3:]) <= 4 for call in calls)
    assert all(call.current_dtype == torch.float32 for call in diffusion.calls)
    assert [transition for _, transition in events] == sorted(
        (transition for _, transition in events), reverse=True
    )
    # Streaming can finalize a slab before the next model call, but never
    # advances diffusion time before all tiles of the current time complete.
    assert events[0] == ("model", 2)

    assert len(model.calls) == 6
    assert all(call.vf is None for call in model.calls)
    for offset, transition in enumerate((2, 1, 0)):
        left, right = model.calls[offset * 2 : offset * 2 + 2]
        assert left.transition == right.transition == transition
        assert left.timestep is right.timestep
        assert left.latent is right.latent
        assert left.current.shape == (1, 3, 4, 4, 4)
        assert right.current.shape == (1, 3, 4, 4, 4)


def test_partial_anchor_strength_scales_mask_on_one_state() -> None:
    model = _AnchorTraceModel()
    diffusion = _TraceDiffusion(timesteps=2)
    gen = _generator(model, diffusion)
    gen.generate_probs(
        anchors=(PlaneAnchor(torch.zeros(4, 4, dtype=torch.long), 0, 1),),
        anchor_strength=0.5,
    )
    assert len(model.calls) == len(diffusion.calls) == 2
    assert all(call.anchor_mask.max() == 0.5 for call in model.calls)


def test_anchor_never_overwrites_a_different_model_prediction() -> None:
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

    assert torch.all(volume == 2)


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


def test_mixed_axis_anchors_use_one_joint_conditioned_state() -> None:
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
    for call in model.calls:
        assert call.anchor_mask is not None
        assert int(call.anchor_mask.sum()) == 28


def test_scaled_generation_reuses_vf_for_every_tile_and_transition() -> None:
    model = _TraceModel()
    generator = _generator(model, _TraceDiffusion(timesteps=3))

    TiledGenerator(generator).generate(
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
    scaled = TiledGenerator(_generator(_ControlledModel(), Diffusion(1)))

    probs = scaled.generate_probs(shape=(6, 4, 4), overlap=0, progress=False)
    vol = scaled.generate(shape=(6, 4, 4), overlap=0, progress=False)

    assert probs.shape == (3, 6, 4, 4)
    assert torch.allclose(probs.sum(dim=0), torch.ones(6, 4, 4))
    assert vol.shape == (6, 4, 4)
    assert vol.dtype == torch.uint8
    assert int(vol.max()) < 3
    assert scaled.stats is not None


def test_blocks_define_fixed_tiles_and_margin_reduced_output() -> None:
    scaled = TiledGenerator(_generator(_ControlledModel(), Diffusion(1), patch_size=8))

    volume = scaled.generate(
        blocks=(3, 2, 1),
        overlap=2,
        progress=False,
    )
    plan = scaled.stats

    assert plan is not None
    assert volume.shape == plan.shape == (16, 12, 8)
    assert plan.grid == (3, 2, 1)
    assert plan.tile_size == 8
    assert plan.stride == 4
    assert plan.seams == ((6, 10), (6,), ())
    assert all(
        tuple(region.stop - region.start for region in tile.source) == (8, 8, 8)
        for tile in make_tiles(plan)
    )


def test_scaled_generator_uses_default_overlap() -> None:
    scaled = TiledGenerator(_generator(_ControlledModel(), Diffusion(1), patch_size=32))

    plan = scaled.plan(shape=(48, 32, 32))
    probs = scaled.generate_probs(shape=(48, 32, 32), progress=False)
    vol = scaled.generate(shape=(48, 32, 32), progress=False)

    assert plan.overlap == 8
    assert probs.shape == (3, 48, 32, 32)
    assert vol.shape == (48, 32, 32)
    assert scaled.stats is not None
    assert scaled.stats.overlap == 8


def test_scaled_generation_uses_model_derived_outer_margin() -> None:
    model = _PhaseModel(phase=1)
    model.downsample_factor = 8
    scaled = TiledGenerator(_generator(model, Diffusion(1), patch_size=32))

    volume = _SCALED_GENERATE(
        scaled,
        shape=32,
        overlap=8,
        progress=False,
    )
    plan = scaled.stats

    assert volume.shape == (32, 32, 32)
    assert torch.all(volume == 1)
    assert plan is not None
    assert plan.shape == (32, 32, 32)
    assert plan.generation_shape == (48, 48, 48)
    assert plan.margin == 8
    assert plan.tile_count == 8


def test_crop_output_keeps_the_requested_rectangular_center() -> None:
    volume = torch.arange(7 * 8 * 9).reshape(7, 8, 9)

    cropped = crop_output(volume, (3, 4, 5), 2)

    assert torch.equal(cropped, volume[2:5, 2:6, 2:7])
    assert cropped.untyped_storage().data_ptr() != volume.untyped_storage().data_ptr()


def test_overlapping_tiles_read_the_same_unchanged_global_state() -> None:
    model = _OverlapTraceModel()

    TiledGenerator(_generator(model, _TraceDiffusion(timesteps=2))).generate(
        shape=(6, 4, 4),
        overlap=1,
        progress=False,
    )

    assert len(model.calls) == 4
    first, second = model.calls[:2]
    assert first.current.shape == (1, 3, 4, 4, 4)
    assert second.current.shape == (1, 3, 4, 4, 4)
    assert torch.equal(first.current[:, :, 2:4], second.current[:, :, :2])


def test_posterior_receives_clean_prediction_before_state_quantization() -> None:
    diffusion = _TraceDiffusion(timesteps=2)

    TiledGenerator(_generator(_PreciseModel(), diffusion)).generate(
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

    TiledGenerator(_generator(model, diffusion, patch_size=8)).generate_probs(
        shape=(12, 8, 8),
        overlap=2,
        progress=False,
    )

    calls = [call for call in diffusion.calls if call.transition == 1]
    clean = torch.cat([call.clean for call in calls], dim=2)
    probs = (clean + 1.0) * 0.5
    expected = torch.tensor(
        (
            (0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 0.4, 0.2, 0.2, 0.2, 0.2, 0.2),
            (0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.6, 0.8, 0.8, 0.8, 0.8, 0.8),
            (0.0,) * 12,
        ),
    ).view(1, 3, 12, 1, 1)

    torch.testing.assert_close(probs, expected.expand_as(probs))
    assert sum(math.prod(call.current_shape) for call in calls) == 3 * 12 * 8 * 8
    assert sum(math.prod(call.noise_shape or ()) for call in calls) == 3 * 12 * 8 * 8


def test_final_labels_use_the_fused_prediction() -> None:
    predictions = ((0.55, 0.45, 0.0), (0.05, 0.95, 0.0))
    probs = TiledGenerator(
        _generator(_TileProbabilityModel(predictions), Diffusion(1))
    ).generate_probs(
        shape=(8, 4, 4),
        overlap=1,
        progress=False,
    )
    vol = TiledGenerator(
        _generator(_TileProbabilityModel(predictions), Diffusion(1))
    ).generate(
        shape=(8, 4, 4),
        overlap=1,
        progress=False,
    )

    assert torch.equal(vol, probs.argmax(dim=0).to(torch.uint8))
    assert torch.all(vol[2] == 0)
    assert torch.all(vol[3] == 1)


def test_boundary_tiles_use_only_real_volume_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scaled = TiledGenerator(_generator(_PaddingModel(), Diffusion(1)))

    def fill_ones(state: VolumeState, tiles: object) -> None:
        del tiles
        state.values.fill_(1.0)

    monkeypatch.setattr(scaled, "fill_noise", fill_ones)
    probs = scaled.generate_probs(
        shape=(9, 7, 5),
        overlap=1,
        progress=False,
    )

    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=0), torch.ones(9, 7, 5))
    assert torch.all(probs[0] == 1.0)
    assert torch.all(probs[1:] == 0.0)


def test_real_denoiser_accepts_one_sided_boundary_tiles() -> None:
    model = Denoiser3D(
        num_phases=3,
        base_channels=4,
        channel_multipliers=(1, 2),
        embedding_channels=8,
        latent_channels=4,
        num_domains=1,
    ).eval()
    scaled = TiledGenerator(_generator(model, Diffusion(1)))

    volume = scaled.generate(
        shape=8,
        overlap=1,
        progress=False,
    )

    assert volume.shape == (8, 8, 8)
    assert volume.dtype == torch.uint8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_and_cuda_storage_use_the_same_fusion() -> None:
    device = torch.device("cuda")
    predictions = ((0.8, 0.2, 0.0), (0.2, 0.8, 0.0))
    cpu_diffusion = _TraceDiffusion(timesteps=2)
    cuda_diffusion = _TraceDiffusion(timesteps=2)
    cpu_scaled = TiledGenerator(
        _generator(
            _TileProbabilityModel(predictions).to(device),
            cpu_diffusion,
            device=device,
        )
    )
    cuda_scaled = TiledGenerator(
        _generator(
            _TileProbabilityModel(predictions).to(device),
            cuda_diffusion,
            device=device,
        )
    )
    cpu_vol = cpu_scaled.generate(
        shape=(8, 4, 4),
        overlap=1,
        storage="cpu",
        progress=False,
    )
    cuda_vol = cuda_scaled.generate(
        shape=(8, 4, 4),
        overlap=1,
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
    scaled = TiledGenerator(_generator(model, Diffusion(1), patch_size=224))

    def reject_allocation(*args, **kwargs):
        del args, kwargs
        raise AssertionError("plan must not allocate a tensor")

    monkeypatch.setattr(torch, "empty", reject_allocation)
    plan = scaled.plan(2048, overlap=16)

    assert plan.shape == (2048, 2048, 2048)
    assert plan.overlap == 16
    assert plan.tile_size == 224
    assert plan.stride == 192
    assert plan.grid == (11, 11, 11)
    assert plan.tile_count == 1331


def test_cpu_generation_returns_cpu_uint8_without_creating_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    diffusion = _TraceDiffusion(timesteps=2)
    scaled = TiledGenerator(_generator(_PhaseModel(phase=2), diffusion))

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
    scaled = TiledGenerator(_generator(_FailModel(), Diffusion(1)))

    with pytest.raises(RuntimeError, match="failed prediction"):
        scaled.generate(
            shape=4,
            overlap=0,
            storage="cpu",
            progress=False,
        )

    assert not tuple(tmp_path.rglob("*"))


def test_model_input_matches_each_bounded_tile_source() -> None:
    model = _OverlapTraceModel()
    scaled = TiledGenerator(_generator(model, Diffusion(2)))
    plan = scaled.plan((9, 7, 5), overlap=1)
    tiles = make_tiles(plan)

    scaled.generate(
        shape=plan.shape,
        overlap=plan.overlap,
        storage="cpu",
        progress=False,
    )

    expected = [
        tuple(region.stop - region.start for region in tile.source) for tile in tiles
    ]
    observed = [tuple(call.current.shape[-3:]) for call in model.calls[: len(tiles)]]

    assert observed == expected
    assert all(max(shape) <= plan.tile_size for shape in observed)


def test_tile_targets_cover_non_divisible_shape_exactly_once() -> None:
    scaled = TiledGenerator(_generator(_OverlapTraceModel(), Diffusion(1)))
    plan = scaled.plan((9, 7, 5), overlap=1)
    tiles = make_tiles(plan)
    coverage = torch.zeros(plan.shape, dtype=torch.int32)

    for tile in tiles:
        coverage[tile.target].add_(1)
        source_shape = tuple(region.stop - region.start for region in tile.source)
        assert source_shape == (4, 4, 4)

    assert tiles[0].margins == ((0, 1), (0, 1), (0, 1))
    assert tiles[-1].margins == ((1, 0), (1, 0), (1, 0))

    assert plan.tile_size == 4
    assert plan.stride == 2
    assert plan.grid == (4, 3, 2)
    assert len(tiles) == plan.tile_count == 24
    assert torch.equal(coverage, torch.ones_like(coverage))


def test_scale_plan_uses_configured_overlap_and_patch_core() -> None:
    model = _ControlledModel()
    model.downsample_factor = 4

    plan = TiledGenerator(_generator(model, Diffusion(1), patch_size=64)).plan(
        128, overlap=16
    )

    assert plan.overlap == 16
    assert plan.tile_size == 64
    assert plan.stride == 32
    assert plan.base_shell == 8
    assert plan.grid == (3, 3, 3)
    assert plan.tile_count == 27


def test_tile_core_prediction_matches_full_non_periodic_prediction() -> None:
    model = _LocalModel()
    scaled = TiledGenerator(_generator(model, Diffusion(1)))
    plan = scaled.plan((7, 6, 6), overlap=1)
    tiles = make_tiles(plan)
    current = VolumeState(3, plan.shape, torch.device("cpu"))
    next_state = VolumeState(3, plan.shape, torch.device("cpu"))
    assert current.values.dtype == next_state.values.dtype == torch.float16
    values = torch.arange(current.values.numel(), dtype=torch.float32)
    values = values.reshape_as(current.values).remainder(17).div_(8).sub_(1)
    current.values.copy_(values)
    time = torch.zeros(1, dtype=torch.long)
    latent = torch.zeros(1, 4)
    fusion = make_fusion(
        plan,
        tiles,
        scaled.generator.num_phases,
        current.values.device,
        scaled.generator.device,
    )

    scaled.step(
        current,
        next_state,
        tiles,
        time,
        latent,
        None,
        torch.zeros(1, dtype=torch.long),
        0,
        plan,
        None,
        fusion,
    )
    expected = (
        F.avg_pool3d(current.values.float(), 3, stride=1, padding=1).half().float()
    )

    assert torch.equal(next_state.values.float(), expected)


@pytest.mark.parametrize(
    "shape,overlap", [((17, 7, 6), 1), ((15, 9, 7), 0), ((4, 4, 4), 1)]
)
def test_circular_slab_matches_full_weighted_fusion(shape, overlap):
    scaled = TiledGenerator(_generator(_LocalModel(), Diffusion(1)))
    plan = scaled.plan(shape, overlap=overlap)
    tiles = make_tiles(plan)
    current = VolumeState(3, shape, torch.device("cpu"))
    current.values.copy_(torch.randn_like(current.values))
    next_state = VolumeState(3, shape, torch.device("cpu"))
    fusion = make_fusion(plan, tiles, 3, torch.device("cpu"), torch.device("cpu"))
    assert fusion.pred_sum.shape == (1, 3, min(shape[0], 4), *shape[1:])
    expected = torch.zeros_like(current.values, dtype=torch.float32)
    weights = torch.zeros((1, 1, *shape))
    for tile in tiles:
        pred = F.avg_pool3d(current.read(tile.source).float(), 3, stride=1, padding=1)
        axes = get_axis_windows(tile, overlap, torch.device("cpu"), {})
        window = (
            axes[0][None, None, :, None, None]
            * axes[1][None, None, None, :, None]
            * axes[2][None, None, None, None, :]
        )
        region = (slice(None), slice(None), *tile.source)
        expected[region] += pred * window
        weights[region] += window
    expected.div_(weights)
    for _ in range(2):
        scaled.step(
            current,
            next_state,
            tiles,
            torch.zeros(1, dtype=torch.long),
            torch.zeros(1, 4),
            None,
            torch.zeros(1, dtype=torch.long),
            0,
            plan,
            None,
            fusion,
        )
        torch.testing.assert_close(next_state.values, expected.half())
        assert not torch.count_nonzero(fusion.weight_sum)


def test_tiled_probabilities_allow_explicit_cpu_storage():
    scaled = TiledGenerator(_generator(_ControlledModel(), Diffusion(1)))
    probs = scaled.generate_probs((6, 5, 7), overlap=1, storage="cpu", progress=False)
    assert probs.device.type == "cpu"
    torch.testing.assert_close(probs.sum(dim=0), torch.ones(6, 5, 7))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("storage", ["cpu", "cuda", "auto"])
def test_cuda_denoiser_streams_slabs_with_each_storage_mode(storage):
    device = torch.device("cuda")
    model = Denoiser3D(3, 4, [1, 2], 8, 4, 1).to(device).eval()
    generator = Generator(model, Diffusion(2).to(device), device, 8, 3, 4, True)
    scaled = TiledGenerator(generator)
    probs = scaled.generate_probs(
        (13, 10, 10), overlap=2, storage=storage, progress=False
    )
    assert probs.device.type == "cpu"
    assert torch.isfinite(probs).all()
    assert (probs >= 0).all() and (probs <= 1).all()
    torch.testing.assert_close(probs.sum(0), torch.ones(13, 10, 10))


def test_boundary_tile_reads_only_bounded_context() -> None:
    model = _OverlapTraceModel()
    scaled = TiledGenerator(_generator(model, Diffusion(1)))
    plan = scaled.plan((12, 12, 12), overlap=1)
    tiles = make_tiles(plan)
    tile = tiles[0]
    current = VolumeState(3, plan.shape, torch.device("cpu"))
    next_state = VolumeState(3, plan.shape, torch.device("cpu"))
    coordinates = torch.arange(math.prod(plan.shape), dtype=torch.float32).reshape(
        plan.shape
    )
    current.values.zero_()
    current.values[0, 0].copy_(coordinates)
    fusion = make_fusion(
        plan,
        tiles,
        scaled.generator.num_phases,
        current.values.device,
        scaled.generator.device,
    )

    scaled.step(
        current,
        next_state,
        tiles,
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, 4),
        None,
        torch.zeros(1, dtype=torch.long),
        0,
        plan,
        None,
        fusion,
    )

    expected = coordinates[tile.source]
    observed = model.calls[0].current[0, 0]

    assert tile.margins == ((0, 1), (0, 1), (0, 1))
    assert observed.shape == (4, 4, 4)
    assert torch.equal(observed, expected)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_labels_are_selected_before_cpu_transfer(monkeypatch, device):
    generator = _generator(_ControlledModel(), Diffusion(1))
    clean = torch.randn(1, 3, 4, 5, 6, device=device).softmax(1).mul(2).sub(1)
    expected = clean.argmax(1).squeeze(0).cpu().to(torch.uint8)
    monkeypatch.setattr(generator, "_sample_clean", lambda **kwargs: clean)
    monkeypatch.setattr(
        generator,
        "generate_probs",
        lambda **kwargs: pytest.fail("labels used probability path"),
    )
    transfers = []
    original = torch.Tensor.to

    def track(tensor, *args, **kwargs):
        transfers.append((tensor.shape, tensor.dtype, tensor.device.type))
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", track)
    labels = generator.generate()
    assert transfers == [(torch.Size((4, 5, 6)), torch.int64, device)]
    torch.testing.assert_close(labels, expected)


def test_direct_memory_budget_matches_returned_representation(monkeypatch):
    import src.predict.generator as module

    choices = []
    original = module.estimate_memory

    def record(*args, **kwargs):
        choices.append(kwargs["probabilities"])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "estimate_memory", record)
    generator = _generator(_ControlledModel(), Diffusion(1))
    generator.generate()
    generator.generate_probs()
    assert choices == [False, True]


def test_bounded_tile_reads_reuse_the_workspace() -> None:
    state = VolumeState(3, (6, 7, 5), torch.device("cpu"))
    values = torch.arange(state.values.numel(), dtype=torch.float32)
    state.values.copy_(values.reshape_as(state.values))
    buffer = TileBuffer(3, 8, enabled=False)
    first_region = (slice(0, 6), slice(0, 7), slice(0, 5))
    second_region = (
        slice(1, 5),
        slice(2, 7),
        slice(0, 5),
    )

    first = buffer.read(state, first_region, torch.device("cpu"))
    second = buffer.read(state, second_region, torch.device("cpu"))

    assert first.data_ptr() == second.data_ptr()
    assert torch.equal(second, state.read(second_region).float())


def test_scaled_generation_supports_anisotropic_shape() -> None:
    scaled = TiledGenerator(_generator(_TraceModel(), _TraceDiffusion(timesteps=1)))
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
    assert stats.seams == ((3,), (), (4,))


def test_scaled_generation_does_not_force_the_base_into_the_output() -> None:
    diffusion = _TraceDiffusion(timesteps=1)
    scaled = TiledGenerator(_generator(_PhaseModel(phase=0), diffusion))
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
    scaled = TiledGenerator(_generator(_TraceModel(), Diffusion(1)))
    base = torch.zeros((4, 4, 4), dtype=torch.uint8)
    plan = scaled.plan((6, 4, 4), overlap=0)

    condition = scaled.prepare_base(base, plan)

    assert condition is not None
    assert condition.region == (slice(1, 5), slice(0, 4), slice(0, 4))
    assert torch.all(condition.weight == 1.0)


def test_base_uses_cosine_transition_only_on_expanded_axes() -> None:
    scaled = TiledGenerator(_generator(_TraceModel(), Diffusion(1), patch_size=8))
    plan = scaled.plan((12, 8, 8), overlap=2)

    condition = scaled.prepare_base(
        torch.zeros((8, 8, 8), dtype=torch.uint8),
        plan,
    )

    assert condition is not None
    expected = torch.tensor((0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5))
    torch.testing.assert_close(condition.weight[0, 0, :, 4, 4], expected)
    assert torch.all(condition.weight[0, 0, 4] == 1.0)


def test_base_condition_blends_the_transition() -> None:
    scaled = TiledGenerator(_generator(_TraceModel(), Diffusion(1), patch_size=8))
    plan = scaled.plan((12, 8, 8), overlap=2)
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
    scaled = TiledGenerator(generator)

    probs = scaled.generate_probs(
        shape=(12, 8, 8),
        overlap=2,
        base=base,
        progress=False,
    )
    vol = scaled.generate(
        shape=(12, 8, 8),
        overlap=2,
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
    scaled = TiledGenerator(_generator(_TraceModel(), Diffusion(1)))
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


def test_scaled_base_offset_uses_output_coordinates_outside_generation_margin() -> None:
    scaled = TiledGenerator(_generator(_TraceModel(), Diffusion(1), patch_size=8))
    plan = scaled._generation_plan((12, 12, 12), overlap=2, margin=2)

    condition = scaled.prepare_base(
        torch.zeros((8, 8, 8), dtype=torch.uint8),
        plan,
        offset=(0, None, 4),
    )

    assert condition is not None
    assert condition.region == (slice(2, 10), slice(4, 12), slice(6, 14))
    assert condition.weight[0, 0, 0, 4, 4] == 1
    assert condition.weight[0, 0, -1, 4, 4] < 1
    assert condition.weight[0, 0, 4, 4, 0] < 1
    assert condition.weight[0, 0, 4, 4, -1] == 1


def test_scaled_base_offset_rejects_out_of_bounds_and_requires_base() -> None:
    scaled = TiledGenerator(_generator(_TraceModel(), Diffusion(1)))
    plan = scaled.plan((6, 6, 6), overlap=0)

    with pytest.raises(ValueError, match="axis 0 must be between 0 and 2"):
        scaled.prepare_base(
            torch.zeros((4, 4, 4), dtype=torch.uint8),
            plan,
            offset=(3, None, None),
        )
    with pytest.raises(ValueError, match="requires base"):
        scaled.prepare_base(None, plan, offset=(0, None, None))


def test_scaled_base_keeps_constant_prediction_in_every_core() -> None:
    model = _PhaseModel(phase=0)
    diffusion = _TraceDiffusion(timesteps=3)
    base = torch.zeros((4, 4, 4), dtype=torch.uint8)

    vol = TiledGenerator(_generator(model, diffusion)).generate(
        shape=(6, 6, 6),
        base=base,
        overlap=0,
        progress=False,
    )

    assert model.call_count == 3 * 8
    assert (
        sum(math.prod(call.current_shape[-3:]) for call in diffusion.calls) == 2 * 6**3
    )
    for call in diffusion.calls:
        assert torch.all(call.clean[:, 0] == 1.0)
        assert torch.all(call.clean[:, 1:] == -1.0)
    assert isinstance(vol, torch.Tensor)
    assert torch.all(vol == 0)


def test_scaled_generation_conditions_every_step_with_one_base_noise() -> None:
    model = _TraceModel()
    diffusion = _NoiseTraceDiffusion(timesteps=3)

    TiledGenerator(_generator(model, diffusion)).generate(
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
    scaled = TiledGenerator(_generator(_PhaseModel(phase=0), Diffusion(1)))

    vol = scaled.generate(
        shape=(4, 4, 4),
        base=base,
        overlap=0,
        progress=False,
    )

    assert torch.all(vol == 0)


def test_zero_overlap_does_not_force_the_whole_base() -> None:
    base = torch.arange(4 * 4 * 4).reshape(4, 4, 4).remainder(3).to(torch.uint8)
    scaled = TiledGenerator(_generator(_PhaseModel(phase=0), Diffusion(1)))

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
        (torch.zeros((7, 4, 4), dtype=torch.uint8), "shape"),
        (torch.zeros((4, 4, 4), dtype=torch.float32), "integer"),
        (torch.full((4, 4, 4), 3, dtype=torch.uint8), "phase"),
    ),
)
def test_scaled_generation_rejects_invalid_base(
    base: torch.Tensor,
    error: str,
) -> None:
    scaled = TiledGenerator(_generator(_TraceModel(), Diffusion(1)))

    with pytest.raises(ValueError, match=error):
        scaled.generate(shape=6, overlap=0, base=base, progress=False)


def test_repeated_extension_preserves_arbitrary_base_and_accepts_new_anchor():
    model = _AnchorTraceModel()
    scaled = TiledGenerator(_generator(model, Diffusion(2)))
    base = torch.ones(5, 4, 4, dtype=torch.uint8)
    before = base.clone()
    extra = PlaneAnchor(torch.full((4, 4), 2, dtype=torch.uint8), 0, 7, (0, 0))
    first = scaled.generate(
        shape=(8, 4, 4),
        base=base,
        base_offset=(0, 0, 0),
        preserve_base=True,
        anchors=(extra,),
        vf=(0.3, 0.3, 0.4),
        overlap=0,
        progress=False,
    )
    assert torch.equal(first[:5], base)
    assert any(
        call.anchor_mask is not None and call.anchor_mask.any() for call in model.calls
    )
    assert all(call.vf is not None for call in model.calls)
    again = scaled.generate(
        shape=(10, 4, 4),
        base=first,
        base_offset=(1, 0, 0),
        preserve_base=True,
        overlap=0,
        progress=False,
    )
    assert torch.equal(again[1:9], first)
    assert torch.equal(base, before)


def test_preserved_base_rejects_conflicting_measurements():
    scaled = TiledGenerator(_generator(_PhaseModel(phase=0), Diffusion(1)))
    base = torch.ones(4, 4, 4, dtype=torch.uint8)
    anchor = PlaneAnchor(torch.zeros(4, 4, dtype=torch.uint8), 1, 2, (0, 0))
    with pytest.raises(ValueError, match="conflicts"):
        scaled.generate(
            shape=(8, 4, 4),
            base=base,
            base_offset=(0, 0, 0),
            preserve_base=True,
            anchors=(anchor,),
            overlap=0,
            progress=False,
        )


def test_known_tiles_skip_model_and_fractional_base_is_retained():
    model = _TraceModel()
    scaled = TiledGenerator(_generator(model, Diffusion(2)))
    base = (
        torch.tensor([0.125, 0.375, 0.5])[:, None, None, None]
        .expand(3, 8, 4, 4)
        .clone()
    )
    before = base.clone()
    result = scaled.generate_probs(
        shape=(8, 4, 4), base=base, preserve_base=True, overlap=0, progress=False
    )
    assert torch.allclose(result, base, atol=5e-4)
    assert torch.equal(base, before)
    assert not model.calls


@pytest.mark.parametrize("vf", (None, (0.5, 0.1, 0.4)))
def test_single_core_matches_regular_categorical_prediction(
    vf: tuple[float, ...] | None,
) -> None:
    expected = _generator(_PhaseModel(phase=2), Diffusion(1)).generate(vf=vf)
    scaled = TiledGenerator(_generator(_PhaseModel(phase=2), Diffusion(1)))
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


def test_scaled_generation_validates_shape_overlap_and_storage() -> None:
    generator = _generator(_TraceModel(), _TraceDiffusion(timesteps=1))

    with pytest.raises(TypeError, match="shape must be"):
        TiledGenerator(generator).generate(shape="2", overlap=0, progress=False)
    with pytest.raises(ValueError, match="three positive integers"):
        TiledGenerator(generator).generate(shape=(2, 0, 1), overlap=0, progress=False)
    with pytest.raises(ValueError, match="overlap"):
        TiledGenerator(generator).generate(shape=4, overlap=-1, progress=False)
    with pytest.raises(ValueError, match="storage must be"):
        TiledGenerator(generator).generate(
            shape=4,
            overlap=0,
            storage="mmap",
            progress=False,
        )
    vol = TiledGenerator(generator).generate(
        shape=4,
        overlap=0,
        storage="cpu",
        progress=False,
    )
    assert vol.device.type == "cpu"
    with pytest.raises(ValueError, match="requires a CUDA"):
        TiledGenerator(generator).generate(
            shape=4,
            overlap=0,
            storage="cuda",
            progress=False,
        )


def test_scale_quality_skips_axes_without_seams() -> None:
    vol = torch.arange(12 * 8 * 8, dtype=torch.long).reshape(12, 8, 8) % 3

    quality = measure_seams(
        vol.to(torch.uint8),
        ((6,), (), ()),
        4,
        3,
    )

    assert quality.transition_tv[0] is not None
    assert quality.transition_tv[1:] == (None, None)
    assert quality.continuation_delta[0] is not None
    assert quality.continuation_delta[1:] == (None, None)


@dataclass(frozen=True)
class _ModelCall:
    transition: int
    timestep: torch.Tensor
    latent: torch.Tensor
    current: torch.Tensor
    current_ptr: int
    domain: torch.Tensor
    vf: torch.Tensor | None


@dataclass(frozen=True)
class _AnchorModelCall:
    transition: int
    timestep: torch.Tensor
    latent: torch.Tensor
    current_ptr: int
    vf: torch.Tensor | None
    anchor_image_id: int | None
    anchor_mask_id: int | None
    anchor_mask: torch.Tensor | None
    current: torch.Tensor


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
        domain: torch.Tensor,
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
                domain=domain,
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
        domain: torch.Tensor,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
        vf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self.compute_logits(
            current,
            timestep,
            latent,
            domain=domain,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
            vf=vf,
        )
        return Denoiser3D.decode(logits)

    def compute_logits(
        self,
        current: torch.Tensor,
        timestep: torch.Tensor,
        latent: torch.Tensor,
        *,
        domain: torch.Tensor,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
        vf: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls.append(
            _AnchorModelCall(
                transition=int(timestep.item()),
                timestep=timestep,
                latent=latent,
                current_ptr=current.untyped_storage().data_ptr(),
                vf=vf,
                anchor_image_id=None if anchor_image is None else id(anchor_image),
                anchor_mask_id=None if anchor_mask is None else id(anchor_mask),
                anchor_mask=(
                    None if anchor_mask is None else anchor_mask.detach().clone()
                ),
                current=current.detach().clone(),
            )
        )
        anchor_bias = 0.0 if anchor_image is None else 0.05 * anchor_image
        return 0.25 * current + anchor_bias


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
        domain: torch.Tensor,
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
        domain: torch.Tensor,
        vf: torch.Tensor | None = None,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del anchor_image, anchor_mask
        return super().forward(current, timestep, latent, domain=domain, vf=vf)

    def compute_logits(
        self,
        current: torch.Tensor,
        timestep: torch.Tensor,
        latent: torch.Tensor,
        *,
        domain: torch.Tensor,
        vf: torch.Tensor | None = None,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward(
            current,
            timestep,
            latent,
            domain=domain,
            vf=vf,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
        )


class _LocalModel(torch.nn.Module):
    downsample_factor = 1

    def forward(
        self,
        current: torch.Tensor,
        timestep: torch.Tensor,
        latent: torch.Tensor,
        *,
        domain: torch.Tensor,
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
        domain: torch.Tensor,
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
        domain: torch.Tensor,
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
        domain: torch.Tensor,
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
        domain: torch.Tensor,
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
        domain: torch.Tensor,
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

    compute_logits = forward


class _GuidanceTraceModel(_ControlledModel):
    def __init__(self) -> None:
        super().__init__()
        self.guidances: list[float] = []
        self.guidance_inputs: list[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
            ]
        ] = []

    def apply_guidance_logits(
        self,
        current: torch.Tensor,
        timestep: torch.Tensor,
        latent: torch.Tensor,
        guidance: float,
        domain: torch.Tensor,
        vf: torch.Tensor | None = None,
        anchor_image: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del anchor_image, anchor_mask
        self.guidances.append(guidance)
        self.guidance_inputs.append((current, timestep, latent, vf))
        return self.compute_logits(current, timestep, latent, domain=domain, vf=vf)


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
    ) -> torch.Tensor:
        self.sample_calls += 1
        return super().sample(
            model,
            initial_noise,
            latent_channels,
            conditions=conditions,
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
    if not hasattr(model, "downsample_factor"):
        model.downsample_factor = 1
    if not hasattr(model, "num_domains"):
        model.num_domains = 1
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
            "domains": {
                0: {("xy", "xz", "yz")[axis]: [root / str(axis)] for axis in (0, 1, 2)}
            },
            "crop_size": 8,
            "num_phases": 3,
            "lo_res_size": 8,
        },
        "model": {
            "generator": {
                "channels": [4, 8],
                "latent_channels": 4,
                "embedding_channels": 8,
                "anchor_multiscale_input": False,
            },
            "critic": {
                "channels": [4, 8],
                "plane_groups": [
                    [plane]
                    for plane in ("xy", "xz", "yz")
                    if any(
                        (
                            plane in planes
                            for planes in {
                                0: {
                                    ("xy", "xz", "yz")[axis]: [root / str(axis)]
                                    for axis in (0, 1, 2)
                                }
                            }.values()
                        )
                    )
                ],
            },
            "gradient_checkpointing": False,
            "diffusion": {"num_steps": 2, "beta_min": 0.1, "beta_max": 2.0},
        },
        "optim": {
            "generator_lr": 0.001,
            "critic_lr": 0.001,
            "adam_betas": [0.0, 0.9],
            "ema_decay": 0.9,
        },
        "train": {
            "volume_batch_size": 1,
            "real_batch_size": 2,
            "num_workers": 0,
            "total_steps": 10,
            "mixed_precision": False,
            "slice_pairs_per_plane": 2,
            "initial_weights": None,
            "weights_every_steps": 1,
            "archive_every_steps": 10,
        },
        "conditioning": {
            "domain_keep_probability": 1.0,
            "anchor": {
                "probability": 0.0,
                "start_step": 0,
                "ramp_steps": 0,
                "borrowed_plane_probability": 0.0,
            },
            "dropout_probability_per_case": 0.0,
        },
        "augmentation": {
            "probability": 0.0,
            "planes": {
                "xy": {"flip_axes": [], "rotate_90": False},
                "xz": {"flip_axes": [], "rotate_90": False},
                "yz": {"flip_axes": [], "rotate_90": False},
            },
        },
        "loss": {
            "anchor_pixel_weight": 0.05,
            "connectivity": {
                "adversarial_weight": 0.0,
                "normal_transition_weight": 0.0,
            },
            "volume_fraction_weight": 1.0,
            "critic_local_weight": 0.5,
            "r1_weight": 0.0,
            "r1_every_steps": 2,
        },
    }


@pytest.mark.parametrize("guidance", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("tiled", [False, True])
def test_generation_rejects_nonfinite_guidance(guidance, tiled):
    generator = _generator(_GuidanceTraceModel(), _TraceDiffusion(timesteps=1))
    with pytest.raises(ValueError, match="guidance"):
        if tiled:
            TiledGenerator(generator).generate(
                shape=generator.patch_size, overlap=0, guidance=guidance, progress=False
            )
        else:
            generator.generate(guidance=guidance)


def test_tiled_plan_rejects_boolean_overlap():
    generator = _generator(_TraceModel(), _TraceDiffusion(timesteps=1))
    with pytest.raises(ValueError, match="overlap"):
        TiledGenerator(generator).plan(generator.patch_size, overlap=True)
