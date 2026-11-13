from functools import partial

import pytest
import torch
import torch.nn.functional as F

from src.model.critic import CriticScores, PairCritic2D
from src.model.denoiser import ChannelNormSiLU3D, Denoiser3D, channel_norm_silu

MPS = torch.backends.mps.is_available()


def _denoiser(**kwargs):
    options = dict(
        num_phases=3,
        base_channels=4,
        channel_multipliers=(1, 2),
        embedding_channels=8,
        latent_channels=4,
    )
    return Denoiser3D(**{**options, **kwargs})


def _critic(**kwargs):
    return PairCritic2D(num_phases=3, channels=(4, 8), embedding_channels=8, **kwargs)


def _inputs(batch=1, size=(4, 4, 4), phases=3):
    torch.manual_seed(0)
    return (
        torch.randn(batch, phases, *size),
        torch.arange(batch),
        torch.randn(batch, 4),
    )


def test_channel_norm_preserves_large_half_precision_features():
    inputs = torch.tensor([-1000.0, 500.0, 1000.0, -500.0], dtype=torch.float16)
    inputs = inputs.reshape(1, 4, 1, 1, 1).requires_grad_()
    norm = ChannelNormSiLU3D(4)
    output = norm(inputs)
    expected = norm(inputs.detach().float()).to(inputs.dtype)

    assert output.dtype == inputs.dtype
    assert torch.allclose(output, expected, atol=1e-3, rtol=1e-3)
    output[:, 0].sum().backward()
    assert bool(torch.isfinite(inputs.grad).all())
    assert bool((inputs.grad != 0).any())


@pytest.mark.skipif(not MPS, reason="MPS unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_apple_gpu_fused_norm_matches_the_eager_one(dtype):
    torch.manual_seed(0)
    inputs = torch.randn(2, 8, 6, 7, 9, device="mps", dtype=dtype) * 3
    scale = torch.randn(2, 8, 1, 1, 1, device="mps", dtype=dtype) + 1
    shift = torch.randn(2, 8, 1, 1, 1, device="mps", dtype=dtype)
    weights = torch.randn_like(inputs)
    results = []
    module = ChannelNormSiLU3D(8, affine=False).to("mps")
    for norm in (module, partial(channel_norm_silu, eps=module.eps)):
        values = [v.clone().requires_grad_() for v in (inputs, scale, shift)]
        output = norm(*values)
        (output.float() * weights.float()).sum().backward()
        results.append([output, *(v.grad for v in values)])
    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    for fused, eager in zip(*results):
        assert fused.dtype == eager.dtype == dtype
        torch.testing.assert_close(fused, eager, rtol=tolerance, atol=tolerance)


@pytest.mark.skipif(not MPS, reason="MPS unavailable")
def test_apple_gpu_fused_norm_compiles_once_per_grad_mode():
    # fp32 affine weights, fp16 modulation and frozen EMA weights share one
    # kernel per grad mode instead of using up the recompile limit.
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    affine = ChannelNormSiLU3D(8).to("mps")
    adaptive = ChannelNormSiLU3D(8, affine=False).to("mps")
    x = torch.randn(2, 8, 4, 4, 4, device="mps", dtype=torch.float16)
    modulation = torch.randn(
        1, 8, 1, 1, 1, device="mps", dtype=torch.float16, requires_grad=True
    )
    for grad in (True, False):
        with torch.set_grad_enabled(grad):
            affine(x)
            adaptive(x, modulation, modulation)
    affine.requires_grad_(False)
    with torch.no_grad():
        affine(x)
    assert counters["stats"]["unique_graphs"] == 2


def test_3d_norm_is_independent_at_each_spatial_position():
    norm = ChannelNormSiLU3D(4)
    inputs = torch.randn(2, 4, 3, 4, 5)
    changed = inputs.clone()
    changed[:, :, -1, -1, -1].add_(100.0 * torch.randn(2, 4))
    output, changed_output = norm(inputs), norm(changed)
    assert torch.equal(output[:, :, 0, 0, 0], changed_output[:, :, 0, 0, 0])
    normalized = F.layer_norm(inputs.movedim(1, -1), (4,)).movedim(-1, 1)
    assert torch.allclose(output, F.silu(normalized), atol=1.0e-6)
    assert norm(inputs.half()).dtype == torch.float16


def test_denoiser_rejects_a_single_normalized_channel():
    with pytest.raises(ValueError, match="at least two"):
        Denoiser3D(3, 1, (1, 2), 8, 4)


def test_denoiser_downsamples_by_two():
    # The LR API's default margin.
    assert _denoiser().downsample_factor == 2


def test_clean_output_is_a_latent_conditioned_simplex_with_gradients():
    model = _denoiser()
    inputs = torch.randn(2, 3, 4, 6, 8, requires_grad=True)
    time = torch.tensor([0, 1])
    latent = torch.ones(2, 4, requires_grad=True)
    first = model(inputs, time, torch.zeros(2, 4))
    clean = model(inputs, time, latent)
    probs = (clean + 1.0) / 2.0
    assert clean.shape == inputs.shape
    assert -1.0 <= float(clean.detach().min()) <= float(clean.detach().max()) <= 1.0
    torch.testing.assert_close(probs.sum(1), torch.ones_like(probs[:, 0]))
    assert not torch.equal(first, clean)
    clean.square().mean().backward()
    for grad in (inputs.grad, latent.grad):
        assert bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0


def test_forward_is_the_body_of_the_embedding():
    model = _denoiser(height_enabled=True, profile_enabled=True).eval()
    x, time, latent = _inputs(2)
    vf = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]])
    present = torch.tensor([True, False])
    conditions = dict(
        height=torch.rand(2, 1, 4, 4, 4),
        profile=torch.full((2, 3, 4), 1 / 3),
        profile_present=present,
    )
    with torch.no_grad():
        logits = model.logits(x, time, latent, vf=vf, vf_present=present, **conditions)
        emb = model.embed(time, latent, vf, present)
        body = model.forward_body(x, emb, **conditions)
    torch.testing.assert_close(logits, body)


def test_vf_presence_mask_matches_the_null_and_conditioned_paths():
    model = _denoiser()
    with torch.no_grad():
        for parameter in model.vf_mlp.parameters():
            parameter.normal_()
    _, time, latent = _inputs(2)
    vf = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]])
    masked = model.embed(time, latent, vf, torch.tensor([True, False]))
    torch.testing.assert_close(masked[0], model.embed(time, latent, vf)[0])
    torch.testing.assert_close(masked[1], model.embed(time, latent)[1])


def test_masked_vf_rows_receive_no_gradient():
    model = _denoiser()
    with torch.no_grad():
        for parameter in model.vf_mlp.parameters():
            parameter.normal_()
    _, time, latent = _inputs(2)
    vf = torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]], requires_grad=True)
    model.embed(time, latent, vf, torch.tensor([True, False])).sum().backward()
    assert float(vf.grad[0].abs().sum()) > 0
    assert torch.equal(vf.grad[1], torch.zeros(3))


def test_zero_initialized_vf_adapter_keeps_the_unconditioned_output():
    model = _denoiser()
    x, time, latent = _inputs()
    vf = torch.tensor([[0.2, 0.3, 0.5]], requires_grad=True)
    torch.testing.assert_close(model(x, time, latent, vf=vf), model(x, time, latent))
    model(x, time, latent, vf=vf).square().sum().backward()
    assert model.vf_mlp[-1].weight.grad.abs().sum() > 0


def test_guidance_blends_the_unconditional_branch_without_vf_and_profile():
    model = _denoiser(height_enabled=True, profile_enabled=True).eval()
    with torch.no_grad():
        for parameter in model.vf_mlp.parameters():
            parameter.normal_()
    x, time, latent = _inputs()
    height = torch.rand(1, 1, 4, 4, 4)
    conditioned = dict(
        vf=torch.tensor([[0.2, 0.3, 0.5]]),
        profile=torch.full((1, 3, 4), 1 / 3),
        height=height,
    )
    with torch.no_grad():
        conditional = model.logits(x, time, latent, **conditioned)
        unconditional = model.logits(x, time, latent, height=height)
        for guidance in (0.0, 0.5, 1.0, 2.0):
            expected = unconditional + guidance * (conditional - unconditional)
            torch.testing.assert_close(
                model.guided_logits(x, time, latent, guidance, **conditioned),
                expected,
            )
        # Without vf and profile there is nothing to drop.
        torch.testing.assert_close(
            model.guided_logits(x, time, latent, 0.0, height=height), unconditional
        )


def test_guidance_takes_one_pass_without_a_guided_condition(monkeypatch):
    model = _denoiser()
    calls = []
    original = model.logits
    monkeypatch.setattr(
        model, "logits", lambda *a, **k: calls.append(1) or original(*a, **k)
    )
    x, time, latent = _inputs()
    model.guided_logits(x, time, latent, 0.9)
    model.guided_logits(x, time, latent, 1.0, vf=torch.full((1, 3), 1 / 3))
    assert len(calls) == 2


def test_sr_denoiser_takes_coarse_fractions_and_rejects_lr_conditions():
    model = _denoiser(coarse_enabled=True).eval()
    x, time, latent = _inputs()
    coarse = torch.full_like(x, 1 / 3)
    with torch.no_grad():
        clean = model(x, time, latent, coarse=coarse)
        corrupted = model(
            x, time, latent, coarse=coarse, corruption_level=torch.tensor([0.5])
        )
    assert clean.shape == x.shape and not torch.equal(clean, corrupted)
    assert model.vf_mlp is None
    with pytest.raises(ValueError, match="volume fractions"):
        model(x, time, latent, coarse=coarse, vf=torch.full((1, 3), 1 / 3))
    with pytest.raises(ValueError, match="coarse"):
        model(x, time, latent)
    with pytest.raises(ValueError, match="coarse conditioning requires"):
        _denoiser()(x, time, latent, coarse=coarse)


def test_height_and_profile_inputs_are_checked():
    model = _denoiser(height_enabled=True, profile_enabled=True)
    x, time, latent = _inputs()
    with pytest.raises(ValueError, match="height field"):
        model(x, time, latent)
    with pytest.raises(ValueError, match="profile must match"):
        model(
            x,
            time,
            latent,
            height=torch.zeros(1, 1, 4, 4, 4),
            profile=torch.full((1, 3, 5), 1 / 3),
        )
    with pytest.raises(ValueError, match="height conditioning is disabled"):
        _denoiser()(x, time, latent, height=torch.zeros(1, 1, 4, 4, 4))
    with pytest.raises(ValueError, match="profile conditioning is disabled"):
        _denoiser()(x, time, latent, profile=torch.full((1, 3, 4), 1 / 3))


def test_single_voxel_bottleneck_is_supported():
    model = _denoiser(channel_multipliers=(1, 2, 4, 8))
    output = model(torch.randn(1, 3, 8, 8, 8), torch.zeros(1), torch.zeros(1, 4))
    assert output.shape == (1, 3, 8, 8, 8)


def test_critic_scores_pairs_at_every_pyramid_level():
    critic = _critic(pyramid_min_size=4)
    previous = torch.randn(2, 3, 16, 16, requires_grad=True)
    current = torch.randn(2, 3, 16, 16, requires_grad=True)
    time = torch.tensor([0, 9])
    scores = critic(previous, current, time)
    assert isinstance(scores, CriticScores)
    assert scores.logits_global.shape == (2,)
    assert [level.logits_global.shape for level in scores.levels] == [(2,)] * 3
    assert not torch.equal(
        scores.logits_global, critic(previous, current, time.flip(0)).logits_global
    )
    sum(level.logits_global.sum() for level in scores.levels).backward()
    assert float(current.grad.abs().sum()) > 0


def test_critic_requires_two_feature_levels():
    with pytest.raises(ValueError, match="two levels"):
        PairCritic2D(3, (4,), 8)


def test_height_critic_scores_xy_sections_without_a_height_field():
    critic = _critic(height_channels=1)
    x = torch.randn(1, 3, 8, 8)
    time = torch.zeros(1, dtype=torch.long)
    plain = critic(x, x, time).logits_global
    assert plain.shape == (1,)
    assert not torch.equal(
        plain, critic(x, x, time, torch.rand(1, 1, 8, 8)).logits_global
    )


def test_gradient_checkpointing_keeps_outputs_and_gradients():
    torch.manual_seed(0)
    results = []
    for checkpointing in (False, True):
        torch.manual_seed(1)
        model = _denoiser(gradient_checkpointing=checkpointing).train()
        critic = _critic(gradient_checkpointing=checkpointing).train()
        x, time, latent = _inputs()
        out = model(x, time, latent)
        scores = critic(out[:, :, 0], out[:, :, 1], time)
        scores.logits_global.sum().backward()
        results.append(
            [
                out.detach(),
                *(p.grad.clone() for p in model.parameters() if p.grad is not None),
            ]
        )
    for plain, checkpointed in zip(*results, strict=True):
        torch.testing.assert_close(plain, checkpointed)
