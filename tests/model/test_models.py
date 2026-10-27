import unittest
from functools import partial

import pytest
import torch
import torch.nn.functional as F

from src.build.model import build_denoiser
from src.config.train import load_train_config
from src.model.critic import CriticScores, GroupNormSiLU, PairCritic2D
from src.model.denoiser import ChannelNormSiLU3D, Denoiser3D, channel_norm_silu
from src.model.layers import AdaptiveNorm

MPS = torch.backends.mps.is_available()


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


def _denoiser(
    *,
    checkpointing: bool = False,
) -> Denoiser3D:
    return Denoiser3D(
        num_phases=3,
        base_channels=4,
        channel_multipliers=(1, 2),
        embedding_channels=8,
        latent_channels=4,
        num_domains=2,
        gradient_checkpointing=checkpointing,
    )


def _critic(*, checkpointing: bool = False) -> PairCritic2D:
    return PairCritic2D(
        num_phases=3,
        channels=(4, 8),
        embedding_channels=8,
        num_domains=2,
        gradient_checkpointing=checkpointing,
    )


def _domain(inputs: torch.Tensor, value: int = 0) -> torch.Tensor:
    return torch.full(
        (inputs.shape[0],),
        value,
        device=inputs.device,
        dtype=torch.long,
    )


class Denoiser3DTest(unittest.TestCase):
    def test_denoiser_rejects_a_single_normalized_channel(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            Denoiser3D(3, 1, (1, 2), 8, 4, 1)

    def test_lr_and_sr_builders_reject_degenerate_channel_settings(self):
        for stage in ("low_res", "sr"):
            with self.subTest(stage=stage):
                cfg = load_train_config(
                    f"tests/fixtures/config/train/{stage}.yaml", stage
                )
                cfg["model"]["generator"]["channels"] = [1, 2]
                with self.assertRaisesRegex(ValueError, "generator.channels"):
                    build_denoiser(cfg)

    def test_smallest_supported_channels_keep_input_gradients(self):
        torch.manual_seed(9)
        model = Denoiser3D(3, 2, (1, 2), 8, 4, 1)
        inputs = torch.randn(1, 3, 4, 4, 4, requires_grad=True)
        time, latent, domain = torch.zeros(1), torch.zeros(1, 4), _domain(inputs)
        output = model(inputs, time, latent, domain)
        changed = model(-inputs, time, latent, domain)
        self.assertFalse(torch.equal(output, changed))
        output[:, 0].mean().backward()
        self.assertTrue(bool(torch.isfinite(inputs.grad).all()))
        self.assertGreater(float(inputs.grad.abs().sum()), 0)

    def test_profile_input_matches_scaled_first_level_channels(self):
        model = Denoiser3D(
            3, 2, (2, 4), 8, 4, 1, height_enabled=True, profile_enabled=True
        )
        inputs = torch.randn(1, 3, 4, 4, 4)
        time, latent, domain = torch.zeros(1), torch.zeros(1, 4), _domain(inputs)
        output = model(
            inputs,
            time,
            latent,
            domain,
            height=torch.zeros(1, 1, 4, 4, 4),
            profile=torch.full((1, 3, 4), 1 / 3),
        )
        self.assertEqual(output.shape, inputs.shape)

    def test_3d_norm_is_independent_at_each_spatial_position(self):
        norm = ChannelNormSiLU3D(4)
        inputs = torch.randn(2, 4, 3, 4, 5)
        changed = inputs.clone()
        changed[:, :, -1, -1, -1].add_(100.0 * torch.randn(2, 4))

        output = norm(inputs)
        changed_output = norm(changed)

        self.assertTrue(
            torch.equal(
                output[:, :, 0, 0, 0],
                changed_output[:, :, 0, 0, 0],
            )
        )
        normalized = F.layer_norm(inputs.movedim(1, -1), (4,)).movedim(-1, 1)
        self.assertTrue(torch.allclose(output, F.silu(normalized), atol=1.0e-6))
        self.assertEqual(norm(inputs.half()).dtype, torch.float16)

    def test_3d_model_uses_channel_norm(self):
        model = _denoiser()

        self.assertIsInstance(model.encoder[0].norm1, AdaptiveNorm)
        self.assertIsInstance(model.encoder[0].norm1.norm, ChannelNormSiLU3D)
        self.assertIsInstance(model.output_norm, ChannelNormSiLU3D)
        self.assertEqual(model.downsample_factor, 2)

    def test_clean_output_is_a_latent_conditioned_simplex_with_gradients(self):
        model = _denoiser()
        inputs = torch.randn(2, 3, 4, 6, 8, requires_grad=True)
        time = torch.tensor([0.0, 1.0])
        first_latent = torch.zeros(2, 4)
        second_latent = torch.ones(2, 4, requires_grad=True)
        domain = _domain(inputs)

        first = model(inputs, time, first_latent, domain)
        clean = model(inputs, time, second_latent, domain)
        probs = (clean + 1.0) / 2.0

        self.assertEqual(clean.shape, inputs.shape)
        self.assertGreaterEqual(float(clean.detach().min()), -1.0)
        self.assertLessEqual(float(clean.detach().max()), 1.0)
        self.assertTrue(
            torch.allclose(
                probs.sum(dim=1),
                torch.ones_like(probs[:, 0]),
                atol=1.0e-6,
            )
        )
        self.assertFalse(torch.equal(first, clean))

        clean.square().mean().backward()

        self.assertIsNotNone(inputs.grad)
        self.assertIsNotNone(second_latent.grad)
        self.assertTrue(bool(torch.isfinite(inputs.grad).all()))
        self.assertTrue(bool(torch.isfinite(second_latent.grad).all()))
        self.assertGreater(float(inputs.grad.abs().sum()), 0.0)
        self.assertGreater(float(second_latent.grad.abs().sum()), 0.0)

    def test_domain_conditioning_changes_the_prediction(self):
        model = _denoiser()
        inputs = torch.randn(1, 3, 4, 4, 4)
        time = torch.zeros(1)
        latent = torch.randn(1, 4)

        first = model(inputs, time, latent, _domain(inputs, 0))
        second = model(inputs, time, latent, _domain(inputs, 1))

        self.assertFalse(torch.equal(first, second))

    def test_negative_one_masks_the_domain_embedding(self):
        model = _denoiser()
        inputs = torch.randn(1, 3, 4, 4, 4)
        time = torch.zeros(1)
        latent = torch.randn(1, 4)
        null_domain = _domain(inputs, -1)

        first = model(inputs, time, latent, null_domain)
        with torch.no_grad():
            model.domain_embedding.weight.add_(100.0)
        second = model(inputs, time, latent, null_domain)

        torch.testing.assert_close(first, second)

    def test_group_norm_accepts_a_single_voxel_bottleneck(self):
        model = Denoiser3D(
            num_phases=3,
            base_channels=4,
            channel_multipliers=(1, 2, 4, 8),
            embedding_channels=8,
            latent_channels=4,
            num_domains=2,
        )

        output = model(
            torch.randn(1, 3, 8, 8, 8),
            torch.zeros(1),
            torch.zeros(1, 4),
            _domain(torch.empty(1)),
        )

        self.assertEqual(output.shape, torch.Size([1, 3, 8, 8, 8]))

    def test_logit_guidance_preserves_default_and_combines_conditions(self):
        model = _denoiser()
        inputs = torch.randn(1, 3, 4, 4, 4)
        time = torch.ones(1)
        latent = torch.randn(1, 4)
        vf = torch.tensor([[0.2, 0.3, 0.5]])
        domain = _domain(inputs)
        with torch.no_grad():
            model.vf_mlp[-1].weight.fill_(0.25)
            model.vf_mlp[-1].bias.fill_(0.1)

        unconditional = model.compute_logits(inputs, time, latent, domain)
        conditional = model.compute_logits(inputs, time, latent, domain, vf=vf)
        expected = model.decode(unconditional + 2.0 * (conditional - unconditional))

        default_logits = model.apply_guidance_logits(
            inputs,
            time,
            latent,
            guidance=1.0,
            domain=domain,
            vf=vf,
        )
        default = model.decode(default_logits)
        guided_logits = model.apply_guidance_logits(
            inputs,
            time,
            latent,
            guidance=2.0,
            domain=domain,
            vf=vf,
        )
        guided = model.decode(guided_logits)
        disabled_logits = model.apply_guidance_logits(
            inputs,
            time,
            latent,
            guidance=0.0,
            domain=domain,
            vf=vf,
        )
        disabled = model.decode(disabled_logits)

        self.assertTrue(
            torch.equal(default, model(inputs, time, latent, domain, vf=vf))
        )
        torch.testing.assert_close(guided, expected)
        torch.testing.assert_close(
            guided_logits,
            unconditional + 2.0 * (conditional - unconditional),
        )
        self.assertTrue(torch.equal(disabled, model.decode(unconditional)))

    def test_guidance_passes_share_stochastic_inputs(self):
        model = _denoiser()
        inputs = torch.randn(1, 3, 4, 4, 4)
        time = torch.ones(1)
        latent = torch.randn(1, 4)
        vf = torch.tensor([[0.2, 0.3, 0.5]])
        domain = _domain(inputs)
        calls: list[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor | None,
                torch.Tensor | None,
            ]
        ] = []
        original = model.compute_logits

        def traced(
            x_current: torch.Tensor,
            timestep: torch.Tensor,
            style: torch.Tensor,
            domain: torch.Tensor,
            vf: torch.Tensor | None = None,
            vf_present: torch.Tensor | None = None,
            height: torch.Tensor | None = None,
            profile: torch.Tensor | None = None,
            profile_present: torch.Tensor | None = None,
            coarse: torch.Tensor | None = None,
            corruption_level: torch.Tensor | None = None,
        ) -> torch.Tensor:
            calls.append((x_current, timestep, style, domain, vf, vf_present))
            return original(
                x_current,
                timestep,
                style,
                domain,
                vf=vf,
                vf_present=vf_present,
                height=height,
                profile=profile,
                profile_present=profile_present,
                coarse=coarse,
                corruption_level=corruption_level,
            )

        model.compute_logits = traced
        model.apply_guidance_logits(
            inputs,
            time,
            latent,
            guidance=1.5,
            domain=domain,
            vf=vf,
        )

        self.assertEqual(len(calls), 2)
        for index in range(4):
            self.assertIs(calls[0][index], calls[1][index])
        self.assertIsNone(calls[0][4])
        self.assertIsNone(calls[0][5])
        self.assertIs(calls[1][4], vf)
        self.assertIsNone(calls[1][5])

    def test_vf_presence_mask_matches_exact_null_and_conditioned_paths(self):
        model = _denoiser()
        inputs = torch.randn(3, 3, 4, 4, 4)
        time = torch.tensor([0.0, 0.5, 1.0])
        latent = torch.randn(3, 4)
        vf = torch.tensor(
            [
                [0.2, 0.3, 0.5],
                [0.5, 0.25, 0.25],
                [0.1, 0.7, 0.2],
            ]
        )
        domain = _domain(inputs)
        with torch.no_grad():
            model.vf_mlp[-1].weight.fill_(0.25)
            model.vf_mlp[-1].bias.fill_(0.1)

        plain = model.compute_logits(inputs, time, latent, domain)
        conditioned = model.compute_logits(inputs, time, latent, domain, vf=vf)
        mixed = model.compute_logits(
            inputs,
            time,
            latent,
            domain,
            vf=vf,
            vf_present=torch.tensor([False, True, False]),
        )
        all_false = model.compute_logits(
            inputs,
            time,
            latent,
            domain,
            vf=vf,
            vf_present=torch.zeros(3, dtype=torch.bool),
        )
        all_true = model.compute_logits(
            inputs,
            time,
            latent,
            domain,
            vf=vf,
            vf_present=torch.ones(3, dtype=torch.bool),
        )

        self.assertTrue(torch.equal(mixed[0], plain[0]))
        self.assertTrue(torch.equal(mixed[1], conditioned[1]))
        self.assertTrue(torch.equal(mixed[2], plain[2]))
        self.assertTrue(torch.equal(all_false, plain))
        self.assertTrue(torch.equal(all_true, conditioned))
        self.assertFalse(torch.equal(conditioned, plain))

    def test_masked_vf_rows_receive_no_gradient(self):
        model = _denoiser()
        inputs = torch.randn(2, 3, 4, 4, 4)
        time = torch.zeros(2)
        latent = torch.randn(2, 4)
        domain = _domain(inputs)
        vf = torch.tensor(
            [[0.2, 0.3, 0.5], [0.5, 0.25, 0.25]],
            requires_grad=True,
        )
        first = model.vf_mlp[0]
        output = model.vf_mlp[-1]
        with torch.no_grad():
            first.weight.zero_()
            first.bias.zero_()
            first.weight[:3].copy_(torch.eye(3))
            output.weight.zero_()
            output.bias.zero_()
            output.weight[:3, :3].copy_(torch.eye(3))

        embedding = model.embed(
            inputs,
            time,
            latent,
            domain,
            vf,
            vf_present=torch.tensor([True, False]),
        )
        embedding[:, :3].sum().backward()

        self.assertIsNotNone(vf.grad)
        self.assertGreater(float(vf.grad[0].abs().sum()), 0.0)
        self.assertEqual(float(vf.grad[1].abs().sum()), 0.0)

    def test_vf_adapter_preserves_none_path_and_receives_gradients(self):
        model = _denoiser()
        inputs = torch.randn(1, 3, 4, 4, 4)
        time = torch.zeros(1)
        latent = torch.randn(1, 4)
        vf = torch.tensor([[0.25, 0.5, 0.25]])
        domain = _domain(inputs)
        output_layer = model.vf_mlp[-1]

        self.assertEqual(int(torch.count_nonzero(output_layer.weight)), 0)
        self.assertEqual(int(torch.count_nonzero(output_layer.bias)), 0)

        plain = model.compute_logits(inputs, time, latent, domain)
        conditioned = model.compute_logits(
            inputs,
            time,
            latent,
            domain,
            vf=vf,
        )
        self.assertTrue(torch.equal(plain, conditioned))

        predicted = conditioned.softmax(dim=1).mean(dim=(2, 3, 4))
        loss = (predicted - vf).abs().sum(dim=1).mean()
        loss.backward()

        gradient = output_layer.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.0)

        with torch.no_grad():
            output_layer.bias.fill_(0.5)
        learned_plain = model.compute_logits(inputs, time, latent, domain)
        self.assertTrue(torch.equal(plain, learned_plain))


class PairCritic2DTest(unittest.TestCase):
    def test_2d_critic_keeps_spatial_group_norm(self):
        model = _critic()

        self.assertIsInstance(model.blocks[0].norm1, AdaptiveNorm)
        self.assertIsInstance(model.blocks[0].norm1.norm, GroupNormSiLU)
        self.assertIsInstance(model.output_norm, torch.nn.GroupNorm)

    def test_pair_and_time_conditioning_return_global_logits(self):
        model = _critic()
        previous = torch.randn(2, 3, 6, 8, requires_grad=True)
        current = torch.randn(2, 3, 6, 8, requires_grad=True)
        domain = _domain(previous)

        first = model(previous, current, torch.zeros(2), domain)
        scores = model(previous, current, torch.ones(2), domain)
        other = model(previous, current, torch.zeros(2), _domain(previous, 1))

        self.assertIsInstance(scores, CriticScores)
        self.assertEqual(scores.logits_global.shape, torch.Size([2]))
        self.assertFalse(torch.equal(first.logits_global, scores.logits_global))
        self.assertFalse(torch.equal(first.logits_global, other.logits_global))

        scores.logits_global.mean().backward()

        self.assertIsNotNone(previous.grad)
        self.assertIsNotNone(current.grad)
        self.assertIsNotNone(model.output.weight.grad)
        self.assertTrue(bool(torch.isfinite(previous.grad).all()))
        self.assertTrue(bool(torch.isfinite(current.grad).all()))
        self.assertGreater(float(previous.grad.abs().sum()), 0.0)
        self.assertGreater(float(current.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0.0)

    def test_requires_two_feature_levels(self):
        with self.assertRaisesRegex(ValueError, "at least two levels"):
            PairCritic2D(
                num_phases=3,
                channels=(4,),
                embedding_channels=8,
                num_domains=2,
            )


class GradientCheckpointingTest(unittest.TestCase):
    def test_denoiser_and_critic_forward_backward(self):
        denoiser = _denoiser(checkpointing=True).train()
        inputs = torch.randn(1, 3, 4, 4, 4, requires_grad=True)
        latent = torch.randn(1, 4, requires_grad=True)
        clean = denoiser(inputs, torch.zeros(1), latent, _domain(inputs))
        clean.square().mean().backward()

        self.assertEqual(clean.shape, inputs.shape)
        self.assertGreater(float(inputs.grad.abs().sum()), 0.0)
        self.assertGreater(float(latent.grad.abs().sum()), 0.0)

        critic = _critic(checkpointing=True).train()
        previous = torch.randn(1, 3, 4, 4, requires_grad=True)
        current = torch.randn(1, 3, 4, 4, requires_grad=True)
        scores = critic(previous, current, torch.zeros(1), _domain(previous))
        loss = scores.logits_global.mean()
        loss.backward()

        self.assertEqual(scores.logits_global.shape, torch.Size([1]))
        self.assertGreater(float(previous.grad.abs().sum()), 0.0)
        self.assertGreater(float(current.grad.abs().sum()), 0.0)
        self.assertGreater(float(critic.output.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
