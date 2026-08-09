import unittest

import torch

from src.model.blocks import (
    AdaptiveChannelNorm3D,
    AdaptiveGroupNorm,
    ChannelNorm3D,
)
from src.model.critic import CriticScores, PairCritic2D
from src.model.denoiser import Denoiser3D


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
        gradient_checkpointing=checkpointing,
    )


def _critic(*, checkpointing: bool = False) -> PairCritic2D:
    return PairCritic2D(
        num_phases=3,
        channels=(4, 8),
        embedding_channels=8,
        gradient_checkpointing=checkpointing,
    )


class Denoiser3DTest(unittest.TestCase):
    def test_3d_norm_is_independent_at_each_spatial_position(self):
        norm = ChannelNorm3D(4)
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
        self.assertTrue(
            torch.allclose(
                output.mean(dim=1),
                torch.zeros_like(output[:, 0]),
                atol=1.0e-6,
            )
        )
        self.assertEqual(norm(inputs.half()).dtype, torch.float16)

    def test_3d_model_uses_channel_norm(self):
        model = _denoiser()

        self.assertIsInstance(model.encoder[0].norm1, AdaptiveChannelNorm3D)
        self.assertIsInstance(model.output_norm, ChannelNorm3D)
        self.assertEqual(model.downsample_factor, 2)

    def test_clean_output_is_a_latent_conditioned_simplex_with_gradients(self):
        model = _denoiser()
        inputs = torch.randn(2, 3, 4, 6, 8, requires_grad=True)
        time = torch.tensor([0.0, 1.0])
        first_latent = torch.zeros(2, 4)
        second_latent = torch.ones(2, 4, requires_grad=True)

        first = model(inputs, time, first_latent)
        clean = model(inputs, time, second_latent)
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

    def test_group_norm_accepts_a_single_voxel_bottleneck(self):
        model = Denoiser3D(
            num_phases=3,
            base_channels=4,
            channel_multipliers=(1, 2, 4, 8),
            embedding_channels=8,
            latent_channels=4,
        )

        output = model(
            torch.randn(1, 3, 8, 8, 8),
            torch.zeros(1),
            torch.zeros(1, 4),
        )

        self.assertEqual(output.shape, torch.Size([1, 3, 8, 8, 8]))

    def test_soft_anchor_adapter_preserves_null_path_and_receives_gradients(self):
        model = _denoiser()
        inputs = torch.randn(1, 3, 4, 4, 4)
        time = torch.zeros(1)
        latent = torch.randn(1, 4)
        anchor_image = torch.zeros_like(inputs)
        anchor_image[:, 0, 2] = 1.0
        anchor_image[:, 1:, 2] = -1.0
        anchor_mask = torch.zeros(1, 1, 4, 4, 4, dtype=torch.bool)
        anchor_mask[:, :, 2] = True

        plain = model(inputs, time, latent)
        anchored = model(
            inputs,
            time,
            latent,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
        )
        empty = model(
            inputs,
            time,
            latent,
            anchor_image=torch.zeros_like(inputs),
            anchor_mask=torch.zeros_like(anchor_mask),
        )

        self.assertTrue(torch.equal(plain, anchored))
        self.assertTrue(torch.equal(plain, empty))
        anchored.square().mean().backward()
        self.assertIsNotNone(model.anchor_input.weight.grad)
        self.assertGreater(float(model.anchor_input.weight.grad.abs().sum()), 0.0)

        with torch.no_grad():
            model.anchor_input.weight.fill_(0.5)
        learned_plain = model(inputs, time, latent)
        learned_empty = model(
            inputs,
            time,
            latent,
            anchor_image=torch.zeros_like(inputs),
            anchor_mask=torch.zeros_like(anchor_mask),
        )
        self.assertTrue(torch.equal(learned_plain, learned_empty))
        learned_anchor = model(
            inputs,
            time,
            latent,
            anchor_image=anchor_image,
            anchor_mask=anchor_mask,
        )
        self.assertFalse(torch.equal(learned_plain, learned_anchor))

    def test_anchor_adapter_requires_paired_inputs(self):
        model = _denoiser()
        inputs = torch.randn(1, 3, 4, 4, 4)
        time = torch.zeros(1)
        latent = torch.randn(1, 4)

        with self.assertRaisesRegex(ValueError, "provided together"):
            model(
                inputs,
                time,
                latent,
                anchor_image=torch.zeros_like(inputs),
            )

    def test_logit_guidance_preserves_default_and_combines_conditions(self):
        model = _denoiser()
        inputs = torch.randn(1, 3, 4, 4, 4)
        time = torch.ones(1)
        latent = torch.randn(1, 4)
        vf = torch.tensor([[0.2, 0.3, 0.5]])
        with torch.no_grad():
            model.vf_mlp[-1].weight.fill_(0.25)
            model.vf_mlp[-1].bias.fill_(0.1)

        unconditional = model.predict_logits(inputs, time, latent)
        conditional = model.predict_logits(inputs, time, latent, vf=vf)
        expected = model.decode(unconditional + 2.0 * (conditional - unconditional))

        default = model.predict_guided(
            inputs,
            time,
            latent,
            guidance_scale=1.0,
            vf=vf,
        )
        guided = model.predict_guided(
            inputs,
            time,
            latent,
            guidance_scale=2.0,
            vf=vf,
        )
        disabled = model.predict_guided(
            inputs,
            time,
            latent,
            guidance_scale=0.0,
            vf=vf,
        )

        self.assertTrue(torch.equal(default, model(inputs, time, latent, vf=vf)))
        torch.testing.assert_close(guided, expected)
        self.assertTrue(torch.equal(disabled, model.decode(unconditional)))

        for invalid in (-1.0, 1e39, float("nan"), float("inf"), True):
            with self.assertRaisesRegex(ValueError, "guidance_scale"):
                model.predict_guided(
                    inputs,
                    time,
                    latent,
                    guidance_scale=invalid,
                    vf=vf,
                )

    def test_guidance_passes_share_stochastic_inputs(self):
        model = _denoiser()
        inputs = torch.randn(1, 3, 4, 4, 4)
        time = torch.ones(1)
        latent = torch.randn(1, 4)
        vf = torch.tensor([[0.2, 0.3, 0.5]])
        calls: list[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor | None,
            ]
        ] = []
        original = model.predict_logits

        def traced(
            x_current: torch.Tensor,
            timestep: torch.Tensor,
            style: torch.Tensor,
            vf: torch.Tensor | None = None,
            vf_present: torch.Tensor | None = None,
            anchor_image: torch.Tensor | None = None,
            anchor_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            calls.append((x_current, timestep, style, vf, vf_present))
            return original(
                x_current,
                timestep,
                style,
                vf=vf,
                vf_present=vf_present,
                anchor_image=anchor_image,
                anchor_mask=anchor_mask,
            )

        model.predict_logits = traced
        model.predict_guided(
            inputs,
            time,
            latent,
            guidance_scale=1.5,
            vf=vf,
        )

        self.assertEqual(len(calls), 2)
        for index in range(3):
            self.assertIs(calls[0][index], calls[1][index])
        self.assertIsNone(calls[0][3])
        self.assertIsNone(calls[0][4])
        self.assertIs(calls[1][3], vf)
        self.assertIsNone(calls[1][4])

    def test_guidance_returns_the_current_state_dtype(self):
        model = _denoiser()
        inputs = torch.randn(1, 3, 4, 4, 4, dtype=torch.float32)
        time = torch.ones(1)
        latent = torch.randn(1, 4)
        vf = torch.tensor([[0.2, 0.3, 0.5]])
        original = model.predict_logits

        def half_logits(*args, **kwargs):
            return original(*args, **kwargs).to(torch.float16)

        model.predict_logits = half_logits

        guided = model.predict_guided(
            inputs,
            time,
            latent,
            guidance_scale=1.5,
            vf=vf,
        )

        self.assertEqual(guided.dtype, inputs.dtype)

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
        with torch.no_grad():
            model.vf_mlp[-1].weight.fill_(0.25)
            model.vf_mlp[-1].bias.fill_(0.1)

        plain = model.predict_logits(inputs, time, latent)
        conditioned = model.predict_logits(inputs, time, latent, vf=vf)
        mixed = model.predict_logits(
            inputs,
            time,
            latent,
            vf=vf,
            vf_present=torch.tensor([False, True, False]),
        )
        all_false = model.predict_logits(
            inputs,
            time,
            latent,
            vf=vf,
            vf_present=torch.zeros(3, dtype=torch.bool),
        )
        all_true = model.predict_logits(
            inputs,
            time,
            latent,
            vf=vf,
            vf_present=torch.ones(3, dtype=torch.bool),
        )

        self.assertTrue(torch.equal(mixed[0], plain[0]))
        self.assertTrue(torch.equal(mixed[1], conditioned[1]))
        self.assertTrue(torch.equal(mixed[2], plain[2]))
        self.assertTrue(torch.equal(all_false, plain))
        self.assertTrue(torch.equal(all_true, conditioned))
        self.assertFalse(torch.equal(conditioned, plain))

    def test_vf_presence_validation_rejects_invalid_inputs(self):
        model = _denoiser()
        inputs = torch.randn(2, 3, 4, 4, 4)
        time = torch.zeros(2)
        latent = torch.randn(2, 4)
        vf = torch.tensor([[0.2, 0.3, 0.5], [0.5, 0.25, 0.25]])

        with self.assertRaisesRegex(ValueError, "vf_present requires vf"):
            model.predict_logits(
                inputs,
                time,
                latent,
                vf_present=torch.ones(2, dtype=torch.bool),
            )
        with self.assertRaisesRegex(TypeError, "vf must be a floating-point"):
            model.predict_logits(inputs, time, latent, vf=vf.to(torch.int64))
        with self.assertRaisesRegex(ValueError, "vf must have shape"):
            model.predict_logits(inputs, time, latent, vf=vf[:1])
        with self.assertRaisesRegex(TypeError, "vf_present must be a boolean"):
            model.predict_logits(
                inputs,
                time,
                latent,
                vf=vf,
                vf_present=torch.ones(2),
            )
        with self.assertRaisesRegex(ValueError, "vf_present must have shape"):
            model.predict_logits(
                inputs,
                time,
                latent,
                vf=vf,
                vf_present=torch.ones(2, 1, dtype=torch.bool),
            )

    def test_masked_vf_rows_receive_no_gradient(self):
        model = _denoiser()
        inputs = torch.randn(2, 3, 4, 4, 4)
        time = torch.zeros(2)
        latent = torch.randn(2, 4)
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
        output_layer = model.vf_mlp[-1]

        self.assertEqual(int(torch.count_nonzero(output_layer.weight)), 0)
        self.assertEqual(int(torch.count_nonzero(output_layer.bias)), 0)

        plain = model.predict_logits(inputs, time, latent)
        conditioned = model.predict_logits(
            inputs,
            time,
            latent,
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
        learned_plain = model.predict_logits(inputs, time, latent)
        self.assertTrue(torch.equal(plain, learned_plain))


class PairCritic2DTest(unittest.TestCase):
    def test_2d_critic_keeps_spatial_group_norm(self):
        model = _critic()

        self.assertIsInstance(model.blocks[0].norm1, AdaptiveGroupNorm)
        self.assertIsInstance(model.local_norm, torch.nn.GroupNorm)
        self.assertIsInstance(model.output_norm, torch.nn.GroupNorm)

    def test_pair_and_time_conditioning_return_global_and_logits_local(self):
        model = _critic()
        previous = torch.randn(2, 3, 6, 8, requires_grad=True)
        current = torch.randn(2, 3, 6, 8, requires_grad=True)

        first = model(previous, current, torch.zeros(2))
        scores = model(previous, current, torch.ones(2))

        self.assertIsInstance(scores, CriticScores)
        self.assertEqual(scores.logits_global.shape, torch.Size([2]))
        self.assertEqual(scores.logits_local.shape, torch.Size([2, 3, 4]))
        self.assertFalse(torch.equal(first.logits_global, scores.logits_global))
        self.assertFalse(torch.equal(first.logits_local, scores.logits_local))

        (scores.logits_global.mean() + scores.logits_local.mean()).backward()

        self.assertIsNotNone(previous.grad)
        self.assertIsNotNone(current.grad)
        self.assertIsNotNone(model.output.weight.grad)
        self.assertIsNotNone(model.local_output.weight.grad)
        self.assertTrue(bool(torch.isfinite(previous.grad).all()))
        self.assertTrue(bool(torch.isfinite(current.grad).all()))
        self.assertGreater(float(previous.grad.abs().sum()), 0.0)
        self.assertGreater(float(current.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.local_output.weight.grad.abs().sum()), 0.0)

    def test_requires_two_feature_levels(self):
        with self.assertRaisesRegex(ValueError, "at least two levels"):
            PairCritic2D(
                num_phases=3,
                channels=(4,),
                embedding_channels=8,
            )


class GradientCheckpointingTest(unittest.TestCase):
    def test_denoiser_and_critic_forward_backward(self):
        denoiser = _denoiser(checkpointing=True).train()
        inputs = torch.randn(1, 3, 4, 4, 4, requires_grad=True)
        latent = torch.randn(1, 4, requires_grad=True)
        clean = denoiser(inputs, torch.zeros(1), latent)
        clean.square().mean().backward()

        self.assertEqual(clean.shape, inputs.shape)
        self.assertGreater(float(inputs.grad.abs().sum()), 0.0)
        self.assertGreater(float(latent.grad.abs().sum()), 0.0)

        critic = _critic(checkpointing=True).train()
        previous = torch.randn(1, 3, 4, 4, requires_grad=True)
        current = torch.randn(1, 3, 4, 4, requires_grad=True)
        scores = critic(previous, current, torch.zeros(1))
        loss = scores.logits_global.mean() + scores.logits_local.mean()
        loss.backward()

        self.assertEqual(scores.logits_global.shape, torch.Size([1]))
        self.assertEqual(scores.logits_local.shape, torch.Size([1, 2, 2]))
        self.assertGreater(float(previous.grad.abs().sum()), 0.0)
        self.assertGreater(float(current.grad.abs().sum()), 0.0)
        self.assertGreater(float(critic.output.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(critic.local_output.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
