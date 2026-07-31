import unittest

import torch

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
    def test_clean_output_is_a_latent_conditioned_simplex_with_gradients(self):
        model = _denoiser()
        inputs = torch.randn(2, 3, 4, 6, 8, requires_grad=True)
        time = torch.tensor([0.0, 1.0])
        first_latent = torch.zeros(2, 4)
        second_latent = torch.ones(2, 4, requires_grad=True)

        first = model(inputs, time, first_latent)
        clean = model(inputs, time, second_latent)
        probabilities = (clean + 1.0) / 2.0

        self.assertEqual(clean.shape, inputs.shape)
        self.assertGreaterEqual(float(clean.detach().min()), -1.0)
        self.assertLessEqual(float(clean.detach().max()), 1.0)
        self.assertTrue(
            torch.allclose(
                probabilities.sum(dim=1),
                torch.ones_like(probabilities[:, 0]),
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

    def test_rejects_spatial_sizes_outside_the_level_multiple(self):
        model = Denoiser3D(
            num_phases=3,
            base_channels=4,
            channel_multipliers=(1, 2, 4),
            embedding_channels=8,
            latent_channels=4,
        )

        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            model(
                torch.randn(1, 3, 8, 8, 10),
                torch.zeros(1),
                torch.zeros(1, 4),
            )

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
            model.anchor_input.bias.fill_(0.5)
        learned_plain = model(inputs, time, latent)
        learned_empty = model(
            inputs,
            time,
            latent,
            anchor_image=torch.zeros_like(inputs),
            anchor_mask=torch.zeros_like(anchor_mask),
        )
        self.assertTrue(torch.equal(learned_plain, learned_empty))

    def test_anchor_image_and_mask_must_be_passed_together(self):
        model = _denoiser()
        inputs = torch.randn(1, 3, 4, 4, 4)

        with self.assertRaisesRegex(ValueError, "provided together"):
            model(
                inputs,
                torch.zeros(1),
                torch.randn(1, 4),
                anchor_image=torch.zeros_like(inputs),
            )


class PairCritic2DTest(unittest.TestCase):
    def test_pair_and_time_conditioning_return_global_and_local_logits(self):
        model = _critic()
        previous = torch.randn(2, 3, 6, 8, requires_grad=True)
        current = torch.randn(2, 3, 6, 8, requires_grad=True)

        first = model(previous, current, torch.zeros(2))
        scores = model(previous, current, torch.ones(2))

        self.assertIsInstance(scores, CriticScores)
        self.assertEqual(scores.global_logits.shape, torch.Size([2]))
        self.assertEqual(scores.local_logits.shape, torch.Size([2, 3, 4]))
        self.assertFalse(torch.equal(first.global_logits, scores.global_logits))
        self.assertFalse(torch.equal(first.local_logits, scores.local_logits))

        (scores.global_logits.mean() + scores.local_logits.mean()).backward()

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
        with self.assertRaisesRegex(ValueError, r"time must have shape \[B\]"):
            model(previous.detach(), current.detach(), torch.zeros(2, 1))

    def test_requires_two_feature_levels(self):
        with self.assertRaisesRegex(ValueError, "at least two levels"):
            PairCritic2D(
                num_phases=3,
                channels=(4,),
                embedding_channels=8,
            )

    def test_rejects_spatial_sizes_outside_the_level_multiple(self):
        model = PairCritic2D(
            num_phases=3,
            channels=(4, 8, 16),
            embedding_channels=8,
        )
        previous = torch.randn(1, 3, 8, 10)

        with self.assertRaisesRegex(ValueError, "divisible by 4"):
            model(previous, previous.clone(), torch.zeros(1))


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
        loss = scores.global_logits.mean() + scores.local_logits.mean()
        loss.backward()

        self.assertEqual(scores.global_logits.shape, torch.Size([1]))
        self.assertEqual(scores.local_logits.shape, torch.Size([1, 2, 2]))
        self.assertGreater(float(previous.grad.abs().sum()), 0.0)
        self.assertGreater(float(current.grad.abs().sum()), 0.0)
        self.assertGreater(float(critic.output.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(critic.local_output.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
