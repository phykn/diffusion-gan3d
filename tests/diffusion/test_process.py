import pytest
import torch

from src.diffusion import Diffusion


def _extract(
    values: torch.Tensor,
    indices: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    selected = values.index_select(0, indices)
    return selected.reshape(reference.shape[0], *([1] * (reference.ndim - 1)))


def test_vp_schedule_is_monotonic() -> None:
    process = Diffusion(11, beta_min=0.1, beta_max=20.0)
    normalized_time = torch.linspace(0.0, 1.0, 12)
    expected = torch.exp(
        -0.5 * (20.0 - 0.1) * normalized_time.square() - 0.1 * normalized_time
    )

    assert torch.allclose(process.alpha_bars, expected)
    assert process.alpha_bars[0].item() == 1.0
    assert bool(torch.all(process.alpha_bars[1:] < process.alpha_bars[:-1]))
    assert process.betas[0].item() == 0.0
    assert bool(torch.all((process.betas[1:] > 0) & (process.betas[1:] < 1)))


def test_sample_pair_uses_correlated_markov_transition() -> None:
    process = Diffusion(6)
    clean = torch.linspace(-1.0, 1.0, 2 * 2 * 3 * 3).reshape(2, 2, 3, 3)
    previous_noise = torch.linspace(1.0, -1.0, clean.numel()).reshape_as(clean)
    step_noise = torch.full_like(clean, 0.25)
    transitions = torch.tensor([0, 4])

    previous, current = process.sample_pair(
        clean,
        transitions,
        previous_noise=previous_noise,
        step_noise=step_noise,
    )

    alpha_bar = _extract(process.alpha_bars, transitions, clean)
    expected_previous = (
        alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * previous_noise
    )
    alpha = _extract(process.alphas, transitions + 1, clean)
    beta = _extract(process.betas, transitions + 1, clean)
    expected_current = alpha.sqrt() * expected_previous + beta.sqrt() * step_noise

    assert torch.allclose(previous, expected_previous)
    assert torch.allclose(current, expected_current)


def test_transition_zero_posterior_is_exactly_deterministic() -> None:
    process = Diffusion(4)
    current = torch.randn(3, 1, 4, 4)
    clean_prediction = torch.randn_like(current)
    noise = torch.full_like(current, 1_000.0)

    mean, variance = process.get_posterior(
        current,
        clean_prediction,
        0,
    )
    sample = process.sample_posterior(
        current,
        clean_prediction,
        0,
        noise=noise,
    )

    assert torch.count_nonzero(variance).item() == 0
    assert torch.allclose(mean, clean_prediction)
    assert torch.equal(sample, clean_prediction)


def test_scalar_and_uniform_tensor_transitions_are_equivalent() -> None:
    process = Diffusion(6)
    clean = torch.linspace(-1.0, 1.0, 2 * 2 * 3 * 3).reshape(2, 2, 3, 3)
    current = clean.flip(-1)
    prediction = clean.tanh()
    previous_noise = torch.linspace(1.0, -1.0, clean.numel()).reshape_as(clean)
    step_noise = torch.full_like(clean, 0.25)
    posterior_noise = torch.full_like(clean, -0.5)
    transitions = torch.full((clean.shape[0],), 3, dtype=torch.long)

    scalar_pair = process.sample_pair(
        clean,
        3,
        previous_noise=previous_noise,
        step_noise=step_noise,
    )
    tensor_pair = process.sample_pair(
        clean,
        transitions,
        previous_noise=previous_noise,
        step_noise=step_noise,
    )
    scalar_mean, scalar_variance = process.get_posterior(current, prediction, 3)
    tensor_mean, tensor_variance = process.get_posterior(
        current,
        prediction,
        transitions,
    )
    scalar_sample = process.sample_posterior(
        current,
        prediction,
        3,
        noise=posterior_noise,
    )
    tensor_sample = process.sample_posterior(
        current,
        prediction,
        transitions,
        noise=posterior_noise,
    )

    assert all(
        torch.equal(scalar, tensor)
        for scalar, tensor in zip(scalar_pair, tensor_pair, strict=True)
    )
    assert torch.equal(scalar_mean, tensor_mean)
    assert torch.equal(scalar_variance, tensor_variance)
    assert torch.equal(scalar_sample, tensor_sample)


def test_mixed_posterior_transitions_preserve_deterministic_items() -> None:
    process = Diffusion(5)
    current = torch.randn(3, 2, 3, 3)
    prediction = torch.randn_like(current)
    noise = torch.randn_like(current)
    transitions = torch.tensor([0, 2, 4])

    mean, variance = process.get_posterior(current, prediction, transitions)
    sample = process.sample_posterior(
        current,
        prediction,
        transitions,
        noise=noise,
    )
    expected = mean + variance.sqrt() * noise
    expected[0] = prediction[0]

    assert torch.equal(sample[0], prediction[0])
    assert torch.allclose(sample, expected)


def test_add_noise_accepts_one_state_per_batch_item() -> None:
    process = Diffusion(5)
    clean = torch.zeros(3, 1, 2, 2, 2)
    noise = torch.ones_like(clean)
    states = torch.tensor([0, 2, 5])

    sample = process.add_noise(clean, states, noise=noise)
    alpha_bar = _extract(process.alpha_bars, states, clean)

    assert torch.allclose(sample, (1.0 - alpha_bar).sqrt())
    assert torch.equal(sample[0], clean[0])
    assert sample[1].mean() < sample[2].mean()


def test_sample_preserves_shape_and_refreshes_latent() -> None:
    class LatentRecorder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.latents: list[torch.Tensor] = []

        def forward(
            self,
            current: torch.Tensor,
            timestep: torch.Tensor,
            latent: torch.Tensor,
        ) -> torch.Tensor:
            del timestep
            self.latents.append(latent.detach().clone())
            latent_value = latent.mean(dim=1).reshape(
                current.shape[0],
                *([1] * (current.ndim - 1)),
            )
            return torch.tanh(0.1 * current + latent_value)

    process = Diffusion(3)
    model = LatentRecorder()
    terminal = torch.randn(2, 1, 4, 4, 4)

    result = process.sample(
        model,
        terminal,
        8,
    )

    assert result.shape == terminal.shape
    assert all(latent.shape == (2, 8) for latent in model.latents)
    assert not torch.equal(model.latents[0], model.latents[1])


def test_sample_forwards_fixed_anchor_at_every_step() -> None:
    class AnchorRecorder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

        def forward(
            self,
            current: torch.Tensor,
            timestep: torch.Tensor,
            latent: torch.Tensor,
            *,
            anchor_image: torch.Tensor,
            anchor_mask: torch.Tensor,
        ) -> torch.Tensor:
            del timestep, latent
            self.calls.append((anchor_image, anchor_mask))
            return current.tanh()

    process = Diffusion(3)
    model = AnchorRecorder()
    terminal = torch.randn(1, 2, 4, 4, 4)
    anchor_image = torch.zeros_like(terminal)
    anchor_mask = torch.zeros(1, 1, 4, 4, 4, dtype=torch.bool)

    process.sample(
        model,
        terminal,
        4,
        model_kwargs={
            "anchor_image": anchor_image,
            "anchor_mask": anchor_mask,
        },
    )

    assert len(model.calls) == process.timesteps
    assert all(image is anchor_image for image, _ in model.calls)
    assert all(mask is anchor_mask for _, mask in model.calls)


def test_sample_projects_clean_prediction_at_every_step() -> None:
    class ZeroModel(torch.nn.Module):
        def forward(
            self,
            current: torch.Tensor,
            timestep: torch.Tensor,
            latent: torch.Tensor,
        ) -> torch.Tensor:
            del timestep, latent
            return torch.zeros_like(current)

    process = Diffusion(3)
    terminal = torch.randn(1, 2, 4, 4, 4)
    calls = 0

    def project(clean: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        projected = clean.clone()
        projected[:, 1, 2] = 1.0
        return projected

    result = process.sample(
        ZeroModel(),
        terminal,
        4,
        project=project,
    )

    assert calls == process.timesteps
    assert torch.all(result[:, 1, 2] == 1.0)


def test_invalid_state_and_transition_ranges_are_rejected() -> None:
    process = Diffusion(3)
    values = torch.zeros(2, 1, 4, 4)

    with pytest.raises(ValueError, match="state must be between"):
        process.add_noise(values, 4)
    with pytest.raises(ValueError, match="transition must be between"):
        process.sample_pair(values, 3)
    with pytest.raises(ValueError, match="transition must be between"):
        process.sample_posterior(values, values, torch.tensor([0, -1]))
