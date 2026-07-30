import pytest
import torch

from src.diffusion import DiffusionProcess, extract


def test_vp_schedule_is_monotonic() -> None:
    process = DiffusionProcess(11, beta_min=0.1, beta_max=20.0)
    normalized_time = torch.linspace(0.0, 1.0, 12)
    expected = torch.exp(
        -0.5 * (20.0 - 0.1) * normalized_time.square()
        - 0.1 * normalized_time
    )

    assert torch.allclose(process.alpha_bars, expected)
    assert process.alpha_bars[0].item() == 1.0
    assert bool(torch.all(process.alpha_bars[1:] < process.alpha_bars[:-1]))
    assert process.betas[0].item() == 0.0
    assert bool(torch.all((process.betas[1:] > 0) & (process.betas[1:] < 1)))


def test_forward_pair_uses_correlated_markov_transition() -> None:
    process = DiffusionProcess(6)
    clean = torch.linspace(-1.0, 1.0, 2 * 2 * 3 * 3).reshape(2, 2, 3, 3)
    previous_noise = torch.linspace(1.0, -1.0, clean.numel()).reshape_as(clean)
    step_noise = torch.full_like(clean, 0.25)
    transitions = torch.tensor([0, 4])

    previous, current = process.forward_pair(
        clean,
        transitions,
        previous_noise=previous_noise,
        step_noise=step_noise,
    )

    alpha_bar = extract(process.alpha_bars, transitions, clean)
    expected_previous = (
        alpha_bar.sqrt() * clean
        + (1.0 - alpha_bar).sqrt() * previous_noise
    )
    alpha = extract(process.alphas, transitions + 1, clean)
    beta = extract(process.betas, transitions + 1, clean)
    expected_current = (
        alpha.sqrt() * expected_previous
        + beta.sqrt() * step_noise
    )

    assert torch.allclose(previous, expected_previous)
    assert torch.allclose(current, expected_current)


def test_transition_zero_posterior_is_exactly_deterministic() -> None:
    process = DiffusionProcess(4)
    current = torch.randn(3, 1, 4, 4)
    clean_prediction = torch.randn_like(current)
    noise = torch.full_like(current, 1_000.0)

    mean, variance = process.posterior_mean_variance(
        current,
        clean_prediction,
        0,
    )
    sample = process.posterior_sample(
        current,
        clean_prediction,
        0,
        noise=noise,
    )

    assert torch.count_nonzero(variance).item() == 0
    assert torch.allclose(mean, clean_prediction)
    assert torch.equal(sample, clean_prediction)


def test_q_sample_accepts_one_state_per_batch_item() -> None:
    process = DiffusionProcess(5)
    clean = torch.zeros(3, 1, 2, 2, 2)
    noise = torch.ones_like(clean)
    states = torch.tensor([0, 2, 5])

    sample = process.q_sample(clean, states, noise=noise)
    alpha_bar = extract(process.alpha_bars, states, clean)

    assert torch.allclose(sample, (1.0 - alpha_bar).sqrt())
    assert torch.equal(sample[0], clean[0])
    assert sample[1].mean() < sample[2].mean()


def test_reverse_chain_preserves_shape_and_refreshes_latent() -> None:
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

    process = DiffusionProcess(3)
    model = LatentRecorder()
    terminal = torch.randn(2, 1, 4, 4, 4)

    result, history = process.reverse_chain(
        model,
        terminal,
        8,
        return_history=True,
    )

    assert result.shape == terminal.shape
    assert all(state.shape == terminal.shape for state in history)
    assert len(history) == process.timesteps + 1
    assert all(latent.shape == (2, 8) for latent in model.latents)
    assert not torch.equal(model.latents[0], model.latents[1])


def test_reverse_chain_forwards_fixed_anchor_at_every_step() -> None:
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

    process = DiffusionProcess(3)
    model = AnchorRecorder()
    terminal = torch.randn(1, 2, 4, 4, 4)
    anchor_image = torch.zeros_like(terminal)
    anchor_mask = torch.zeros(1, 1, 4, 4, 4, dtype=torch.bool)

    process.reverse_chain(
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


def test_invalid_state_and_transition_ranges_are_rejected() -> None:
    process = DiffusionProcess(3)
    values = torch.zeros(2, 1, 4, 4)

    with pytest.raises(ValueError, match="state must be between"):
        process.q_sample(values, 4)
    with pytest.raises(ValueError, match="transition must be between"):
        process.forward_pair(values, 3)
    with pytest.raises(ValueError, match="transition must be between"):
        process.posterior_sample(values, values, torch.tensor([0, -1]))
