import math
from collections.abc import Sequence
from contextlib import nullcontext
from numbers import Integral

import torch
from torch import nn


def extract(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Gather one scalar per batch item and add trailing singleton dimensions."""
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be one-dimensional.")
    if timesteps.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("timesteps must use an integer dtype.")
    if reference.ndim < 2:
        raise ValueError("reference must have batch and channel dimensions.")
    if timesteps.shape[0] != reference.shape[0]:
        raise ValueError("timesteps and reference must have the same batch size.")
    if timesteps.numel() and (
        int(timesteps.min().item()) < 0
        or int(timesteps.max().item()) >= values.shape[0]
    ):
        raise ValueError("timesteps contain an out-of-range index.")

    selected = values.to(
        device=reference.device,
        dtype=reference.dtype,
    ).index_select(0, timesteps.to(reference.device, dtype=torch.long))
    return selected.reshape(reference.shape[0], *([1] * (reference.ndim - 1)))


class DiffusionProcess(nn.Module):
    """Variance-preserving Gaussian diffusion with ``T`` discrete transitions.

    State indices run from ``0`` (clean) through ``T`` (most noisy). Transition
    index ``t`` maps state ``t`` to state ``t + 1`` in the forward process and
    state ``t + 1`` back to state ``t`` in the reverse process.
    """

    def __init__(
        self,
        timesteps: int,
        *,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
    ) -> None:
        super().__init__()
        if (
            not isinstance(timesteps, int)
            or isinstance(timesteps, bool)
            or timesteps < 1
        ):
            raise ValueError("timesteps must be a positive integer.")
        if not math.isfinite(beta_min) or beta_min < 0:
            raise ValueError("beta_min must be finite and non-negative.")
        if not math.isfinite(beta_max) or beta_max <= 0:
            raise ValueError("beta_max must be finite and positive.")
        if beta_max < beta_min:
            raise ValueError("beta_max must be at least beta_min.")

        normalized_time = torch.linspace(
            0.0,
            1.0,
            timesteps + 1,
            dtype=torch.float64,
        )
        alpha_bars = torch.exp(
            -0.5 * (beta_max - beta_min) * normalized_time.square()
            - beta_min * normalized_time
        )
        alphas = torch.ones_like(alpha_bars)
        alphas[1:] = alpha_bars[1:] / alpha_bars[:-1]
        betas = 1.0 - alphas

        self.timesteps = timesteps
        self.beta_min = float(beta_min)
        self.beta_max = float(beta_max)
        self.register_buffer("alpha_bars", alpha_bars.to(torch.float32))
        self.register_buffer("alphas", alphas.to(torch.float32))
        self.register_buffer("betas", betas.to(torch.float32))

    def q_sample(
        self,
        clean: torch.Tensor,
        state: int | torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample ``q(x_state | x_0)`` from a clean 2D or 3D batch."""
        self._validate_batch("clean", clean)
        states = self._batch_timesteps(
            state,
            clean,
            maximum=self.timesteps,
            label="state",
        )
        if noise is None:
            noise = torch.randn_like(clean)
        else:
            self._validate_matching("noise", noise, clean)

        alpha_bar = extract(self.alpha_bars, states, clean)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).clamp_min(0).sqrt() * noise

    def forward_pair(
        self,
        clean: torch.Tensor,
        transition: int | torch.Tensor,
        *,
        previous_noise: torch.Tensor | None = None,
        step_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a correlated ``(x_t, x_{t+1})`` forward-process pair.

        ``x_t`` is first sampled from the clean marginal. ``x_{t+1}`` is then
        sampled from the one-step Markov transition conditioned on that exact
        ``x_t``; the two outputs are therefore not independent noisy views.
        """
        self._validate_batch("clean", clean)
        transitions = self._batch_timesteps(
            transition,
            clean,
            maximum=self.timesteps - 1,
            label="transition",
        )
        previous = self.q_sample(
            clean,
            transitions,
            noise=previous_noise,
        )
        if step_noise is None:
            step_noise = torch.randn_like(clean)
        else:
            self._validate_matching("step_noise", step_noise, clean)

        current_states = transitions + 1
        alpha = extract(self.alphas, current_states, clean)
        beta = extract(self.betas, current_states, clean)
        current = alpha.sqrt() * previous + beta.clamp_min(0).sqrt() * step_noise
        return previous, current

    def posterior_mean_variance(
        self,
        current: torch.Tensor,
        clean_prediction: torch.Tensor,
        transition: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return parameters of ``q(x_t | x_{t+1}, x0_prediction)``."""
        self._validate_batch("current", current)
        self._validate_matching("clean_prediction", clean_prediction, current)
        transitions = self._batch_timesteps(
            transition,
            current,
            maximum=self.timesteps - 1,
            label="transition",
        )
        current_states = transitions + 1

        alpha_bar_previous = extract(self.alpha_bars, transitions, current)
        alpha_bar_current = extract(self.alpha_bars, current_states, current)
        alpha = extract(self.alphas, current_states, current)
        beta = extract(self.betas, current_states, current)
        denominator = (1.0 - alpha_bar_current).clamp_min(
            torch.finfo(current.dtype).tiny
        )

        clean_coefficient = beta * alpha_bar_previous.sqrt() / denominator
        current_coefficient = (
            alpha.sqrt() * (1.0 - alpha_bar_previous) / denominator
        )
        mean = (
            clean_coefficient * clean_prediction
            + current_coefficient * current
        )
        variance = (
            beta
            * (1.0 - alpha_bar_previous)
            / denominator
        ).clamp_min(0)
        return mean, variance

    def posterior_sample(
        self,
        current: torch.Tensor,
        clean_prediction: torch.Tensor,
        transition: int | torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample ``x_t`` from a predicted reverse transition.

        Transition ``t=0`` is deterministic and returns the posterior mean,
        with an explicit exact return of ``clean_prediction``.
        """
        mean, variance = self.posterior_mean_variance(
            current,
            clean_prediction,
            transition,
        )
        transitions = self._batch_timesteps(
            transition,
            current,
            maximum=self.timesteps - 1,
            label="transition",
        )
        if noise is None:
            noise = torch.randn_like(current)
        else:
            self._validate_matching("noise", noise, current)

        stochastic = (transitions != 0).to(current.dtype)
        stochastic = stochastic.reshape(
            current.shape[0],
            *([1] * (current.ndim - 1)),
        )
        sample = mean + stochastic * variance.sqrt() * noise
        return torch.where(
            stochastic.bool(),
            sample,
            clean_prediction,
        )

    def reverse_chain(
        self,
        model: nn.Module,
        terminal: torch.Tensor,
        latent_shape: int | Sequence[int],
        *,
        no_grad: bool = True,
        return_history: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Run ``x_T -> ... -> x_0`` using a fresh latent at every step.

        The model is called as ``model(x_current, transition, z)`` and must
        predict a clean tensor with the same shape as ``x_current``. When
        requested, history is ordered ``(x_T, x_{T-1}, ..., x_0)``.
        """
        self._validate_batch("terminal", terminal)
        latent_dimensions = self._latent_dimensions(latent_shape)
        current = terminal
        history = [current] if return_history else None
        context = torch.no_grad() if no_grad else nullcontext()

        with context:
            for transition in reversed(range(self.timesteps)):
                batch_timesteps = torch.full(
                    (current.shape[0],),
                    transition,
                    device=current.device,
                    dtype=torch.long,
                )
                latent = torch.randn(
                    current.shape[0],
                    *latent_dimensions,
                    device=current.device,
                    dtype=current.dtype,
                    generator=generator,
                )
                clean_prediction = model(current, batch_timesteps, latent)
                if not isinstance(clean_prediction, torch.Tensor):
                    raise TypeError("model must return a torch.Tensor.")
                self._validate_matching(
                    "model output",
                    clean_prediction,
                    current,
                )
                current = self.posterior_sample(
                    current,
                    clean_prediction,
                    batch_timesteps,
                    noise=torch.randn(
                        current.shape,
                        device=current.device,
                        dtype=current.dtype,
                        generator=generator,
                    ),
                )
                if history is not None:
                    history.append(current)

        if history is None:
            return current
        return current, tuple(history)

    @staticmethod
    def _validate_batch(name: str, values: torch.Tensor) -> None:
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if values.ndim < 2:
            raise ValueError(f"{name} must have batch and channel dimensions.")
        if values.shape[0] < 1:
            raise ValueError(f"{name} must contain at least one batch item.")
        if not values.is_floating_point():
            raise ValueError(f"{name} must use a floating-point dtype.")

    @classmethod
    def _validate_matching(
        cls,
        name: str,
        values: torch.Tensor,
        reference: torch.Tensor,
    ) -> None:
        cls._validate_batch(name, values)
        if values.shape != reference.shape:
            raise ValueError(f"{name} must have shape {tuple(reference.shape)}.")
        if values.device != reference.device:
            raise ValueError(f"{name} and the reference must use the same device.")
        if values.dtype != reference.dtype:
            raise ValueError(f"{name} and the reference must use the same dtype.")

    @staticmethod
    def _batch_timesteps(
        value: int | torch.Tensor,
        reference: torch.Tensor,
        *,
        maximum: int,
        label: str,
    ) -> torch.Tensor:
        if isinstance(value, Integral) and not isinstance(value, bool):
            result = torch.full(
                (reference.shape[0],),
                int(value),
                device=reference.device,
                dtype=torch.long,
            )
        elif isinstance(value, torch.Tensor):
            if value.dtype not in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                raise ValueError(f"{label} must use an integer dtype.")
            if value.ndim == 0:
                result = value.to(reference.device, dtype=torch.long).expand(
                    reference.shape[0]
                )
            elif value.ndim == 1 and value.shape[0] == reference.shape[0]:
                result = value.to(reference.device, dtype=torch.long)
            else:
                raise ValueError(
                    f"{label} must be scalar or have one value per batch item."
                )
        else:
            raise TypeError(f"{label} must be an integer or torch.Tensor.")

        if int(result.min().item()) < 0 or int(result.max().item()) > maximum:
            raise ValueError(f"{label} must be between 0 and {maximum}.")
        return result

    @staticmethod
    def _latent_dimensions(value: int | Sequence[int]) -> tuple[int, ...]:
        if isinstance(value, int) and not isinstance(value, bool):
            result = (value,)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            result = tuple(value)
        else:
            raise TypeError("latent_shape must be an integer or sequence.")
        if not result or any(
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 1
            for size in result
        ):
            raise ValueError("latent_shape must contain positive integers.")
        return result
