import math
from collections.abc import Callable, Mapping
from numbers import Integral

import torch
from torch import nn


def _extract(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    selected = values.to(
        device=reference.device,
        dtype=reference.dtype,
    ).index_select(0, timesteps.to(reference.device, dtype=torch.long))
    return selected.reshape(reference.shape[0], *([1] * (reference.ndim - 1)))


class Diffusion(nn.Module):
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

    def add_noise(
        self,
        clean: torch.Tensor,
        state: int | torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample ``q(x_state | x_0)`` from a clean 2D or 3D batch."""
        self._validate_batch("clean", clean)
        states, _ = self._batch_timesteps(
            state,
            clean,
            maximum=self.timesteps,
            label="state",
        )
        return self._add_noise(clean, states, noise)

    def _add_noise(
        self,
        clean: torch.Tensor,
        states: torch.Tensor,
        noise: torch.Tensor | None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean)
        else:
            self._validate_matching("noise", noise, clean)

        alpha_bar = _extract(self.alpha_bars, states, clean)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).clamp_min(0).sqrt() * noise

    def sample_pair(
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
        transitions, _ = self._batch_timesteps(
            transition,
            clean,
            maximum=self.timesteps - 1,
            label="transition",
        )
        previous = self._add_noise(clean, transitions, previous_noise)
        if step_noise is None:
            step_noise = torch.randn_like(clean)
        else:
            self._validate_matching("step_noise", step_noise, clean)

        current_states = transitions + 1
        alpha = _extract(self.alphas, current_states, clean)
        beta = _extract(self.betas, current_states, clean)
        current = alpha.sqrt() * previous + beta.clamp_min(0).sqrt() * step_noise
        return previous, current

    def get_posterior(
        self,
        current: torch.Tensor,
        clean_prediction: torch.Tensor,
        transition: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return parameters of ``q(x_t | x_{t+1}, x0_prediction)``."""
        self._validate_batch("current", current)
        self._validate_matching("clean_prediction", clean_prediction, current)
        transitions, _ = self._batch_timesteps(
            transition,
            current,
            maximum=self.timesteps - 1,
            label="transition",
        )
        return self._get_posterior(current, clean_prediction, transitions)

    def _get_posterior(
        self,
        current: torch.Tensor,
        clean_prediction: torch.Tensor,
        transitions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_states = transitions + 1

        alpha_bar_previous = _extract(self.alpha_bars, transitions, current)
        alpha_bar_current = _extract(self.alpha_bars, current_states, current)
        alpha = _extract(self.alphas, current_states, current)
        beta = _extract(self.betas, current_states, current)
        denominator = (1.0 - alpha_bar_current).clamp_min(
            torch.finfo(current.dtype).tiny
        )

        clean_coefficient = beta * alpha_bar_previous.sqrt() / denominator
        current_coefficient = alpha.sqrt() * (1.0 - alpha_bar_previous) / denominator
        mean = clean_coefficient * clean_prediction + current_coefficient * current
        variance = (beta * (1.0 - alpha_bar_previous) / denominator).clamp_min(0)
        return mean, variance

    def sample_posterior(
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
        self._validate_batch("current", current)
        self._validate_matching("clean_prediction", clean_prediction, current)
        transitions, maximum_transition = self._batch_timesteps(
            transition,
            current,
            maximum=self.timesteps - 1,
            label="transition",
        )
        mean, variance = self._get_posterior(
            current,
            clean_prediction,
            transitions,
        )
        stochastic = transitions != 0
        if maximum_transition == 0:
            return clean_prediction
        if noise is None:
            noise = torch.randn_like(current)
        else:
            self._validate_matching("noise", noise, current)

        stochastic = stochastic.to(current.dtype).reshape(
            current.shape[0],
            *([1] * (current.ndim - 1)),
        )
        sample = mean + stochastic * variance.sqrt() * noise
        return torch.where(
            stochastic.bool(),
            sample,
            clean_prediction,
        )

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        terminal: torch.Tensor,
        latent_channels: int,
        *,
        model_kwargs: Mapping[str, object] | None = None,
        project: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run ``x_T -> ... -> x_0`` using a fresh latent at every step.

        The model is called as ``model(x_current, transition, z, **kwargs)``
        and must predict a clean tensor with the same shape as ``x_current``.
        """
        self._validate_batch("terminal", terminal)
        if (
            not isinstance(latent_channels, int)
            or isinstance(latent_channels, bool)
            or latent_channels < 1
        ):
            raise ValueError("latent_channels must be a positive integer.")
        if project is not None and not callable(project):
            raise TypeError("project must be callable or None.")
        current = terminal
        kwargs = {} if model_kwargs is None else dict(model_kwargs)

        for transition in reversed(range(self.timesteps)):
            times = torch.full(
                (current.shape[0],),
                transition,
                device=current.device,
                dtype=torch.long,
            )
            latent = torch.randn(
                current.shape[0],
                latent_channels,
                device=current.device,
                dtype=current.dtype,
            )
            clean = model(current, times, latent, **kwargs)
            if not isinstance(clean, torch.Tensor):
                raise TypeError("model must return a torch.Tensor.")
            self._validate_matching("model output", clean, current)
            if project is not None:
                clean = project(clean)
                self._validate_matching("projected model output", clean, current)
            current = self.sample_posterior(current, clean, transition)
        return current

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
    ) -> tuple[torch.Tensor, int]:
        if isinstance(value, Integral) and not isinstance(value, bool):
            highest = int(value)
            if highest < 0 or highest > maximum:
                raise ValueError(f"{label} must be between 0 and {maximum}.")
            result = torch.full(
                (reference.shape[0],),
                highest,
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
            if value.ndim != 0 and not (
                value.ndim == 1 and value.shape[0] == reference.shape[0]
            ):
                raise ValueError(
                    f"{label} must be scalar or have one value per batch item."
                )
            lower, upper = torch.stack(torch.aminmax(value)).tolist()
            if lower < 0 or upper > maximum:
                raise ValueError(f"{label} must be between 0 and {maximum}.")
            highest = int(upper)
            if value.ndim == 0:
                result = value.to(reference.device, dtype=torch.long).expand(
                    reference.shape[0]
                )
            else:
                result = value.to(reference.device, dtype=torch.long)
        else:
            raise TypeError(f"{label} must be an integer or torch.Tensor.")

        return result, highest
