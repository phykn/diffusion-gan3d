import math
from collections.abc import Mapping
from numbers import Integral

import torch
from torch import nn


def _extract(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    ref: torch.Tensor,
) -> torch.Tensor:
    coeff = values.to(
        device=ref.device,
        dtype=ref.dtype,
    ).index_select(0, timesteps.to(ref.device, dtype=torch.long))
    return coeff.reshape(ref.shape[0], *([1] * (ref.ndim - 1)))


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

        time = torch.linspace(
            0.0,
            1.0,
            timesteps + 1,
            dtype=torch.float64,
        )
        alpha_bars = torch.exp(
            -0.5 * (beta_max - beta_min) * time.square() - beta_min * time
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
            limit=self.timesteps,
            name="state",
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
            limit=self.timesteps - 1,
            name="transition",
        )
        previous = self._add_noise(clean, transitions, previous_noise)
        if step_noise is None:
            step_noise = torch.randn_like(clean)
        else:
            self._validate_matching("step_noise", step_noise, clean)

        states = transitions + 1
        alpha = _extract(self.alphas, states, clean)
        beta = _extract(self.betas, states, clean)
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
            limit=self.timesteps - 1,
            name="transition",
        )
        return self._get_posterior(current, clean_prediction, transitions)

    def _get_posterior(
        self,
        current: torch.Tensor,
        clean_prediction: torch.Tensor,
        transitions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = transitions + 1

        alpha_bar_prev = _extract(self.alpha_bars, transitions, current)
        alpha_bar_curr = _extract(self.alpha_bars, states, current)
        alpha = _extract(self.alphas, states, current)
        beta = _extract(self.betas, states, current)
        denom = (1.0 - alpha_bar_curr).clamp_min(torch.finfo(current.dtype).tiny)

        clean_coef = beta * alpha_bar_prev.sqrt() / denom
        current_coef = alpha.sqrt() * (1.0 - alpha_bar_prev) / denom
        mean = clean_coef * clean_prediction + current_coef * current
        variance = (beta * (1.0 - alpha_bar_prev) / denom).clamp_min(0)
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
        transitions, max_transition = self._batch_timesteps(
            transition,
            current,
            limit=self.timesteps - 1,
            name="transition",
        )
        mean, variance = self._get_posterior(
            current,
            clean_prediction,
            transitions,
        )
        stochastic = transitions != 0
        if max_transition == 0:
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
        initial_noise: torch.Tensor,
        latent_channels: int,
        *,
        conditions: Mapping[str, object] | None = None,
    ) -> torch.Tensor:
        """Run ``x_T -> ... -> x_0`` using a fresh latent at every step.

        The predictor may be a denoiser or an adapter that fuses several
        conditional predictions. It is called as
        ``model(x_current, transition, z, **conditions)`` and must return one clean
        tensor with the same shape as ``x_current``.
        """
        self._validate_batch("initial_noise", initial_noise)
        if (
            not isinstance(latent_channels, int)
            or isinstance(latent_channels, bool)
            or latent_channels < 1
        ):
            raise ValueError("latent_channels must be a positive integer.")
        current = initial_noise
        cond = {} if conditions is None else dict(conditions)

        for transition in reversed(range(self.timesteps)):
            time = torch.full(
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
            clean_prediction = model(
                current,
                time,
                latent,
                **cond,
            )
            current = self.sample_posterior(
                current,
                clean_prediction,
                transition,
            )
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
        ref: torch.Tensor,
    ) -> None:
        cls._validate_batch(name, values)
        if values.shape != ref.shape:
            raise ValueError(f"{name} must have shape {tuple(ref.shape)}.")
        if values.device != ref.device:
            raise ValueError(f"{name} and the reference must use the same device.")
        if values.dtype != ref.dtype:
            raise ValueError(f"{name} and the reference must use the same dtype.")

    @staticmethod
    def _batch_timesteps(
        value: int | torch.Tensor,
        ref: torch.Tensor,
        *,
        limit: int,
        name: str,
    ) -> tuple[torch.Tensor, int]:
        if isinstance(value, Integral) and not isinstance(value, bool):
            max_step = int(value)
            if max_step < 0 or max_step > limit:
                raise ValueError(f"{name} must be between 0 and {limit}.")
            time = torch.full(
                (ref.shape[0],),
                max_step,
                device=ref.device,
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
                raise ValueError(f"{name} must use an integer dtype.")
            if value.ndim != 0 and not (
                value.ndim == 1 and value.shape[0] == ref.shape[0]
            ):
                raise ValueError(
                    f"{name} must be scalar or have one value per batch item."
                )
            lower, upper = torch.stack(torch.aminmax(value)).tolist()
            if lower < 0 or upper > limit:
                raise ValueError(f"{name} must be between 0 and {limit}.")
            max_step = int(upper)
            if value.ndim == 0:
                time = value.to(ref.device, dtype=torch.long).expand(ref.shape[0])
            else:
                time = value.to(ref.device, dtype=torch.long)
        else:
            raise TypeError(f"{name} must be an integer or torch.Tensor.")

        return time, max_step
