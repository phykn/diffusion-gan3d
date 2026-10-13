import math

import torch
from torch import nn


class Diffusion(nn.Module):
    def __init__(
        self,
        timesteps: int,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
    ) -> None:
        super().__init__()
        if (
            isinstance(timesteps, bool)
            or not isinstance(timesteps, int)
            or timesteps < 1
        ):
            raise ValueError("timesteps must be a positive integer.")
        if not (
            math.isfinite(beta_min)
            and math.isfinite(beta_max)
            and 0 <= beta_min < beta_max
        ):
            raise ValueError(
                "beta_min/beta_max must be finite with 0 <= beta_min < beta_max."
            )

        time = torch.linspace(
            0.0,
            1.0,
            timesteps + 1,
            dtype=torch.float64,
        )
        log_alpha_bars = -0.5 * (beta_max - beta_min) * time.square() - beta_min * time
        alpha_bars = log_alpha_bars.exp()
        alphas = torch.ones_like(alpha_bars)
        alphas[1:] = (log_alpha_bars[1:] - log_alpha_bars[:-1]).exp()
        betas = torch.zeros_like(alphas)
        betas[1:] = -(log_alpha_bars[1:] - log_alpha_bars[:-1]).expm1()
        variances = -log_alpha_bars.expm1()
        clean_coef = betas[1:] * alpha_bars[:-1].sqrt() / variances[1:]
        current_coef = alphas[1:].sqrt() * variances[:-1] / variances[1:]
        posterior_variance = betas[1:] * variances[:-1] / variances[1:]

        self.timesteps = timesteps
        self.register_buffer("alpha_bars", alpha_bars.to(torch.float32))
        self.register_buffer("alphas", alphas.to(torch.float32))
        self.register_buffer("betas", betas.to(torch.float32))
        # Form differences and ratios before rounding near-one schedule values.
        for name, values in (
            ("noise_std", variances.sqrt()),
            ("posterior_clean_coef", clean_coef),
            ("posterior_current_coef", current_coef),
            ("posterior_variance", posterior_variance),
        ):
            self.register_buffer(name, values.to(torch.float32), persistent=False)

    def add_noise(
        self,
        clean: torch.Tensor,
        state: int | torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        states = self.prepare_time(
            state,
            clean,
            self.timesteps,
            "state",
        )
        return self.mix_noise(clean, states, noise)

    def sample_pair(
        self,
        clean: torch.Tensor,
        transition: int | torch.Tensor,
        previous_noise: torch.Tensor | None = None,
        step_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transitions = self.prepare_time(
            transition,
            clean,
            self.timesteps - 1,
            "transition",
        )
        previous = self.mix_noise(clean, transitions, previous_noise)
        if step_noise is None:
            step_noise = torch.randn_like(clean)
        else:
            step_noise = self.validate_shape("step_noise", step_noise, clean)

        states = transitions + 1
        dtype = (
            torch.float32
            if clean.dtype in (torch.float16, torch.bfloat16)
            else clean.dtype
        )
        alpha = self.extract(self.alphas, states, clean, dtype=dtype)
        beta = self.extract(self.betas, states, clean, dtype=dtype)
        current = (
            alpha.sqrt().to(clean.dtype) * previous
            + beta.sqrt().to(clean.dtype) * step_noise
        )
        return previous, current

    def sample_posterior(
        self,
        current: torch.Tensor,
        pred: torch.Tensor,
        transition: int | torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred = self.validate_shape("pred", pred, current)
        transitions = self.prepare_time(
            transition,
            current,
            self.timesteps - 1,
            "transition",
        )
        if isinstance(transition, int) and transition == 0:
            return pred
        if transitions.device.type == "cpu" and not bool(transitions.any()):
            return pred
        mean, variance = self.get_posterior(
            current,
            pred,
            transitions,
        )
        active = transitions != 0
        if noise is None:
            noise = torch.randn_like(current)
        else:
            noise = self.validate_shape("noise", noise, current)

        active = active.reshape((-1,) + (1,) * (current.ndim - 1))
        sample = mean + variance.sqrt().to(current.dtype) * noise
        return torch.where(active, sample, pred)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        initial_noise: torch.Tensor,
        latent_channels: int,
        conditions: dict[str, object] | None = None,
    ) -> torch.Tensor:
        current = initial_noise
        cond = {} if conditions is None else conditions

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
            pred = model(
                current,
                time,
                latent,
                **cond,
            )
            current = self.sample_posterior(
                current,
                pred,
                transition,
            )
        return current

    def mix_noise(
        self,
        clean: torch.Tensor,
        states: torch.Tensor,
        noise: torch.Tensor | None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean)
        else:
            noise = self.validate_shape("noise", noise, clean)

        alpha_bar = self.extract(self.alpha_bars, states, clean)
        std = self.extract(self.noise_std, states, clean)
        return alpha_bar.sqrt() * clean + std * noise

    def get_posterior(
        self,
        current: torch.Tensor,
        pred: torch.Tensor,
        transitions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clean_coef = self.extract(self.posterior_clean_coef, transitions, current)
        current_coef = self.extract(self.posterior_current_coef, transitions, current)
        mean = clean_coef * pred + current_coef * current
        dtype = (
            torch.float32
            if current.dtype in (torch.float16, torch.bfloat16)
            else current.dtype
        )
        variance = self.extract(
            self.posterior_variance, transitions, current, dtype=dtype
        )
        return mean, variance

    @staticmethod
    def extract(
        values: torch.Tensor,
        timesteps: torch.Tensor,
        ref: torch.Tensor,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        coeff = values.to(
            device=ref.device,
            dtype=ref.dtype if dtype is None else dtype,
        ).index_select(0, timesteps.to(ref.device, dtype=torch.long))
        return coeff.reshape((ref.shape[0],) + (1,) * (ref.ndim - 1))

    @staticmethod
    def validate_shape(
        name: str,
        values: torch.Tensor,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        if values.shape != ref.shape:
            raise ValueError(f"{name} must have shape {tuple(ref.shape)}.")
        return values

    @staticmethod
    def prepare_time(
        value: int | torch.Tensor,
        ref: torch.Tensor,
        limit: int,
        name: str,
    ) -> torch.Tensor:
        if type(value) is int:
            if not 0 <= value <= limit:
                raise ValueError(f"{name} must be between 0 and {limit}.")
            return torch.full(
                (ref.shape[0],), value, device=ref.device, dtype=torch.long
            )
        time = torch.as_tensor(value, device=ref.device)
        if time.dtype == torch.bool or time.is_floating_point() or time.is_complex():
            raise ValueError(f"{name} must contain integer indices.")
        time = time.to(dtype=torch.long)
        if time.ndim == 0:
            time = time.expand(ref.shape[0])
        elif time.shape != (ref.shape[0],):
            raise ValueError(f"{name} must be scalar or have one value per batch item.")
        valid = ((time >= 0) & (time <= limit)).all()
        if time.device.type == "cuda":
            torch._assert_async(valid, f"{name} must be between 0 and {limit}.")
        elif not bool(valid):
            raise ValueError(f"{name} must be between 0 and {limit}.")
        return time
