import math

import torch
from torch import nn

from src.device import assert_all


def _stats_dtype(values):
    # Half-precision noise coefficients lose the small step sizes.
    if values.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return values.dtype


class Diffusion(nn.Module):
    def __init__(self, timesteps, beta_min=0.1, beta_max=20.0):
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

        time = torch.linspace(0.0, 1.0, timesteps + 1, dtype=torch.float64)
        log_alpha_bars = -0.5 * (beta_max - beta_min) * time.square() - beta_min * time
        alpha_bars = log_alpha_bars.exp()
        log_step = log_alpha_bars.diff()
        alphas = torch.cat((alpha_bars[:1], log_step.exp()))
        betas = torch.cat((time[:1], -log_step.expm1()))
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

    def sample_pair(self, clean, transition, previous_noise, step_noise):
        transitions = self.prepare_time(
            transition, clean, self.timesteps - 1, "transition"
        )
        previous = self.mix_noise(clean, transitions, previous_noise)
        step_noise = self.validate_shape("step_noise", step_noise, clean)
        states = transitions + 1
        dtype = _stats_dtype(clean)
        alpha = self.extract(self.alphas, states, clean, dtype=dtype)
        beta = self.extract(self.betas, states, clean, dtype=dtype)
        current = (
            alpha.sqrt().to(clean.dtype) * previous
            + beta.sqrt().to(clean.dtype) * step_noise
        )
        return previous, current

    def sample_posterior(self, current, pred, transition, noise=None):
        pred = self.validate_shape("pred", pred, current)
        if type(transition) is int and transition == 0:
            return pred
        transitions = self.prepare_time(
            transition, current, self.timesteps - 1, "transition"
        )
        if transitions.device.type == "cpu" and not bool(transitions.any()):
            return pred
        if noise is None:
            raise ValueError("noise is required for transitions above zero.")
        noise = self.validate_shape("noise", noise, current)
        mean, variance = self.get_posterior(current, pred, transitions)
        active = (transitions != 0).reshape((-1,) + (1,) * (current.ndim - 1))
        sample = mean + variance.sqrt().to(current.dtype) * noise
        return torch.where(active, sample, pred)

    def mix_noise(self, clean, states, noise):
        noise = self.validate_shape("noise", noise, clean)
        alpha_bar = self.extract(self.alpha_bars, states, clean)
        std = self.extract(self.noise_std, states, clean)
        return alpha_bar.sqrt() * clean + std * noise

    def get_posterior(self, current, pred, transitions):
        clean_coef = self.extract(self.posterior_clean_coef, transitions, current)
        current_coef = self.extract(self.posterior_current_coef, transitions, current)
        mean = clean_coef * pred + current_coef * current
        variance = self.extract(
            self.posterior_variance, transitions, current, dtype=_stats_dtype(current)
        )
        return mean, variance

    @staticmethod
    def extract(values, timesteps, ref, dtype=None):
        coeff = values.to(
            device=ref.device,
            dtype=ref.dtype if dtype is None else dtype,
        ).index_select(0, timesteps.to(ref.device, dtype=torch.long))
        return coeff.reshape((ref.shape[0],) + (1,) * (ref.ndim - 1))

    @staticmethod
    def validate_shape(name, values, ref):
        if values is None or values.shape != ref.shape:
            raise ValueError(f"{name} must have shape {tuple(ref.shape)}.")
        return values

    @staticmethod
    def prepare_time(value, ref, limit, name):
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
        assert_all(valid, f"{name} must be between 0 and {limit}.")
        return time
