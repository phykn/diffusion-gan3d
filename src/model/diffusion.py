import torch
from torch import nn

from src.device import assert_all


class Diffusion(nn.Module):
    def __init__(self, timesteps, beta_min, beta_max):
        super().__init__()
        if not beta_min < beta_max:
            raise ValueError("beta_min must be below beta_max.")

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
        transitions = self.prepare_transitions(transition, clean)
        previous = self.mix_noise(clean, transitions, previous_noise)
        step_noise = self._validate_shape("step_noise", step_noise, clean)
        states = transitions + 1
        dtype = self._coefficient_dtype(clean)
        alpha = self._extract(self.alphas, states, clean, dtype=dtype)
        beta = self._extract(self.betas, states, clean, dtype=dtype)
        current = (
            alpha.sqrt().to(clean.dtype) * previous
            + beta.sqrt().to(clean.dtype) * step_noise
        )
        return previous, current

    def sample_posterior(self, current, prediction, transition, noise=None):
        prediction = self._validate_shape("pred", prediction, current)
        if type(transition) is int and transition == 0:
            return prediction
        transitions = self.prepare_transitions(transition, current)
        if transitions.device.type == "cpu" and not bool(transitions.any()):
            return prediction
        if noise is None:
            raise ValueError("noise is required for transitions above zero.")
        noise = self._validate_shape("noise", noise, current)
        mean, variance = self.get_posterior(current, prediction, transitions)
        is_noisy = (transitions != 0).reshape((-1,) + (1,) * (current.ndim - 1))
        sample = mean + variance.sqrt().to(current.dtype) * noise
        return torch.where(is_noisy, sample, prediction)

    def mix_noise(self, clean, states, noise):
        noise = self._validate_shape("noise", noise, clean)
        alpha_bar = self._extract(self.alpha_bars, states, clean)
        noise_std = self._extract(self.noise_std, states, clean)
        return alpha_bar.sqrt() * clean + noise_std * noise

    def get_posterior(self, current, prediction, transitions):
        clean_coef = self._extract(self.posterior_clean_coef, transitions, current)
        current_coef = self._extract(self.posterior_current_coef, transitions, current)
        mean = clean_coef * prediction + current_coef * current
        variance = self._extract(
            self.posterior_variance,
            transitions,
            current,
            dtype=self._coefficient_dtype(current),
        )
        return mean, variance

    def prepare_transitions(self, value, like):
        limit, name = self.timesteps - 1, "transition"
        batch = like.shape[0]
        if type(value) is int:
            if not 0 <= value <= limit:
                raise ValueError(f"{name} must be between 0 and {limit}.")
            return torch.full((batch,), value, device=like.device, dtype=torch.long)
        time = torch.as_tensor(value, device=like.device)
        if time.dtype == torch.bool or time.is_floating_point() or time.is_complex():
            raise ValueError(f"{name} must contain integer indices.")
        time = time.to(dtype=torch.long)
        if time.ndim == 0:
            time = time.expand(batch)
        elif time.shape != (batch,):
            raise ValueError(f"{name} must be scalar or have one value per batch item.")
        in_range = ((time >= 0) & (time <= limit)).all()
        assert_all(in_range, f"{name} must be between 0 and {limit}.")
        return time

    @staticmethod
    def _extract(schedule, timesteps, like, dtype=None):
        per_item = schedule.to(
            device=like.device,
            dtype=like.dtype if dtype is None else dtype,
        ).index_select(0, timesteps.to(like.device, dtype=torch.long))
        return per_item.reshape((like.shape[0],) + (1,) * (like.ndim - 1))

    @staticmethod
    def _validate_shape(name, values, like):
        if values is None or values.shape != like.shape:
            raise ValueError(f"{name} must have shape {tuple(like.shape)}.")
        return values

    @staticmethod
    def _coefficient_dtype(values):
        # Half-precision noise coefficients lose the small step sizes.
        if values.dtype in (torch.float16, torch.bfloat16):
            return torch.float32
        return values.dtype
