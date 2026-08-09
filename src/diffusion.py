import math

import torch
from torch import nn


class Diffusion(nn.Module):
    """States use ``0..T`` so forward and reverse transitions share one index."""

    def __init__(
        self,
        timesteps: int,
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
        log_alpha_bars = (
            -0.5 * (beta_max - beta_min) * time.square() - beta_min * time
        )
        minimum_log = math.log(torch.finfo(torch.float32).tiny)
        if float(log_alpha_bars[-1]) < minimum_log:
            raise ValueError(
                "diffusion schedule is not representable in float32; "
                "reduce beta_max."
            )
        alpha_bars = log_alpha_bars.exp()
        alphas = torch.ones_like(alpha_bars)
        alphas[1:] = (log_alpha_bars[1:] - log_alpha_bars[:-1]).exp()
        betas = 1.0 - alphas

        self.timesteps = timesteps
        self.register_buffer("alpha_bars", alpha_bars.to(torch.float32))
        self.register_buffer("alphas", alphas.to(torch.float32))
        self.register_buffer("betas", betas.to(torch.float32))

    def add_noise(
        self,
        clean: torch.Tensor,
        state: int | torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.check_batch("clean", clean)
        states, _ = self.prepare_time(
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
        """The critic needs one Markov pair, not two independent noisy views."""
        self.check_batch("clean", clean)
        transitions, _ = self.prepare_time(
            transition,
            clean,
            self.timesteps - 1,
            "transition",
        )
        previous = self.mix_noise(clean, transitions, previous_noise)
        if step_noise is None:
            step_noise = torch.randn_like(clean)
        else:
            self.check_matching("step_noise", step_noise, clean)

        states = transitions + 1
        alpha = self.extract(self.alphas, states, clean)
        beta = self.extract(self.betas, states, clean)
        current = alpha.sqrt() * previous + beta.clamp_min(0).sqrt() * step_noise
        return previous, current

    def sample_posterior(
        self,
        current: torch.Tensor,
        pred: torch.Tensor,
        transition: int | torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Transition zero bypasses noise so the final state is exactly ``pred``."""
        self.check_batch("current", current)
        self.check_matching("pred", pred, current)
        transitions, max_transition = self.prepare_time(
            transition,
            current,
            self.timesteps - 1,
            "transition",
        )
        if max_transition == 0:
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
            self.check_matching("noise", noise, current)

        active = active.reshape(
            current.shape[0],
            *([1] * (current.ndim - 1)),
        )
        sample = mean + variance.sqrt() * noise
        return torch.where(active, sample, pred)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        initial_noise: torch.Tensor,
        latent_channels: int,
        conditions: dict[str, object] | None = None,
        known_clean: torch.Tensor | None = None,
        known_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample while keeping an optional known region on its DDPM bridge."""
        self.check_batch("initial_noise", initial_noise)
        if (
            not isinstance(latent_channels, int)
            or isinstance(latent_channels, bool)
            or latent_channels < 1
        ):
            raise ValueError("latent_channels must be a positive integer.")
        if (known_clean is None) != (known_mask is None):
            raise ValueError("known_clean and known_mask must be provided together.")
        current = initial_noise
        if known_clean is not None and known_mask is not None:
            known_at_start = self.add_noise(
                known_clean,
                self.timesteps,
                noise=initial_noise,
            )
            current = self.blend_known(current, known_at_start, known_mask)
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
            if known_clean is not None and known_mask is not None:
                pred = self.blend_known(pred, known_clean, known_mask)
            current = self.sample_posterior(
                current,
                pred,
                transition,
            )
        return current

    @classmethod
    def blend_known(
        cls,
        values: torch.Tensor,
        known: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Replace or blend a known region without changing unmasked values."""
        cls.check_batch("values", values)
        cls.check_matching("known", known, values)
        if not isinstance(mask, torch.Tensor):
            raise TypeError("mask must be a torch.Tensor.")
        expected = (values.shape[0], 1, *values.shape[2:])
        if mask.shape not in (expected, values.shape):
            raise ValueError(
                f"mask must have shape {expected} or {tuple(values.shape)}."
            )
        if mask.device != values.device:
            raise ValueError("mask and values must use the same device.")
        if mask.dtype == torch.bool:
            return torch.where(mask, known, values)
        if not mask.is_floating_point():
            raise TypeError("mask must use a boolean or floating-point dtype.")
        if not bool(torch.isfinite(mask).all()):
            raise ValueError("mask values must be finite.")
        if bool(((mask < 0.0) | (mask > 1.0)).any()):
            raise ValueError("mask values must be between zero and one.")
        return torch.lerp(
            values,
            known,
            mask.to(dtype=values.dtype),
        )

    def mix_noise(
        self,
        clean: torch.Tensor,
        states: torch.Tensor,
        noise: torch.Tensor | None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean)
        else:
            self.check_matching("noise", noise, clean)

        alpha_bar = self.extract(self.alpha_bars, states, clean)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).clamp_min(0).sqrt() * noise

    def get_posterior(
        self,
        current: torch.Tensor,
        pred: torch.Tensor,
        transitions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        states = transitions + 1

        alpha_bar_prev = self.extract(self.alpha_bars, transitions, current)
        alpha_bar_curr = self.extract(self.alpha_bars, states, current)
        alpha = self.extract(self.alphas, states, current)
        beta = self.extract(self.betas, states, current)
        denom = (1.0 - alpha_bar_curr).clamp_min(torch.finfo(current.dtype).tiny)

        clean_coef = beta * alpha_bar_prev.sqrt() / denom
        current_coef = alpha.sqrt() * (1.0 - alpha_bar_prev) / denom
        mean = clean_coef * pred + current_coef * current
        variance = (beta * (1.0 - alpha_bar_prev) / denom).clamp_min(0)
        return mean, variance

    @staticmethod
    def extract(
        values: torch.Tensor,
        timesteps: torch.Tensor,
        ref: torch.Tensor,
    ) -> torch.Tensor:
        coeff = values.to(
            device=ref.device,
            dtype=ref.dtype,
        ).index_select(0, timesteps.to(ref.device, dtype=torch.long))
        return coeff.reshape(ref.shape[0], *([1] * (ref.ndim - 1)))

    @staticmethod
    def check_batch(name: str, values: torch.Tensor) -> None:
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if values.ndim < 2:
            raise ValueError(f"{name} must have batch and channel dimensions.")
        if values.shape[0] < 1:
            raise ValueError(f"{name} must contain at least one batch item.")
        if not values.is_floating_point():
            raise ValueError(f"{name} must use a floating-point dtype.")

    @classmethod
    def check_matching(
        cls,
        name: str,
        values: torch.Tensor,
        ref: torch.Tensor,
    ) -> None:
        cls.check_batch(name, values)
        if values.shape != ref.shape:
            raise ValueError(f"{name} must have shape {tuple(ref.shape)}.")
        if values.device != ref.device:
            raise ValueError(f"{name} and the reference must use the same device.")
        if values.dtype != ref.dtype:
            raise ValueError(f"{name} and the reference must use the same dtype.")

    @staticmethod
    def prepare_time(
        value: int | torch.Tensor,
        ref: torch.Tensor,
        limit: int,
        name: str,
    ) -> tuple[torch.Tensor, int]:
        if isinstance(value, int) and not isinstance(value, bool):
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
