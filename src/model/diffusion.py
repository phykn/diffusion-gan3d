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
        if beta_max <= beta_min:
            raise ValueError("beta_max must be greater than beta_min.")

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
            self.check_matching("step_noise", step_noise, clean)

        states = transitions + 1
        alpha = self.extract(self.alphas, states, clean)
        beta = self.extract(self.betas, states, clean)
        current = alpha.sqrt() * previous + beta.sqrt() * step_noise
        return previous, current

    def sample_posterior(
        self,
        current: torch.Tensor,
        pred: torch.Tensor,
        transition: int | torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.check_matching("pred", pred, current)
        transitions = self.prepare_time(
            transition,
            current,
            self.timesteps - 1,
            "transition",
        )
        if not bool(transitions.any()):
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

        active = active.reshape((-1,) + (1,) * (current.ndim - 1))
        sample = mean + variance.sqrt() * noise
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
            self.check_matching("noise", noise, clean)

        alpha_bar = self.extract(self.alpha_bars, states, clean)
        return alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise

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
        return coeff.reshape((ref.shape[0],) + (1,) * (ref.ndim - 1))

    @staticmethod
    def check_matching(
        name: str,
        values: torch.Tensor,
        ref: torch.Tensor,
    ) -> None:
        if values.shape != ref.shape:
            raise ValueError(f"{name} must have shape {tuple(ref.shape)}.")

    @staticmethod
    def prepare_time(
        value: int | torch.Tensor,
        ref: torch.Tensor,
        limit: int,
        name: str,
    ) -> torch.Tensor:
        time = torch.as_tensor(value, device=ref.device, dtype=torch.long)
        if time.ndim == 0:
            time = time.expand(ref.shape[0])
        elif time.shape != (ref.shape[0],):
            raise ValueError(
                f"{name} must be scalar or have one value per batch item."
            )
        if bool(((time < 0) | (time > limit)).any()):
            raise ValueError(f"{name} must be between 0 and {limit}.")
        return time
