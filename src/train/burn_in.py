"""Burn-in schedule for optimizer, regularization and EMA settings (R3GAN).

Over the first `steps` steps a cosine curve moves the learning rates, Adam β2,
the R1/R2 weights and the EMA half-life from their start values to their end
values; afterwards they stay at the end values. Large early steps need a low
β2, and a smaller learning rate and weaker regularization later slow the critic
down once the generator has the structure (Huang et al., The GAN is dead; long
live the GAN!, 2024, Appendix D).
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BurnInValues:
    generator_lr: float
    critic_lr: float
    betas: tuple[float, float]
    regularization_scale: float
    ema_decay: float


@dataclass(frozen=True)
class BurnIn:
    steps: int
    generator_lr: float
    critic_lr: float
    betas: tuple[float, float]
    ema_decay: float
    lr_scale: float
    beta2: float
    regularization_scale: float

    def progress(self, step: int) -> float:
        """0 at step 0, rising along a cosine to 1 at `steps` and after."""
        if self.steps == 0:
            return 1.0
        return 0.5 - 0.5 * math.cos(math.pi * min(step / self.steps, 1.0))

    def values(self, step: int) -> BurnInValues:
        progress = self.progress(step)

        def move(start, end):
            return start + (end - start) * progress

        lr_scale = move(1.0, self.lr_scale)
        # The EMA half-life grows from zero, so early averages hold no weights
        # from the random initialization; ema_decay sets the final half-life.
        half_life = (
            progress * math.log(0.5) / math.log(self.ema_decay)
            if self.ema_decay > 0
            else 0.0
        )
        return BurnInValues(
            generator_lr=self.generator_lr * lr_scale,
            critic_lr=self.critic_lr * lr_scale,
            betas=(self.betas[0], move(self.betas[1], self.beta2)),
            regularization_scale=move(1.0, self.regularization_scale),
            ema_decay=0.5 ** (1.0 / half_life) if half_life > 0 else 0.0,
        )
