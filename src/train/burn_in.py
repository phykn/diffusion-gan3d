import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BurnInValues:
    generator_lr: float
    critic_lr: float
    betas: tuple[float, float]
    regularization_scale: float
    ema_decay: float


def burn_in_values(config, step: int) -> BurnInValues:
    """Over the first burn_in.steps steps a cosine curve moves the learning rates,
    Adam beta2, the R1/R2 weights and the EMA half-life from their start values to
    their end values; afterwards they stay at the end values. Large early steps
    need a low beta2, and a smaller learning rate and weaker regularization later
    slow the critic down once the generator has the structure (Huang et al., The
    GAN is dead; long live the GAN!, 2024, Appendix D).
    """
    optim, burn_in = config.optim, config.burn_in
    progress = (
        0.5 - 0.5 * math.cos(math.pi * min(step / burn_in.steps, 1.0))
        if burn_in.steps
        else 1.0
    )

    def move(start, end):
        return start + (end - start) * progress

    lr_scale = move(1.0, burn_in.lr_scale)
    # The EMA half-life grows from zero, so early averages hold no weights
    # from the random initialization; ema_decay sets the final half-life.
    half_life = (
        progress * math.log(0.5) / math.log(optim.ema_decay)
        if optim.ema_decay > 0
        else 0.0
    )
    beta1, beta2 = optim.adam_betas
    return BurnInValues(
        generator_lr=optim.generator_lr * lr_scale,
        critic_lr=optim.critic_lr * lr_scale,
        betas=(beta1, move(beta2, burn_in.beta2)),
        regularization_scale=move(1.0, burn_in.regularization_scale),
        ema_decay=0.5 ** (1.0 / half_life) if half_life > 0 else 0.0,
    )
