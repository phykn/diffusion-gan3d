import math
from unittest.mock import patch

import torch
from torch import nn

from src.build import build_models, build_optimizers
from src.diffusion import DiffusionProcess
from src.train import DiffusionGANTrainer, build_ema
from src.train.config import DataConfig, ModelConfig, OptimConfig


class _ConstantStream:
    def __init__(self, labels: torch.Tensor) -> None:
        self.labels = labels

    def next(self) -> torch.Tensor:
        return self.labels.clone()


def test_training_step_updates_denoiser_and_all_critics() -> None:
    data = DataConfig(
        folder={0: ".", 1: ".", 2: "."},
        crop_size=8,
        patch_size=8,
        num_phases=3,
        batch_size=2,
    )
    model = ModelConfig(
        base_channels=4,
        channel_multipliers=(1, 2),
        embedding_channels=8,
        latent_channels=4,
        critic_channels=(4, 8),
        gradient_checkpointing=False,
    )
    optim = OptimConfig(
        generator_lr=1e-3,
        critic_lr=1e-3,
        beta1=0.0,
        beta2=0.9,
        r1_gamma=0.01,
        r1_interval=1,
    )
    denoiser, critics = build_models(data, model)
    ema = build_ema(denoiser)
    denoiser_optimizer, critic_optimizers = build_optimizers(
        denoiser,
        critics,
        optim,
    )
    labels = torch.randint(0, data.num_phases, (data.batch_size, 8, 8))
    streams = {axis: _ConstantStream(labels) for axis in (0, 1, 2)}
    trainer = DiffusionGANTrainer(
        denoiser=denoiser,
        ema_denoiser=ema,
        critics=critics,
        streams=streams,
        diffusion=DiffusionProcess(2, beta_min=0.1, beta_max=2.0),
        denoiser_optimizer=denoiser_optimizer,
        critic_optimizers=critic_optimizers,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        num_phases=data.num_phases,
        patch_size=data.patch_size,
        latent_channels=model.latent_channels,
        volume_batch_size=1,
        slices_per_axis=2,
        mixed_precision=False,
        ema_decay=0.9,
        r1_gamma=optim.r1_gamma,
        r1_interval=optim.r1_interval,
        device=torch.device("cpu"),
    )
    denoiser_before = _parameters(denoiser)
    critic_before = {
        axis: _parameters(critics[str(axis)]) for axis in (0, 1, 2)
    }

    metrics = trainer.train_step(0)

    assert math.isfinite(metrics.generator)
    assert math.isfinite(metrics.critic)
    assert math.isfinite(metrics.r1)
    assert 0 <= metrics.transition < 2
    assert _changed(denoiser_before, denoiser)
    assert all(
        _changed(critic_before[axis], critics[str(axis)])
        for axis in (0, 1, 2)
    )
    assert all(not parameter.requires_grad for parameter in ema.parameters())


def test_anchor_training_uses_real_plane_and_updates_adapter() -> None:
    data = DataConfig(
        folder={0: ".", 1: ".", 2: "."},
        crop_size=8,
        patch_size=8,
        num_phases=3,
        batch_size=2,
    )
    model = ModelConfig(
        base_channels=4,
        channel_multipliers=(1, 2),
        embedding_channels=8,
        latent_channels=4,
        critic_channels=(4, 8),
        gradient_checkpointing=False,
    )
    optim = OptimConfig(
        generator_lr=1e-3,
        critic_lr=1e-3,
        beta1=0.0,
        beta2=0.9,
        r1_gamma=0.0,
        r1_interval=2,
    )
    denoiser, critics = build_models(data, model)
    ema = build_ema(denoiser)
    denoiser_optimizer, critic_optimizers = build_optimizers(
        denoiser,
        critics,
        optim,
    )
    labels = torch.randint(0, data.num_phases, (data.batch_size, 8, 8))
    trainer = DiffusionGANTrainer(
        denoiser=denoiser,
        ema_denoiser=ema,
        critics=critics,
        streams={axis: _ConstantStream(labels) for axis in (0, 1, 2)},
        diffusion=DiffusionProcess(2, beta_min=0.1, beta_max=2.0),
        denoiser_optimizer=denoiser_optimizer,
        critic_optimizers=critic_optimizers,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        num_phases=data.num_phases,
        patch_size=data.patch_size,
        latent_channels=model.latent_channels,
        volume_batch_size=1,
        slices_per_axis=2,
        mixed_precision=False,
        ema_decay=0.9,
        r1_gamma=optim.r1_gamma,
        r1_interval=optim.r1_interval,
        device=torch.device("cpu"),
        anchor_probability=1.0,
        anchor_weight=1.0,
    )
    adapter_before = denoiser.anchor_input.weight.detach().clone()

    with (
        patch.object(
            denoiser,
            "forward",
            wraps=denoiser.forward,
        ) as forward,
        patch.object(
            denoiser,
            "forward_logits",
            wraps=denoiser.forward_logits,
        ) as forward_logits,
    ):
        metrics = trainer.train_step(0, transition=0)

    assert metrics.anchor_used
    assert forward.call_count == 1
    assert forward_logits.call_count == 2
    assert all(
        call.kwargs["anchor_image"] is not None
        and call.kwargs["anchor_mask"] is not None
        for call in forward_logits.call_args_list
    )
    assert math.isfinite(metrics.anchor_loss)
    assert 0.0 <= metrics.anchor_accuracy <= 1.0
    assert math.isclose(
        metrics.generator_total,
        metrics.generator + metrics.anchor_loss,
        rel_tol=1e-5,
    )
    assert not torch.equal(adapter_before, denoiser.anchor_input.weight.detach())


def _parameters(model: nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in model.parameters())


def _changed(before: tuple[torch.Tensor, ...], model: nn.Module) -> bool:
    return any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.parameters(), strict=True)
    )
