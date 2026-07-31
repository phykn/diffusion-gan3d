import math
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from src.build import build_models, build_optimizers
from src.diffusion import Diffusion
from src.train.config import (
    AnchorConfig,
    DataConfig,
    DiffusionConfig,
    LoopConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)
from src.train.ema import build_ema
from src.train.engine import Trainer
from src.train.loss import critic_r1_penalty


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
        critic_local_weight=0.5,
    )
    cfg = _config(data, model, optim)
    denoiser, critics = build_models(cfg)
    ema = build_ema(denoiser)
    denoiser_optimizer, critic_optimizers = build_optimizers(
        denoiser,
        critics,
        cfg,
    )
    labels = torch.randint(0, data.num_phases, (data.batch_size, 8, 8))
    streams = {axis: _ConstantStream(labels) for axis in (0, 1, 2)}
    trainer = _make_trainer(
        cfg,
        denoiser=denoiser,
        ema_denoiser=ema,
        critics=critics,
        streams=streams,
        diffusion=Diffusion(2, beta_min=0.1, beta_max=2.0),
        denoiser_optimizer=denoiser_optimizer,
        critic_optimizers=critic_optimizers,
        device=torch.device("cpu"),
    )
    denoiser_before = _parameters(denoiser)
    critic_before = {axis: _parameters(critics[str(axis)]) for axis in (0, 1, 2)}
    local_before = {
        axis: _parameters(critics[str(axis)].local_output) for axis in (0, 1, 2)
    }

    r1_values = []

    def track_r1(scores, inputs):
        penalties = critic_r1_penalty(scores, inputs)
        r1_values.append(float(penalties.total(optim.critic_local_weight).detach()))
        return penalties

    with patch("src.train.engine.critic_r1_penalty", side_effect=track_r1):
        metrics = trainer.step(0)

    assert math.isfinite(metrics.generator)
    assert math.isfinite(metrics.critic)
    assert math.isfinite(metrics.r1)
    assert math.isfinite(metrics.generator_global)
    assert math.isfinite(metrics.generator_local)
    assert math.isfinite(metrics.critic_global)
    assert math.isfinite(metrics.critic_local)
    assert len(r1_values) == 3
    assert math.isclose(metrics.r1, sum(r1_values), rel_tol=1e-6)
    assert math.isclose(
        metrics.generator,
        metrics.generator_global + optim.critic_local_weight * metrics.generator_local,
        rel_tol=1e-5,
    )
    assert math.isclose(
        metrics.critic,
        metrics.critic_global
        + optim.critic_local_weight * metrics.critic_local
        + 0.5 * optim.r1_gamma * optim.r1_interval * metrics.r1,
        rel_tol=1e-5,
    )
    assert 0 <= metrics.transition < 2
    assert _changed(denoiser_before, denoiser)
    assert all(_changed(critic_before[axis], critics[str(axis)]) for axis in (0, 1, 2))
    assert all(
        _changed(local_before[axis], critics[str(axis)].local_output)
        for axis in (0, 1, 2)
    )
    assert all(not parameter.requires_grad for parameter in ema.parameters())


def test_critic_warmup_updates_critics_without_changing_denoiser() -> None:
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
        critic_local_weight=0.5,
    )
    cfg = _config(data, model, optim)
    denoiser, critics = build_models(cfg)
    ema = build_ema(denoiser)
    denoiser_optimizer, critic_optimizers = build_optimizers(
        denoiser,
        critics,
        cfg,
    )
    labels = torch.randint(0, data.num_phases, (data.batch_size, 8, 8))
    trainer = _make_trainer(
        cfg,
        denoiser=denoiser,
        ema_denoiser=ema,
        critics=critics,
        streams={axis: _ConstantStream(labels) for axis in (0, 1, 2)},
        diffusion=Diffusion(2, beta_min=0.1, beta_max=2.0),
        denoiser_optimizer=denoiser_optimizer,
        critic_optimizers=critic_optimizers,
        device=torch.device("cpu"),
    )
    denoiser_before = _parameters(denoiser)
    critic_before = {axis: _parameters(critics[str(axis)]) for axis in (0, 1, 2)}

    trainer.warm_critics(1)

    assert not _changed(denoiser_before, denoiser)
    assert all(_changed(critic_before[axis], critics[str(axis)]) for axis in (0, 1, 2))


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
        critic_local_weight=0.5,
    )
    cfg = _config(
        data,
        model,
        optim,
        anchor=AnchorConfig(
            probability=1.0,
            loss_weight=1.0,
            max_planes=3,
        ),
    )
    denoiser, critics = build_models(cfg)
    ema = build_ema(denoiser)
    denoiser_optimizer, critic_optimizers = build_optimizers(
        denoiser,
        critics,
        cfg,
    )
    labels = torch.randint(0, data.num_phases, (data.batch_size, 8, 8))
    trainer = _make_trainer(
        cfg,
        denoiser=denoiser,
        ema_denoiser=ema,
        critics=critics,
        streams={axis: _ConstantStream(labels) for axis in (0, 1, 2)},
        diffusion=Diffusion(2, beta_min=0.1, beta_max=2.0),
        denoiser_optimizer=denoiser_optimizer,
        critic_optimizers=critic_optimizers,
        device=torch.device("cpu"),
    )
    adapter_before = denoiser.anchor_input.weight.detach().clone()
    randint = torch.randint

    def sample_randint(*args, **kwargs):
        if args[:3] == (1, 4, ()):
            return torch.tensor(3, device=kwargs.get("device"))
        return randint(*args, **kwargs)

    with (
        patch(
            "src.train.engine.torch.randint",
            side_effect=sample_randint,
        ),
        patch(
            "src.train.engine.torch.randperm",
            return_value=torch.tensor([0, 1, 2]),
        ),
        patch.object(
            denoiser,
            "forward",
            wraps=denoiser.forward,
        ) as forward,
        patch.object(
            denoiser,
            "predict_logits",
            wraps=denoiser.predict_logits,
        ) as predict_logits,
    ):
        metrics = trainer.step(0, transition=0)

    assert metrics.anchor_planes == 3
    assert 0.0 <= metrics.anchor_conflict_rate <= 1.0
    assert forward.call_count == 1
    assert predict_logits.call_count == 2
    assert all(
        call.kwargs["anchor_image"] is not None
        and call.kwargs["anchor_mask"] is not None
        for call in predict_logits.call_args_list
    )
    assert math.isfinite(metrics.anchor_loss)
    assert 0.0 <= metrics.anchor_accuracy <= 1.0
    assert math.isclose(
        metrics.generator,
        metrics.generator_global + optim.critic_local_weight * metrics.generator_local,
        rel_tol=1e-5,
    )
    assert math.isclose(
        metrics.generator_total,
        metrics.generator + metrics.anchor_loss,
        rel_tol=1e-5,
    )
    assert not torch.equal(adapter_before, denoiser.anchor_input.weight.detach())


def test_interrupt_saves_all_weights_and_is_reraised(tmp_path: Path) -> None:
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.ema_denoiser = nn.Linear(2, 2)
    trainer.critics = nn.ModuleDict({str(axis): nn.Linear(2, 1) for axis in range(3)})
    trainer.step = Mock(side_effect=KeyboardInterrupt)

    with pytest.raises(KeyboardInterrupt):
        trainer.fit(
            steps=1,
            save_every=1,
            critic_warmup_steps=0,
            run_dir=tmp_path,
        )

    assert (tmp_path / "model.pt").is_file()
    assert tuple(path.name for path in sorted(tmp_path.glob("critic_*.pt"))) == (
        "critic_0.pt",
        "critic_1.pt",
        "critic_2.pt",
    )


def _parameters(model: nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in model.parameters())


def _changed(before: tuple[torch.Tensor, ...], model: nn.Module) -> bool:
    return any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.parameters(), strict=True)
    )


def _make_trainer(
    cfg: TrainConfig,
    *,
    denoiser,
    ema_denoiser,
    critics,
    streams,
    diffusion,
    denoiser_optimizer,
    critic_optimizers,
    device,
) -> Trainer:
    use_amp = cfg.train.mixed_precision and device.type == "cuda"
    return Trainer(
        denoiser=denoiser,
        ema_denoiser=ema_denoiser,
        critics=critics,
        streams=streams,
        diffusion=diffusion,
        denoiser_optimizer=denoiser_optimizer,
        critic_optimizers=critic_optimizers,
        scaler=torch.amp.GradScaler("cuda", enabled=use_amp),
        device=device,
        volume_batch_size=cfg.train.volume_batch_size,
        num_phases=cfg.data.num_phases,
        patch_size=cfg.data.patch_size,
        slices_per_axis=cfg.train.slices_per_axis,
        ema_decay=cfg.train.ema_decay,
        r1_gamma=cfg.optim.r1_gamma,
        r1_interval=cfg.optim.r1_interval,
        critic_local_weight=cfg.optim.critic_local_weight,
        anchor_probability=cfg.anchor.probability,
        anchor_loss_weight=cfg.anchor.loss_weight,
        anchor_max_planes=cfg.anchor.max_planes,
        latent_channels=cfg.model.latent_channels,
        amp_enabled=use_amp,
    )


def _config(
    data: DataConfig,
    model: ModelConfig,
    optim: OptimConfig,
    *,
    anchor: AnchorConfig | None = None,
) -> TrainConfig:
    return TrainConfig(
        data=data,
        model=model,
        diffusion=DiffusionConfig(
            timesteps=2,
            beta_min=0.1,
            beta_max=2.0,
        ),
        anchor=AnchorConfig() if anchor is None else anchor,
        optim=optim,
        train=LoopConfig(
            steps=1,
            volume_batch_size=1,
            slices_per_axis=2,
            mixed_precision=False,
            ema_decay=0.9,
            save_every_steps=1,
        ),
    )
