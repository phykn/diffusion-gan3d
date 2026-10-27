from __future__ import annotations

import copy
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from src.build.model import build_models
from src.build.trainer import build_optimizers
from src.data.augment import CriticAugment
from src.model.diffusion import Diffusion
from src.model.layers import NULL_DOMAIN
from src.plane import PLANE_DIRECTIONS, PLANES
from src.prepare.resize import phase_channels
from src.train.batch import RealBatch
from src.train.burn_in import BurnIn
from src.train.ema import build_ema
from src.train.loss.denoiser import DenoiserLossSettings
from src.train.loss.gan import get_gradient_penalty
from src.train.metrics import Metrics
from src.train.run.loop import run_train
from src.train.trainer import (
    Trainer,
    TrainerComponents,
    TrainerSettings,
)


class _ConstantStream:
    def __init__(self, images: torch.Tensor) -> None:
        self.images = images
        self.calls = 0

    def next(self) -> dict:
        self.calls += 1
        return {"image": self.images.clone()}


def test_training_step_uses_null_critics_for_borrowed_axes() -> None:
    cfg = _config(
        domains={0: {"xy": ["."]}, 1: {plane: ["."] for plane in PLANES}},
        data={"num_phases": 2},
    )
    phases = cfg["data"]["num_phases"]
    denoiser, critics = build_models(cfg)
    ema = build_ema(denoiser)
    denoiser_optim, critic_optims = build_optimizers(denoiser, critics, cfg)
    images = (
        torch.arange(64, dtype=torch.long)
        .remainder(phases)
        .reshape(1, 8, 8)
        .expand(cfg["train"]["real_batch_size"], -1, -1)
        .clone()
    )
    streams = {
        0: {0: _ConstantStream(phase_channels(images, phases))},
        1: {
            axis: _ConstantStream(phase_channels(images, phases)) for axis in (0, 1, 2)
        },
    }
    trainer = _make_trainer(
        cfg,
        denoiser=denoiser,
        ema_denoiser=ema,
        critics=critics,
        streams=streams,
        streams_by_domain=True,
        diffusion=Diffusion(2, beta_min=0.1, beta_max=2.0),
        denoiser_optim=denoiser_optim,
        critic_optims=critic_optims,
        device=torch.device("cpu"),
    )
    observed_domains = {axis: [] for axis in (0, 1, 2)}
    hooks = [
        critics[PLANES[axis]].register_forward_pre_hook(
            lambda _module, args, axis=axis: observed_domains[axis].append(
                args[-1].tolist()
            )
        )
        for axis in (0, 1, 2)
    ]
    try:
        with patch.object(trainer.conditions, "sample_target_domain", return_value=0):
            metrics = trainer.step(0, transition=1)
    finally:
        for hook in hooks:
            hook.remove()

    assert metrics.domain == 0
    assert streams[0][0].calls == 1
    assert streams[1][0].calls == 0
    assert streams[1][1].calls == 1
    assert streams[1][2].calls == 1
    assert all(set(values) == {0} for values in observed_domains[0])
    assert all(set(values) == {NULL_DOMAIN} for values in observed_domains[1])
    assert all(set(values) == {NULL_DOMAIN} for values in observed_domains[2])


@pytest.mark.parametrize("real_only", [False, True])
def test_training_step_updates_denoiser_and_all_critics(real_only) -> None:
    cfg = _config(loss={"r1_weight": 0.01, "r1_every_steps": 1})
    phases = cfg["data"]["num_phases"]
    denoiser, critics = build_models(cfg)
    ema = build_ema(denoiser)
    denoiser_optim, critic_optims = build_optimizers(denoiser, critics, cfg)
    images = torch.randint(0, phases, (cfg["train"]["real_batch_size"], 8, 8))
    streams = {
        axis: _ConstantStream(phase_channels(images, phases)) for axis in (0, 1, 2)
    }
    trainer = _make_trainer(
        cfg,
        denoiser=denoiser,
        ema_denoiser=ema,
        critics=critics,
        streams=streams,
        diffusion=Diffusion(2, beta_min=0.1, beta_max=2.0),
        denoiser_optim=denoiser_optim,
        critic_optims=critic_optims,
        device=torch.device("cpu"),
    )
    trainer.critic_augment = Mock(
        wraps=CriticAugment(
            planes={
                plane: {"flip_axes": list(PLANE_DIRECTIONS[plane]), "rotate_90": True}
                for plane in PLANES
            },
            prob=1.0,
            real_only=real_only,
        )
    )
    trainer.critic_augment.real_only = real_only  # Mock does not wrap attributes.
    denoiser_before = _parameters(denoiser)
    critic_before = {axis: _parameters(critics[PLANES[axis]]) for axis in (0, 1, 2)}

    r1_values = []

    def track_r1(scores, inputs):
        penalties = get_gradient_penalty(scores, inputs)
        r1_values.append(float(penalties.detach()))
        return penalties

    with patch("src.train.loss.gan.get_gradient_penalty", side_effect=track_r1):
        metrics = trainer.step(0)

    assert math.isfinite(metrics.generator)
    assert math.isfinite(metrics.critic)
    assert math.isfinite(metrics.r1)
    assert len(r1_values) == 3
    # Real slices of three planes, plus fake slices unless reals only.
    assert trainer.critic_augment.apply_together.call_count == (3 if real_only else 6)
    assert math.isclose(metrics.r1, sum(r1_values) / len(r1_values), rel_tol=1e-6)
    assert 0 <= metrics.transition < 2
    assert _changed(denoiser_before, denoiser)
    assert all(
        _changed(critic_before[axis], critics[PLANES[axis]]) for axis in (0, 1, 2)
    )
    assert all(not parameter.requires_grad for parameter in ema.parameters())


@pytest.mark.parametrize("axis", (0, 2))
def test_training_step_with_one_axis_updates_only_that_critic(axis: int) -> None:
    trainer, denoiser, streams = _conditioning_trainer(axes=(axis,))
    denoiser_before = _parameters(denoiser)
    critic_before = _parameters(trainer.critics[PLANES[axis]])

    metrics = trainer.step(0, transition=1)

    assert trainer.active_axes == (axis,)
    assert set(trainer.critics) == {PLANES[axis]}
    assert set(trainer.critic_optims) == {PLANES[axis]}
    assert streams[axis].calls == 1
    assert metrics.critic_axes[axis] != 0.0
    assert all(
        value == 0.0 for index, value in enumerate(metrics.critic_axes) if index != axis
    )
    assert _changed(denoiser_before, denoiser)
    assert _changed(critic_before, trainer.critics[PLANES[axis]])


def test_step_reuses_each_real_batch_and_conditions_every_reverse_step() -> None:
    trainer, denoiser, streams = _conditioning_trainer()
    vf_batch_ids = []
    critic_batch_ids = []
    original_sample_vf = trainer.conditions.sample_vf
    original_sample_real_pairs = trainer._sample_real_pairs
    targets = []

    def track_vfs(batches, *args):
        vf_batch_ids.extend(id(batches.images[axis]) for axis in (0, 1, 2))
        targets.append(original_sample_vf(batches, *args))
        return targets[-1]

    def track_critics(transition, batches, *args):
        critic_batch_ids.extend(id(batches.images[axis]) for axis in (0, 1, 2))
        return original_sample_real_pairs(transition, batches, *args)

    with (
        patch.object(trainer.conditions, "sample_vf", side_effect=track_vfs),
        patch.object(trainer, "_sample_real_pairs", side_effect=track_critics),
        patch.object(denoiser, "forward", wraps=denoiser.forward) as forward,
        patch.object(
            denoiser,
            "compute_logits",
            wraps=denoiser.compute_logits,
        ) as compute_logits,
    ):
        metrics = trainer.step(0, transition=0)

    assert all(stream.calls == 1 for stream in streams.values())
    assert metrics.vf_active
    assert metrics.vf_loss > 0.0

    calls = [*forward.call_args_list, *compute_logits.call_args_list]
    vfs = [call.kwargs["vf"] for call in calls]
    assert len(forward.call_args_list) == 2
    assert len(compute_logits.call_args_list) == 2
    assert all(vf is targets[0] for vf in vfs)
    assert vfs[0].shape == (1, 3)
    assert vf_batch_ids == critic_batch_ids

    gradient = denoiser.vf_mlp[-1].weight.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().sum()) > 0.0


def test_vf_conditions_follow_individual_fractional_crops():
    trainer, _, streams = _conditioning_trainer()
    trainer.settings = replace(trainer.settings, volume_batch_size=2)
    trainer.conditions.settings = trainer.settings
    fractions = torch.tensor([[0.8, 0.2, 0.0], [0.1, 0.3, 0.6]])
    for stream in streams.values():
        stream.images = fractions[:, :, None, None].expand(-1, -1, 4, 8)

    prepared = trainer.prepare_step(0, 0)

    expected = fractions[fractions[:, 0].argsort()]
    observed = prepared.target_vf[prepared.target_vf[:, 0].argsort()]
    torch.testing.assert_close(observed, expected)
    assert prepared.model_conditions["vf"] is prepared.target_vf


def test_vf_supports_more_volumes_than_real_crops():
    trainer, _, streams = _conditioning_trainer()
    trainer.settings = replace(trainer.settings, volume_batch_size=5)
    trainer.conditions.settings = trainer.settings
    fractions = torch.tensor([[0.8, 0.2, 0.0], [0.1, 0.3, 0.6]])
    for stream in streams.values():
        stream.images = fractions[:, :, None, None].expand(-1, -1, 8, 8)

    prepared = trainer.prepare_step(0, 0)

    assert prepared.target_vf.shape == (5, 3)
    for row in prepared.target_vf:
        assert any(torch.allclose(row, crop) for crop in fractions)
    torch.testing.assert_close(prepared.target_vf[:2], fractions)


def test_vf_and_height_use_the_same_owned_crops():
    trainer, _, _ = _conditioning_trainer()
    trainer.settings = replace(
        trainer.settings,
        volume_batch_size=2,
        height_data={"crop_size": 8, "height_extents": {0: 32}},
    )
    trainer.conditions.settings = trainer.settings
    fractions = torch.tensor([[0.8, 0.2, 0.0], [0.1, 0.3, 0.6]])
    batches = RealBatch(
        images={
            0: torch.full((2, 3, 8, 8), 1 / 3),
            1: fractions[:, :, None, None].expand(-1, -1, 8, 8),
            2: torch.full((2, 3, 8, 8), 1 / 3),
        },
        origins={1: torch.tensor([0.0, 8.0])},
        extents={1: torch.tensor([32.0, 32.0])},
    )
    with patch.object(trainer.conditions, "get_batches", return_value=batches):
        prepared = trainer.prepare_step(0, 0)

    torch.testing.assert_close(prepared.target_vf, fractions)
    torch.testing.assert_close(
        prepared.model_conditions["height"],
        trainer.conditions.volume_height(batches.origins[1], 0, batches.extents[1]),
    )


def test_profile_keeps_priority_over_crop_vf():
    trainer, _, _ = _conditioning_trainer()
    # Profiles require height conditioning, so the side crop carries both.
    trainer.settings = replace(
        trainer.settings,
        profile_settings={"enabled": True},
        height_data={"crop_size": 8, "height_extents": {0: None}},
    )
    trainer.conditions.settings = trainer.settings
    profile = torch.tensor([[[0.2, 0.4], [0.3, 0.2], [0.5, 0.4]]])
    batches = RealBatch(
        images={axis: torch.full((2, 3, 8, 8), 1 / 3) for axis in (0, 1, 2)},
        origins={1: torch.zeros(2)},
        extents={1: torch.full((2,), 8.0)},
        profiles={1: profile},
    )
    with patch.object(trainer.conditions, "get_batches", return_value=batches):
        prepared = trainer.prepare_step(0, 0)
    torch.testing.assert_close(prepared.target_vf, profile.mean(-1))
    assert prepared.model_conditions["vf"] is prepared.target_vf


def test_generate_pair_keeps_initial_noise_and_returns_the_final_prediction() -> None:
    trainer, _, _ = _conditioning_trainer()
    conditions = trainer.prepare_step(0, 0).model_conditions
    trainer.diffusion = Diffusion(1)
    initial = torch.linspace(-0.9, 0.9, 3 * 8**3).reshape(1, 3, 8, 8, 8)
    latent = torch.zeros(1, trainer.settings.latent_channels)
    with (
        patch("src.train.trainer.torch.randn", return_value=initial.clone()),
        patch.object(trainer, "sample_latent", return_value=latent),
    ):
        previous, current, prediction = trainer.generate_pair(
            transition=0, model_conditions=conditions
        )

    assert torch.equal(current, initial)
    assert torch.equal(previous, prediction)


def test_training_volume_and_critic_use_patch_size() -> None:
    trainer, denoiser, streams = _conditioning_trainer(
        data={"crop_size": 8, "lo_res_size": 8},
    )
    shapes = []
    hooks = [
        critic.register_forward_pre_hook(
            lambda _module, args: shapes.append(tuple(args[0].shape))
        )
        for critic in trainer.critics.values()
    ]
    try:
        trainer.step(0, transition=0)
    finally:
        for hook in hooks:
            hook.remove()

    assert all(stream.calls == 1 for stream in streams.values())
    assert shapes
    assert all(shape[-2:] == (8, 8) for shape in shapes)


def test_training_critics_match_each_axis_rectangular_real_shape() -> None:
    trainer, _, streams = _conditioning_trainer()
    shapes = {
        0: (8, 8),
        1: (4, 8),
        2: (8, 6),
    }
    trainer.settings = replace(trainer.settings, r1_weight=0.01, r1_every_steps=1)
    trainer.conditions.settings = trainer.settings
    for axis, shape in shapes.items():
        streams[axis].images = phase_channels(torch.randint(0, 3, (2, *shape)), 3)

    observed = {axis: [] for axis in (0, 1, 2)}
    hooks = [
        trainer.critics[PLANES[axis]].register_forward_pre_hook(
            lambda _module, args, axis=axis: observed[axis].append(
                tuple(args[0].shape[-2:])
            )
        )
        for axis in (0, 1, 2)
    ]
    try:
        metrics = trainer.step(0, transition=0)
    finally:
        for hook in hooks:
            hook.remove()

    assert math.isfinite(metrics.r1)
    assert all(observed[axis] for axis in (0, 1, 2))
    assert all(
        all(actual == shapes[axis] for actual in observed[axis]) for axis in (0, 1, 2)
    )


def test_single_vf_condition_can_be_dropped_for_the_whole_batch() -> None:
    trainer, denoiser, streams = _conditioning_trainer(
        conditioning={"cfg_dropout_probability": 0.1},
    )
    hidden = torch.zeros(1, dtype=torch.bool)

    with (
        patch.object(trainer.conditions, "sample_vf_presence", return_value=hidden),
        patch.object(denoiser, "forward", wraps=denoiser.forward) as forward,
        patch.object(
            denoiser,
            "compute_logits",
            wraps=denoiser.compute_logits,
        ) as compute_logits,
    ):
        metrics = trainer.step(0, transition=0)

    assert all(stream.calls == 1 for stream in streams.values())
    assert not metrics.vf_active
    assert metrics.vf_loss == 0.0
    calls = [*forward.call_args_list, *compute_logits.call_args_list]
    assert len(forward.call_args_list) == 2
    assert len(compute_logits.call_args_list) == 2
    assert all(call.kwargs["vf"] is not None for call in calls)
    assert all(not bool(call.kwargs["vf_present"].any()) for call in calls)
    assert all(
        parameter.grad is None or not bool(parameter.grad.any())
        for parameter in denoiser.vf_mlp.parameters()
    )


def test_exception_inside_step_does_not_publish_partial_weights(tmp_path: Path) -> None:
    trainer = object.__new__(Trainer)
    trainer.settings = SimpleNamespace(loss=SimpleNamespace())
    trainer.settings.profile_settings = {"enabled": False}
    trainer.device = torch.device("cpu")
    trainer.cfg = {"stage": "low_res", "data": {}}
    trainer.streams = {}
    trainer.data_fingerprint = {}
    trainer.selection = None
    trainer.ema_denoiser = nn.Linear(2, 2)
    trainer.critics = nn.ModuleDict(
        {PLANES[axis]: nn.Linear(2, 1) for axis in range(3)}
    )
    trainer.settings.loss.real_transition_weight = 0.0
    trainer.step = Mock(side_effect=KeyboardInterrupt)
    checkpoint = tmp_path / "checkpoints" / "last.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"previous complete training state")

    with pytest.raises(KeyboardInterrupt):
        run_train(
            trainer,
            steps=1,
            save_every=1,
            run_dir=tmp_path,
        )

    assert not (tmp_path / "generator.pt").exists()
    assert not list(tmp_path.glob("critic_*.pt"))
    assert checkpoint.read_bytes() == b"previous complete training state"


@pytest.mark.parametrize(
    "save_every,checkpoint_every,expected_weights,expected_checkpoints",
    [
        (1, 2, [1, 2, 3], [2, 3]),
        (2, 1, [2, 3], [1, 2, 3]),
        (1, None, [1, 2, 3], [3]),
        (10, 10, [3], [3]),
    ],
)
def test_fit_separates_weight_updates_and_checkpoint_archives(
    tmp_path: Path,
    monkeypatch,
    save_every,
    checkpoint_every,
    expected_weights,
    expected_checkpoints,
) -> None:
    monkeypatch.setattr(
        "src.train.run.loop.save_training",
        lambda trainer, path: torch.save({"step": trainer.completed_steps}, path),
    )
    trainer = object.__new__(Trainer)
    trainer.settings = SimpleNamespace(loss=SimpleNamespace())
    trainer.settings.profile_settings = {"enabled": False}
    trainer.device = torch.device("cpu")
    trainer.cfg = {"stage": "low_res", "data": {}}
    trainer.streams = {}
    trainer.data_fingerprint = {}
    trainer.selection = None
    trainer.ema_denoiser = nn.Linear(2, 2)
    trainer.critics = nn.ModuleDict(
        {PLANES[axis]: nn.Linear(2, 1) for axis in range(3)}
    )
    trainer.settings.loss.real_transition_weight = 0.0
    trainer.completed_steps = 0

    def step(index):
        trainer.completed_steps = index + 1
        return Metrics(
            generator=1.0,
            generator_total=1.0,
            critic=1.0,
            r1=0.0,
            transition=0,
            domain=0,
            critic_axes=(1.0, 1.0, 1.0),
        )

    trainer.step = step
    from src.train.run.loop import save_weights

    weight_steps = []

    def record_weights(trainer, root):
        weight_steps.append(trainer.completed_steps)
        save_weights(trainer, root)

    monkeypatch.setattr("src.train.run.loop.save_weights", record_weights)

    weights = run_train(
        trainer,
        steps=3,
        save_every=save_every,
        checkpoint_every=checkpoint_every,
        run_dir=tmp_path,
    )

    assert weights == tmp_path / "generator.pt"
    assert weights.is_file()
    checkpoints = tuple(p for p in (tmp_path / "checkpoints").iterdir() if p.is_dir())
    assert not checkpoints
    assert weight_steps == expected_weights
    states = sorted((tmp_path / "checkpoints").glob("step_*.pt"))
    assert [
        torch.load(p, weights_only=True)["step"] for p in states
    ] == expected_checkpoints
    assert all(p.with_suffix(".complete").is_file() for p in states)


def _parameters(model: nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in model.parameters())


def _changed(before: tuple[torch.Tensor, ...], model: nn.Module) -> bool:
    return any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.parameters(), strict=True)
    )


def _conditioning_trainer(
    *,
    axes: tuple[int, ...] = (0, 1, 2),
    **overrides,
) -> tuple[Trainer, nn.Module, dict[int, _ConstantStream]]:
    cfg = _config(domains={0: {PLANES[axis]: ["."] for axis in axes}})
    cfg["model"]["critic"]["input_mode"] = "single"
    _merge(cfg, overrides)
    denoiser, critics = build_models(cfg)
    ema = build_ema(denoiser)
    denoiser_optim, critic_optims = build_optimizers(denoiser, critics, cfg)
    size = cfg["data"]["crop_size"]
    base = torch.arange(2 * size * size).reshape(2, size, size)
    streams = {
        axis: _ConstantStream(phase_channels((base + axis).remainder(3).long(), 3))
        for axis in axes
    }
    trainer = _make_trainer(
        cfg,
        denoiser=denoiser,
        ema_denoiser=ema,
        critics=critics,
        streams=streams,
        diffusion=Diffusion(2, beta_min=0.1, beta_max=2.0),
        denoiser_optim=denoiser_optim,
        critic_optims=critic_optims,
        device=torch.device("cpu"),
    )
    return trainer, denoiser, streams


def _make_trainer(
    cfg: dict,
    *,
    denoiser,
    ema_denoiser,
    critics,
    streams,
    streams_by_domain=False,
    diffusion,
    denoiser_optim,
    critic_optims,
    device,
) -> Trainer:
    data, train, loss = cfg["data"], cfg["train"], cfg["loss"]
    conditioning = cfg["conditioning"]
    use_amp = train["mixed_precision"] and device.type == "cuda"
    return Trainer(
        components=TrainerComponents(
            denoiser=denoiser,
            ema_denoiser=ema_denoiser,
            critics=critics,
            streams=streams if streams_by_domain else {0: streams},
            diffusion=diffusion,
            denoiser_optim=denoiser_optim,
            critic_optims=critic_optims,
            scaler=torch.amp.GradScaler(device.type, enabled=use_amp),
            device=device,
        ),
        settings=TrainerSettings(
            volume_batch_size=train["volume_batch_size"],
            num_phases=data["num_phases"],
            patch_size=data["lo_res_size"],
            slice_pairs_per_plane=train["slice_pairs_per_plane"],
            # A constant schedule: the configured values from the first step.
            burn_in=BurnIn(
                steps=0,
                generator_lr=cfg["optim"]["generator_lr"],
                critic_lr=cfg["optim"]["critic_lr"],
                betas=tuple(cfg["optim"]["adam_betas"]),
                ema_decay=cfg["optim"]["ema_decay"],
                lr_scale=1.0,
                beta2=cfg["optim"]["adam_betas"][1],
                regularization_scale=1.0,
            ),
            r1_weight=loss["r1_weight"],
            r1_every_steps=loss["r1_every_steps"],
            loss=DenoiserLossSettings(
                vf_weight=loss["volume_fraction_weight"],
                real_transition_weight=0.0,
                profile_bins=16,
                profile_weight=0.0,
                profile_gradient_weight=0.0,
                consistency_weight=0.0,
                consistency_tolerance=0.0,
                num_phases=data["num_phases"],
                group_statistics_weight=0.0,
                group_statistics_max_gap=8,
                group_statistics_tolerance=0.01,
                preserve_height=False,
                mirror_balance_weight=0.0,
                mirror_balance_max_gap=4,
                mirror_balance_tolerance=0.02,
            ),
            domain_keep_probability=conditioning["domain_keep_probability"],
            cfg_dropout_probability=conditioning["cfg_dropout_probability"],
            latent_channels=cfg["model"]["generator"]["latent_channels"],
            denoiser_amp=use_amp,
            critic_amp=use_amp,
            r2_weight=0.0,
            group_statistics_start_step=0,
            group_statistics_ramp_steps=5000,
            real_transition_max_gap=1,
            real_transition_start_step=0,
            real_transition_ramp_steps=0,
            structure_every_steps=100,
            coarse_corruption_probability=0.0,
            coarse_corruption_strength=0.0,
            height_data=None,
            profile_settings={
                "enabled": False,
                "num_bins": 16,
                "critic_enabled": False,
            },
            volume_permute_z=True,
            volume_flip_z=True,
        ),
    )


def _config(domains=None, **overrides) -> dict:
    if domains is None:
        domains = {0: {plane: ["."] for plane in PLANES}}
    cfg = {
        "data": {
            "domains": domains,
            "num_phases": 3,
            "crop_size": 8,
            "lo_res_size": 8,
        },
        "model": {
            "gradient_checkpointing": False,
            "generator": {
                "channels": [4, 8],
                "embedding_channels": 8,
                "latent_channels": 4,
            },
            "critic": {
                "channels": (4, 8),
                "plane_groups": [
                    [plane]
                    for plane in PLANES
                    if any(plane in planes for planes in domains.values())
                ],
            },
            "diffusion": {"num_steps": 2, "beta_min": 0.1, "beta_max": 2.0},
        },
        "conditioning": {
            "domain_keep_probability": 1.0,
            "cfg_dropout_probability": 0.0,
        },
        "loss": {
            "r1_weight": 0.0,
            "r1_every_steps": 2,
            "volume_fraction_weight": 1.0,
        },
        "optim": {
            "generator_lr": 1e-3,
            "critic_lr": 1e-3,
            "adam_betas": [0.0, 0.9],
            "ema_decay": 0.9,
        },
        "train": {
            "initial_weights": None,
            "total_steps": 1,
            "volume_batch_size": 1,
            "slice_pairs_per_plane": 2,
            "mixed_precision": False,
            "weights_every_steps": 1,
            "archive_every_steps": 1,
            "real_batch_size": 2,
            "num_workers": 0,
        },
    }
    _merge(cfg, overrides)
    return cfg


def _merge(cfg: dict, overrides: dict) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            _merge(cfg[key], value)
        else:
            cfg[key] = value


@pytest.mark.parametrize("mode", ["single", "pair"])
def test_generator_input_diagnostics_preserve_training_updates_and_rng(mode):
    trainer, _, _ = _conditioning_trainer(
        axes=(0,), model={"critic": {"input_mode": mode}}
    )
    trainer.settings = replace(trainer.settings, r1_every_steps=1)
    trainer.conditions.settings = trainer.settings
    control = copy.deepcopy(trainer)
    control.settings = replace(
        control.settings, r1_every_steps=3
    )  # R1/R2 weights are zero; only diagnostics differ.
    control.conditions.settings = control.settings
    with torch.random.fork_rng(devices=[]):
        rng = torch.get_rng_state()
        actual = trainer.step(0, transition=0)
        after = torch.get_rng_state()
        torch.set_rng_state(rng)
        expected = control.step(0, transition=0)
        assert torch.equal(after, torch.get_rng_state())
    prefix = "generator_input_gradient/xy/0/t0/"
    assert actual.diagnostics[prefix + "previous"] > 0
    if mode == "pair":
        assert actual.diagnostics[prefix + "current"] > 0
    else:
        assert actual.diagnostics[prefix + "current"] == 0
    assert actual.generator_total == expected.generator_total
    assert not any(
        key.startswith("generator_input_gradient/") for key in expected.diagnostics
    )
    for model, baseline in (
        (trainer.denoiser, control.denoiser),
        (trainer.critics, control.critics),
    ):
        torch.testing.assert_close(
            model.state_dict(), baseline.state_dict(), rtol=0, atol=0
        )
    trainer.settings = replace(trainer.settings, r1_every_steps=3)
    trainer.conditions.settings = trainer.settings
    metrics = trainer.step(1, transition=1)
    assert not any(
        key.startswith("generator_input_gradient/") for key in metrics.diagnostics
    )
