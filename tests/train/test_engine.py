from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from src.anchor import PlaneAnchor, build_anchors
from src.build import build_models, build_optimizers
from src.diffusion import Diffusion
from src.model.domain import NULL_DOMAIN
from src.train import vf
from src.train.augment import CriticAugment
from src.train.connect import TripletBatch
from src.train.ema import build_ema
from src.train.engine import (
    ConditionPresence,
    Metrics,
    Trainer,
    TrainerComponents,
    TrainerSettings,
)
from src.train.loss import get_critic_r1
from src.train.runner import run_training


class Config(dict):
    def __getattr__(self, name):
        return self[name]


DataConfig = Config
ModelConfig = Config
DiffusionConfig = Config
OptimConfig = Config
LoopConfig = Config
TrainConfig = Config


def AnchorConfig(**values):
    cfg = Config(
        multiscale_input=False,
        train_prob=0.0,
        start_step=0,
        ramp_steps=0,
        cross_domain_prob=0.0,
        pixel_weight=0.05,
    )
    cfg.update(values)
    return cfg


def VfConfig(**values):
    return Config(**values)


def ConnectivityConfig(**values):
    cfg = Config(
        weight=0.0,
        phase_transition_weight=0.0,
        volume_count=1,
        refresh_every=500,
    )
    cfg.update(values)
    return cfg


def ConditioningConfig(**values):
    cfg = Config(joint_each_prob=0.0)
    cfg.update(values)
    return cfg


def sample_pairs(
    previous,
    current,
    axis,
    count,
    patch_size,
    axis_masks=None,
    crop_shape=None,
):
    trainer = object.__new__(Trainer)
    trainer.slice_pairs_per_axis = count
    trainer.patch_size = patch_size
    return trainer.sample_pairs(previous, current, axis, axis_masks, crop_shape)


def test_connectivity_augmentation_preserves_triplet_center_slots() -> None:
    trainer = object.__new__(Trainer)
    trainer.num_phases = 2
    trainer.patch_size = 1
    trainer.device = torch.device("cpu")
    trainer.connectivity_weight = 1.0
    trainer.normal_transition_weight = 1.0
    real_centers = torch.tensor((1, 1))
    fake_centers = torch.tensor((0, 2))
    axes = torch.tensor((0, 0))
    real = TripletBatch(
        values=torch.zeros(2, 3, 2, 1, 1),
        axes=axes,
        center_slots=real_centers,
        anchor_flags=torch.tensor((True, False)),
    )
    fake = TripletBatch(
        values=torch.ones(2, 3, 2, 1, 1),
        axes=axes,
        center_slots=fake_centers,
        anchor_flags=torch.tensor((True, False)),
    )
    trainer.connect = Mock()
    trainer.connect.match_anchor.return_value = (real, fake)
    trainer.critic_augment = Mock()
    trainer.critic_augment.apply_together.return_value = (
        real.values + 2.0,
        fake.values + 3.0,
    )

    augmented_real, augmented_fake = trainer.make_connectivity_triplets(
        torch.zeros(1, 2, 3, 3, 3),
        Mock(),
        transition=0,
        domain=0,
    )

    assert augmented_real.center_slots.tolist() == [1, 1]
    assert augmented_fake.center_slots.tolist() == [0, 2]
    assert torch.equal(augmented_real.values, real.values + 2.0)
    assert torch.equal(augmented_fake.values, fake.values + 3.0)


def test_anchor_transitions_prioritize_the_final_step() -> None:
    trainer = object.__new__(Trainer)
    trainer.diffusion = Diffusion(11)

    with (
        patch("src.train.engine.torch.rand", return_value=torch.tensor(0.1)),
        patch("src.train.engine.torch.randint") as randint,
    ):
        assert trainer.sample_transition(anchored=True) == 0
        randint.assert_not_called()

    with (
        patch("src.train.engine.torch.rand", return_value=torch.tensor(0.5)),
        patch(
            "src.train.engine.torch.randint", return_value=torch.tensor(7)
        ) as randint,
    ):
        assert trainer.sample_transition(anchored=True) == 7
        randint.assert_called_once_with(1, 11, ())

    with patch(
        "src.train.engine.torch.randint",
        return_value=torch.tensor(6),
    ) as randint:
        assert trainer.sample_transition(anchored=False) == 6
        randint.assert_called_once_with(11, ())


def test_sample_pairs_centers_half_the_patches_on_focus() -> None:
    previous = torch.zeros(1, 1, 8, 8, 8)
    previous[0, 0, 2, 5, 6] = 1.0
    current = previous.clone()
    axis_masks = torch.zeros(1, 3, 8, 8, 8, dtype=torch.bool)
    axis_masks[0, 1, 2, 5, 6] = True

    selected, _ = sample_pairs(
        previous,
        current,
        axis=0,
        count=4,
        patch_size=4,
        axis_masks=axis_masks,
    )

    assert torch.equal(selected[:2].amax(dim=(1, 2, 3)), torch.ones(2))


def test_sample_pairs_centers_rectangular_patches_on_focus() -> None:
    previous = torch.zeros(1, 1, 8, 8, 8)
    previous[0, 0, 2, 5, 6] = 1.0
    current = previous.clone()
    axis_masks = torch.zeros(1, 3, 8, 8, 8, dtype=torch.bool)
    axis_masks[0, 1, 2, 5, 6] = True

    selected, _ = sample_pairs(
        previous,
        current,
        axis=0,
        count=4,
        patch_size=8,
        axis_masks=axis_masks,
        crop_shape=(2, 4),
    )

    assert selected.shape[-2:] == (2, 4)
    assert torch.equal(selected[:2].amax(dim=(1, 2, 3)), torch.ones(2))


def test_sample_pairs_ignores_anchors_parallel_to_the_critic() -> None:
    previous = torch.zeros(1, 1, 8, 8, 8)
    previous[0, 0, 2, 5, 6] = 2.0
    previous[0, 0, 7, 1, 1] = 1.0
    current = previous.clone()
    axis_masks = torch.zeros(1, 3, 8, 8, 8, dtype=torch.bool)
    axis_masks[0, 1, 2, 5, 6] = True
    axis_masks[0, 0, 7, 1, 1] = True

    selected, _ = sample_pairs(
        previous,
        current,
        axis=0,
        count=2,
        patch_size=1,
        axis_masks=axis_masks,
    )

    assert float(selected[0, 0, 0, 0]) == 2.0


class _ConstantStream:
    def __init__(self, images: torch.Tensor) -> None:
        self.images = images
        self.calls = 0

    def next(self) -> torch.Tensor:
        self.calls += 1
        return self.images.clone()


def test_get_batches_uses_one_domain_for_all_axes() -> None:
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    streams = {
        domain: {
            axis: _ConstantStream(torch.full((1, 2, 2), domain)) for axis in (0, 1, 2)
        }
        for domain in (0, 1)
    }
    trainer.streams = streams
    trainer.active_axes = (0, 1, 2)
    trainer.axis_domains = {axis: (0, 1) for axis in (0, 1, 2)}

    batches = trainer.get_batches(1)

    assert all(bool((batch == 1).all()) for batch in batches.values())
    assert all(stream.calls == 0 for stream in streams[0].values())
    assert all(stream.calls == 1 for stream in streams[1].values())


def test_missing_axes_borrow_from_axis_providers() -> None:
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.streams = {
        0: {0: _ConstantStream(torch.full((1, 2, 2), 10))},
        1: {
            0: _ConstantStream(torch.full((1, 2, 2), 20)),
            1: _ConstantStream(torch.full((1, 2, 2), 21)),
            2: _ConstantStream(torch.full((1, 2, 2), 22)),
        },
    }
    trainer.axis_domains = {0: (0, 1), 1: (1,), 2: (1,)}
    trainer.active_axes = (0, 1, 2)

    sources = trainer.select_batch_domains(0)
    batches = trainer.get_batches(0, sources)
    critic_domains = trainer.make_critic_domains(0, 0, sources)

    assert sources == {0: 0, 1: 1, 2: 1}
    assert [int(batches[axis][0, 0, 0]) for axis in (0, 1, 2)] == [10, 21, 22]
    assert critic_domains == {0: 0, 1: NULL_DOMAIN, 2: NULL_DOMAIN}


def test_connectivity_uses_axis_critic_domain_for_shared_context() -> None:
    triplets = TripletBatch(
        values=torch.zeros(3, 3, 2, 4, 4),
        axes=torch.tensor((0, 1, 2)),
        center_slots=torch.ones(3, dtype=torch.long),
        anchor_flags=torch.tensor((True, False, False)),
    )

    domains = Trainer.get_connectivity_domains(
        critic_domains={0: 0, 1: NULL_DOMAIN, 2: NULL_DOMAIN},
        triplets=triplets,
    )

    assert domains.tolist() == [0, NULL_DOMAIN, NULL_DOMAIN]


def test_domain_dropout_masks_every_axis_critic() -> None:
    sources = {0: 0, 1: 1, 2: 1}

    critic_domains = Trainer.make_critic_domains(0, NULL_DOMAIN, sources)

    assert critic_domains == {axis: NULL_DOMAIN for axis in (0, 1, 2)}


def test_domain_dropout_probability_controls_the_model_condition() -> None:
    trainer = object.__new__(Trainer)
    trainer.domain_dropout = 0.0
    assert trainer.sample_domain_condition(2) == 2

    trainer.domain_dropout = 1.0
    assert trainer.sample_domain_condition(2) == NULL_DOMAIN


def test_initial_prior_build_selects_each_incomplete_domain() -> None:
    trainer = object.__new__(Trainer)
    trainer.prior_required = True
    trainer.anchor_start_step = 4
    trainer.anchor_ramp_steps = 0
    trainer.num_domains = 3
    trainer.connect = Mock(prior_ready=False)
    trainer.connect.needs_prior.side_effect = lambda domain: domain in (1, 2)
    trainer.sample_target_domain = Mock(return_value=2)

    assert trainer.select_target_domain(3) == 2
    trainer.sample_target_domain.assert_called_once_with()

    trainer.sample_target_domain.reset_mock()
    assert trainer.select_target_domain(4) == 1
    trainer.sample_target_domain.assert_not_called()
    assert trainer.connect.needs_prior.call_args_list[-2:] == [
        ((0,),),
        ((1,),),
    ]

    trainer.connect.prior_ready = True
    assert trainer.select_target_domain(5) == 2
    trainer.sample_target_domain.assert_called_once_with()


def test_ready_prior_starts_alternation_and_skipped_request_does_not_toggle() -> None:
    trainer = object.__new__(Trainer)
    trainer.anchor_training_probability = 0.5
    trainer.use_prior_next = True
    trainer.volume_batch_size = 1
    trainer.device = torch.device("cpu")
    mask = torch.ones(1, 1, 2, 2, 2, dtype=torch.bool)
    axis_masks = torch.ones(1, 3, 2, 2, 2, dtype=torch.bool)
    trainer.connect = Mock(prior_ready=True)
    trainer.connect.sample_prior_condition.return_value = Mock(
        condition="prior",
        observed_mask=mask,
        observed_axis_masks=axis_masks,
        references=(),
    )
    trainer.sample_real_anchor = Mock(return_value=Mock(source="real"))

    with patch("src.train.engine.torch.rand", return_value=torch.tensor(0.9)):
        assert trainer.sample_anchor({}, 2, domain=0, owned_axes=()) is None
    assert trainer.use_prior_next

    trainer.anchor_training_probability = 1.0
    sources = [
        trainer.sample_anchor({}, 2, domain=0, owned_axes=()).source for _ in range(3)
    ]

    assert sources == ["prior", "real", "prior"]


def test_training_step_uses_null_critics_for_borrowed_axes() -> None:
    data = DataConfig(
        domains={0: {0: "."}, 1: {0: ".", 1: ".", 2: "."}},
        crop_size=8,
        input_size=8,
        num_phases=2,
        batch_size=2,
        domain_dropout=0.0,
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
        denoiser_lr=1e-3,
        critic_lr=1e-3,
        beta1=0.0,
        beta2=0.9,
        r1_gamma=0.0,
        r1_interval=2,
        local_loss_weight=0.5,
    )
    cfg = _config(
        data,
        model,
        optim,
        anchor=AnchorConfig(
            train_prob=1.0,
            cross_domain_prob=1.0,
        ),
    )
    denoiser, critics, connectivity_critic = build_models(cfg)
    ema = build_ema(denoiser)
    denoiser_optim, critic_optims, connectivity_optim = build_optimizers(
        denoiser,
        critics,
        connectivity_critic,
        cfg,
    )
    images = (
        torch.arange(64, dtype=torch.long)
        .remainder(data.num_phases)
        .reshape(1, 8, 8)
        .expand(data.batch_size, -1, -1)
        .clone()
    )
    streams = {
        0: {0: _ConstantStream(images)},
        1: {axis: _ConstantStream(images) for axis in (0, 1, 2)},
    }
    trainer = _make_trainer(
        cfg,
        denoiser=denoiser,
        ema_denoiser=ema,
        critics=critics,
        connectivity_critic=connectivity_critic,
        streams=streams,
        streams_by_domain=True,
        diffusion=Diffusion(2, beta_min=0.1, beta_max=2.0),
        denoiser_optim=denoiser_optim,
        critic_optims=critic_optims,
        connectivity_optim=connectivity_optim,
        device=torch.device("cpu"),
    )
    observed_domains = {axis: [] for axis in (0, 1, 2)}
    hooks = [
        critics[str(axis)].register_forward_pre_hook(
            lambda _module, args, axis=axis: observed_domains[axis].append(
                args[3].tolist()
            )
        )
        for axis in (0, 1, 2)
    ]
    try:
        with patch.object(trainer, "sample_target_domain", return_value=0):
            metrics = trainer.step(0, transition=1)
    finally:
        for hook in hooks:
            hook.remove()

    assert metrics.domain == 0
    assert metrics.anchor_shared
    assert metrics.target_vfs == pytest.approx((0.5, 0.5))
    assert streams[0][0].calls == 1
    assert streams[1][0].calls == 0
    assert streams[1][1].calls == 1
    assert streams[1][2].calls == 1
    assert all(set(values) == {0} for values in observed_domains[0])
    assert all(set(values) == {NULL_DOMAIN} for values in observed_domains[1])
    assert all(set(values) == {NULL_DOMAIN} for values in observed_domains[2])


def test_training_step_updates_denoiser_and_all_critics() -> None:
    data = DataConfig(
        domains={0: {0: ".", 1: ".", 2: "."}},
        crop_size=8,
        input_size=8,
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
        denoiser_lr=1e-3,
        critic_lr=1e-3,
        beta1=0.0,
        beta2=0.9,
        r1_gamma=0.01,
        r1_interval=1,
        local_loss_weight=0.5,
    )
    cfg = _config(data, model, optim)
    denoiser, critics, connectivity_critic = build_models(cfg)
    ema = build_ema(denoiser)
    denoiser_optim, critic_optims, connectivity_optim = build_optimizers(
        denoiser,
        critics,
        connectivity_critic,
        cfg,
    )
    images = torch.randint(0, data.num_phases, (data.batch_size, 8, 8))
    streams = {axis: _ConstantStream(images) for axis in (0, 1, 2)}
    trainer = _make_trainer(
        cfg,
        denoiser=denoiser,
        ema_denoiser=ema,
        critics=critics,
        connectivity_critic=connectivity_critic,
        streams=streams,
        diffusion=Diffusion(2, beta_min=0.1, beta_max=2.0),
        denoiser_optim=denoiser_optim,
        critic_optims=critic_optims,
        connectivity_optim=connectivity_optim,
        device=torch.device("cpu"),
    )
    trainer.critic_augment = Mock(wraps=CriticAugment("isotropic", prob=1.0))
    denoiser_before = _parameters(denoiser)
    critic_before = {axis: _parameters(critics[str(axis)]) for axis in (0, 1, 2)}
    local_before = {
        axis: _parameters(critics[str(axis)].local_output) for axis in (0, 1, 2)
    }

    r1_values = []

    def track_r1(scores, inputs):
        penalties = get_critic_r1(scores, inputs)
        r1_values.append(float(penalties.combine(optim.local_loss_weight).detach()))
        return penalties

    with patch("src.train.engine.get_critic_r1", side_effect=track_r1):
        metrics = trainer.step(0)

    assert math.isfinite(metrics.generator)
    assert math.isfinite(metrics.critic)
    assert math.isfinite(metrics.r1)
    assert math.isfinite(metrics.generator_global)
    assert math.isfinite(metrics.generator_local)
    assert math.isfinite(metrics.critic_global)
    assert math.isfinite(metrics.critic_local)
    assert metrics.generator_connectivity == 0.0
    assert metrics.critic_connectivity == 0.0
    assert len(r1_values) == 3
    assert trainer.critic_augment.apply_pair.call_count == 6
    assert math.isclose(metrics.r1, sum(r1_values), rel_tol=1e-6)
    assert math.isclose(
        metrics.generator,
        metrics.generator_global + optim.local_loss_weight * metrics.generator_local,
        rel_tol=1e-5,
    )
    assert math.isclose(
        metrics.critic,
        metrics.critic_global
        + optim.local_loss_weight * metrics.critic_local
        + 0.5 * optim.r1_gamma * optim.r1_interval * metrics.r1,
        rel_tol=1e-5,
    )
    assert 0 <= metrics.transition < 2
    assert metrics.volume_size == 8
    assert _changed(denoiser_before, denoiser)
    assert all(_changed(critic_before[axis], critics[str(axis)]) for axis in (0, 1, 2))
    assert all(
        _changed(local_before[axis], critics[str(axis)].local_output)
        for axis in (0, 1, 2)
    )
    assert all(not parameter.requires_grad for parameter in ema.parameters())


@pytest.mark.parametrize("axis", (0, 2))
def test_training_step_with_one_axis_updates_only_that_critic(axis: int) -> None:
    trainer, denoiser, streams = _conditioning_trainer(
        anchored=False,
        axes=(axis,),
    )
    denoiser_before = _parameters(denoiser)
    critic_before = _parameters(trainer.critics[str(axis)])

    metrics = trainer.step(0, transition=1)

    assert trainer.active_axes == (axis,)
    assert set(trainer.critics) == {str(axis)}
    assert set(trainer.critic_optims) == {str(axis)}
    assert streams[axis].calls == 1
    assert metrics.critic_axes[axis] != 0.0
    assert all(
        value == 0.0 for index, value in enumerate(metrics.critic_axes) if index != axis
    )
    assert _changed(denoiser_before, denoiser)
    assert _changed(critic_before, trainer.critics[str(axis)])


def test_anchor_training_uses_real_plane_and_updates_adapter() -> None:
    data = DataConfig(
        domains={0: {0: ".", 1: ".", 2: "."}},
        crop_size=8,
        input_size=8,
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
        denoiser_lr=1e-3,
        critic_lr=1e-3,
        beta1=0.0,
        beta2=0.9,
        r1_gamma=0.0,
        r1_interval=2,
        local_loss_weight=0.5,
    )
    cfg = _config(
        data,
        model,
        optim,
        anchor=AnchorConfig(
            train_prob=1.0,
        ),
    )
    denoiser, critics, connectivity_critic = build_models(cfg)
    ema = build_ema(denoiser)
    denoiser_optim, critic_optims, connectivity_optim = build_optimizers(
        denoiser,
        critics,
        connectivity_critic,
        cfg,
    )
    images = torch.randint(0, data.num_phases, (data.batch_size, 8, 8))
    trainer = _make_trainer(
        cfg,
        denoiser=denoiser,
        ema_denoiser=ema,
        critics=critics,
        connectivity_critic=connectivity_critic,
        streams={axis: _ConstantStream(images) for axis in (0, 1, 2)},
        diffusion=Diffusion(2, beta_min=0.1, beta_max=2.0),
        denoiser_optim=denoiser_optim,
        critic_optims=critic_optims,
        connectivity_optim=connectivity_optim,
        device=torch.device("cpu"),
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
            "predict_logits",
            wraps=denoiser.predict_logits,
        ) as predict_logits,
    ):
        metrics = trainer.step(0, transition=0)

    assert metrics.anchor_planes == 1
    assert metrics.anchor_conflict_rate == 0.0
    assert forward.call_count == 1
    assert predict_logits.call_count == 2
    assert all(
        call.kwargs["anchor_image"] is not None
        and call.kwargs["anchor_mask"] is not None
        for call in predict_logits.call_args_list
    )
    assert math.isfinite(metrics.anchor_loss)
    assert metrics.generator_connectivity == 0.0
    assert metrics.critic_connectivity == 0.0
    assert 0.0 <= metrics.anchor_accuracy <= 1.0
    assert math.isclose(
        metrics.generator,
        metrics.generator_global + optim.local_loss_weight * metrics.generator_local,
        rel_tol=1e-5,
    )
    assert math.isclose(
        metrics.generator_total,
        metrics.generator + metrics.anchor_loss + metrics.vf_loss,
        rel_tol=1e-5,
    )
    assert not torch.equal(adapter_before, denoiser.anchor_input.weight.detach())


def test_step_reuses_each_real_batch_and_conditions_every_reverse_step() -> None:
    trainer, denoiser, streams = _conditioning_trainer(
        anchored=True,
    )
    vf_batch_ids = []
    critic_batch_ids = []
    original_build_pool = vf.build_pool
    original_update_critics = trainer.update_critics

    def track_vfs(batches, num_phases):
        vf_batch_ids.extend(id(batches[axis]) for axis in (0, 1, 2))
        return original_build_pool(batches, num_phases)

    def track_critics(transition, fake, batches, step, domain):
        critic_batch_ids.extend(id(batches[axis]) for axis in (0, 1, 2))
        return original_update_critics(transition, fake, batches, step, domain)

    with (
        patch.object(vf, "build_pool", side_effect=track_vfs),
        patch.object(trainer, "update_critics", side_effect=track_critics),
        patch.object(denoiser, "forward", wraps=denoiser.forward) as forward,
        patch.object(
            denoiser,
            "predict_logits",
            wraps=denoiser.predict_logits,
        ) as predict_logits,
    ):
        metrics = trainer.step(0, transition=0)

    assert all(stream.calls == 1 for stream in streams.values())
    assert metrics.vf_active
    assert metrics.vf_loss > 0.0
    assert len(metrics.target_vfs) == 3
    assert math.isclose(sum(metrics.target_vfs), 1.0, rel_tol=1e-6)
    assert math.isclose(sum(metrics.soft_vfs), 1.0, rel_tol=1e-6)
    assert math.isclose(sum(metrics.hard_vfs), 1.0, rel_tol=1e-6)

    calls = [*forward.call_args_list, *predict_logits.call_args_list]
    vfs = [call.kwargs["vf"] for call in calls]
    assert len(forward.call_args_list) == 1
    assert len(predict_logits.call_args_list) == 2
    assert all(vf is vfs[0] for vf in vfs)
    assert vfs[0].shape == (1, 3)
    expected_pool = vf.build_pool(
        {axis: streams[axis].images for axis in (0, 1, 2)},
        num_phases=3,
    )
    assert any(torch.allclose(vfs[0][0], target) for target in expected_pool)
    assert vf_batch_ids == critic_batch_ids

    gradient = denoiser.vf_mlp[-1].weight.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().sum()) > 0.0


def test_generate_pair_keeps_initial_noise_and_posterior_unprojected() -> None:
    trainer, _, _ = _conditioning_trainer(
        anchored=True,
    )
    batches = trainer.get_batches(0)
    selection = trainer.sample_real_anchor(batches, volume_size=8)
    presence = ConditionPresence(
        anchor=torch.tensor((False,)),
        vf=torch.tensor((False,)),
    )
    model_conditions = trainer.make_model_conditions(
        selection.condition,
        torch.tensor(((0.3, 0.3, 0.4),)),
        presence,
        trainer.make_domain(0, 1),
    )
    trainer.diffusion = Diffusion(1)

    initial = torch.linspace(
        -0.9,
        0.9,
        selection.condition.image.numel(),
        device=trainer.device,
    ).reshape_as(selection.condition.image)
    latent = torch.zeros(
        trainer.volume_batch_size,
        trainer.latent_channels,
        device=trainer.device,
    )
    with (
        patch("src.train.engine.torch.randn", return_value=initial.clone()),
        patch.object(trainer, "sample_latent", return_value=latent),
    ):
        previous, current, _, prediction = trainer.generate_pair(
            transition=0,
            model_conditions=model_conditions,
            volume_size=8,
        )

    mask = selection.condition.mask.expand_as(previous)
    clean = selection.condition.image
    assert not bool(model_conditions["anchor_mask"].any())
    assert not bool(model_conditions["vf_present"].any())
    assert torch.equal(current, initial)
    assert torch.equal(previous, prediction)
    assert not torch.equal(previous[mask], clean[mask])
    assert not torch.equal(prediction[mask], clean[mask])


def test_training_volume_and_critic_use_patch_size() -> None:
    trainer, denoiser, streams = _conditioning_trainer(
        anchored=True,
        crop_size=8,
        patch_size=8,
    )
    shapes = []
    hooks = [
        critic.register_forward_pre_hook(
            lambda _module, args: shapes.append(
                (tuple(args[0].shape), tuple(args[1].shape))
            )
        )
        for critic in trainer.critics.values()
    ]
    try:
        with patch.object(
            denoiser,
            "predict_logits",
            wraps=denoiser.predict_logits,
        ) as predict_logits:
            metrics = trainer.step(0, transition=0)
    finally:
        for hook in hooks:
            hook.remove()

    assert metrics.volume_size == 8
    assert all(stream.calls == 1 for stream in streams.values())
    assert shapes
    assert all(previous[-2:] == current[-2:] == (8, 8) for previous, current in shapes)
    assert all(
        call.kwargs["anchor_image"].shape[-3:] == (8, 8, 8)
        for call in predict_logits.call_args_list
    )


def test_training_critics_match_each_axis_rectangular_real_shape() -> None:
    trainer, _, streams = _conditioning_trainer(anchored=True)
    shapes = {
        0: (8, 8),
        1: (4, 8),
        2: (8, 6),
    }
    trainer.r1_gamma = 0.01
    trainer.r1_interval = 1
    for axis, shape in shapes.items():
        streams[axis].images = torch.randint(0, 3, (2, *shape))

    observed = {axis: [] for axis in (0, 1, 2)}
    hooks = [
        trainer.critics[str(axis)].register_forward_pre_hook(
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


def test_real_anchor_preserves_a_rectangular_observation() -> None:
    trainer, _, _ = _conditioning_trainer(anchored=True)
    batches = {axis: torch.randint(0, 3, (2, 4, 8)) for axis in (0, 1, 2)}

    selection = trainer.sample_real_anchor(batches, volume_size=8)

    condition = selection.condition
    coords = condition.mask[0, 0].nonzero()
    spans = tuple(
        int(coords[:, dim].max() - coords[:, dim].min() + 1) for dim in range(3)
    )
    assert condition.image.shape == (1, 3, 8, 8, 8)
    assert sorted(spans) == [1, 4, 8]
    assert int(condition.mask.sum()) == 4 * 8


def test_incompatible_shared_anchor_falls_back_to_an_owned_axis() -> None:
    trainer = object.__new__(Trainer)
    trainer.volume_batch_size = 1
    trainer.num_phases = 2
    trainer.device = torch.device("cpu")
    trainer.anchor_shared_axis_probability = 1.0
    trainer.vf_target_average_max_samples = 1
    own = torch.zeros(1, 8, 8, dtype=torch.long)
    borrowed = torch.ones(1, 8, 8, dtype=torch.long)
    batches = {0: own, 1: borrowed}
    pool = vf.build_pool({0: own}, num_phases=2)

    shared = trainer.sample_real_anchor(
        batches,
        volume_size=8,
        owned_axes=(0,),
    )

    assert shared.source == "shared"
    assert not vf.anchor_is_compatible(
        shared.condition,
        pool,
        batch_size=1,
        num_phases=2,
    )

    fallback = trainer.sample_real_anchor(
        {0: own},
        volume_size=8,
        owned_axes=(0,),
    )
    target, _ = vf.sample_target(
        pool,
        fallback.condition,
        batch_size=1,
        num_phases=2,
        device=torch.device("cpu"),
        max_samples=1,
    )

    assert fallback.source == "real"
    assert target.tolist() == [[1.0, 0.0]]


def test_single_vf_condition_can_be_dropped_for_the_whole_batch() -> None:
    trainer, denoiser, streams = _conditioning_trainer(
        anchored=False,
        cfg_drop_each_probability=0.1,
    )
    hidden = ConditionPresence(
        anchor=torch.zeros(1, dtype=torch.bool),
        vf=torch.zeros(1, dtype=torch.bool),
    )

    with (
        patch.object(trainer, "sample_condition_presence", return_value=hidden),
        patch.object(denoiser, "forward", wraps=denoiser.forward) as forward,
        patch.object(
            denoiser,
            "predict_logits",
            wraps=denoiser.predict_logits,
        ) as predict_logits,
    ):
        metrics = trainer.step(0, transition=0)

    assert all(stream.calls == 1 for stream in streams.values())
    assert not metrics.vf_active
    assert metrics.vf_loss == 0.0
    calls = [*forward.call_args_list, *predict_logits.call_args_list]
    assert len(forward.call_args_list) == 1
    assert len(predict_logits.call_args_list) == 2
    assert all(call.kwargs["vf"] is not None for call in calls)
    assert all(not bool(call.kwargs["vf_present"].any()) for call in calls)
    assert all(
        parameter.grad is None or not bool(parameter.grad.any())
        for parameter in denoiser.vf_mlp.parameters()
    )


def test_joint_cfg_dropout_uses_four_categorical_anchor_vf_states() -> None:
    trainer = object.__new__(Trainer)
    trainer.volume_batch_size = 4
    trainer.device = torch.device("cpu")
    trainer.cfg_drop_each_probability = 0.1

    with patch(
        "src.train.engine.torch.rand",
        return_value=torch.tensor((0.05, 0.15, 0.25, 0.35)),
    ):
        presence = trainer.sample_condition_presence(has_anchor=True)

    assert presence.anchor.tolist() == [False, False, True, True]
    assert presence.vf.tolist() == [False, True, False, True]
    assert presence.fractions() == (0.25, 0.25, 0.25, 0.25)


def test_single_condition_dropout_matches_joint_marginal_visibility() -> None:
    trainer = object.__new__(Trainer)
    trainer.volume_batch_size = 2
    trainer.device = torch.device("cpu")
    trainer.cfg_drop_each_probability = 0.1

    with patch(
        "src.train.engine.torch.rand",
        return_value=torch.tensor((0.19, 0.21)),
    ):
        presence = trainer.sample_condition_presence(has_anchor=False)

    assert presence.anchor.tolist() == [False, False]
    assert presence.vf.tolist() == [False, True]


def test_anchor_specific_losses_stop_when_cfg_hides_the_anchor() -> None:
    trainer, _, _ = _conditioning_trainer(
        anchored=True,
        connectivity_weight=0.25,
    )
    observed = build_anchors(
        (PlaneAnchor(torch.zeros(8, 8, dtype=torch.uint8), axis=0, index=4),),
        batch_size=1,
        num_phases=3,
        volume_size=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert observed is not None
    trainer.connect.record_prior(torch.zeros(1, 3, 8, 8, 8), observed, 0)
    dropped = ConditionPresence(
        anchor=torch.tensor((False,)),
        vf=torch.tensor((True,)),
    )
    generated = {}
    original_generate_pair = trainer.generate_pair

    def capture_generate_pair(*args, **kwargs):
        result = original_generate_pair(*args, **kwargs)
        generated["conditions"] = args[1]
        generated["previous"] = result[0]
        generated["prediction"] = result[3]
        return result

    with (
        patch.object(trainer, "sample_condition_presence", return_value=dropped),
        patch.object(trainer, "generate_pair", side_effect=capture_generate_pair),
    ):
        metrics = trainer.step(0, transition=0)

    conditions = generated["conditions"]
    assert not bool(conditions["anchor_mask"].any())
    anchor_image = conditions["anchor_image"]
    mask = anchor_image.abs().sum(dim=1, keepdim=True).bool()
    mask = mask.expand_as(generated["previous"])
    assert torch.equal(generated["previous"], generated["prediction"])
    assert not torch.equal(generated["previous"][mask], anchor_image[mask])
    assert not torch.equal(generated["prediction"][mask], anchor_image[mask])
    assert metrics.anchor_input_active_fraction == 0.0
    assert metrics.anchor_loss == 0.0
    assert metrics.anchor_coarse_loss == 0.0
    assert metrics.anchor_pixel_loss == 0.0
    assert metrics.generator_connectivity == 0.0
    assert metrics.normal_transition_loss == 0.0


def test_vf_total_variation_uses_raw_prediction() -> None:
    trainer, _, _ = _conditioning_trainer(anchored=True)
    batches = trainer.get_batches(0)
    selection = trainer.sample_real_anchor(batches, volume_size=8)
    condition = selection.condition
    prediction = torch.full((1, 3, 8, 8, 8), -1.0)
    prediction[:, 0] = 1.0
    prediction.requires_grad_()
    raw_probs = (prediction.detach() + 1.0) * 0.5
    target = raw_probs.mean(dim=(2, 3, 4))
    projected = torch.where(
        condition.mask,
        condition.image,
        prediction.detach(),
    )
    projected_vf = ((projected + 1.0) * 0.5).mean(dim=(2, 3, 4))
    logits = torch.zeros_like(prediction, requires_grad=True)

    with (
        patch.object(trainer, "sample_anchor", return_value=selection),
        patch.object(vf, "sample_target", return_value=(target, 0.0)),
        patch.object(
            trainer,
            "generate_pair",
            return_value=(prediction, prediction, logits, prediction),
        ),
    ):
        metrics = trainer.step(0, transition=0)

    projected_tv = 0.5 * (projected_vf - target).abs().sum()
    assert float(projected_tv) > 0.0
    assert metrics.vf_active
    assert math.isclose(metrics.vf_loss, 0.0, abs_tol=1e-7)


def test_real_anchor_ramp_builds_and_refreshes_conditional_prior() -> None:
    trainer, _, _ = _conditioning_trainer(
        anchored=True,
        anchor_start_step=1,
        connectivity_weight=0.25,
    )
    trainer.prior_refresh_steps = 2

    with (
        patch.object(
            trainer.diffusion,
            "sample",
            wraps=trainer.diffusion.sample,
        ) as prior_samples,
        patch("src.train.prior._log_uniform_count", return_value=4),
    ):
        warmup = trainer.step(0, transition=0)
        initial = trainer.step(1, transition=0)
        connectivity_before = _parameters(trainer.connectivity_critic)
        multi = trainer.step(2, transition=0)
        refreshed = trainer.step(3, transition=0)

    assert prior_samples.call_count == 2
    assert warmup.anchor_ramp == 0.0
    assert warmup.anchor_planes == 0
    assert warmup.prior_volumes == 0
    assert warmup.generator_connectivity == 0.0
    assert warmup.critic_connectivity == 0.0
    assert initial.anchor_ramp == 1.0
    assert initial.anchor_planes == 1
    assert initial.prior_volumes == 1
    assert initial.prior_ready
    assert initial.connectivity_triplets == 7
    assert multi.anchor_ramp == 1.0
    assert multi.anchor_planes == 4
    assert multi.connectivity_triplets == 7
    assert multi.generator_connectivity > 0.0
    assert multi.critic_connectivity > 0.0
    assert refreshed.anchor_ramp == 1.0
    assert refreshed.anchor_planes == 1
    assert refreshed.prior_updates == 1
    for call in prior_samples.call_args_list:
        conditions = call.args[3]
        assert bool(conditions["anchor_mask"].any())
        assert not bool(conditions["vf_present"].any())
    assert _changed(connectivity_before, trainer.connectivity_critic)
    assert all(
        parameter.requires_grad
        for parameter in trainer.connectivity_critic.parameters()
    )
    assert math.isclose(
        multi.generator_total,
        multi.generator
        + 0.25 * multi.generator_connectivity
        + multi.anchor_loss
        + multi.vf_loss,
        rel_tol=1e-5,
    )


def test_interrupt_saves_all_weights_and_is_reraised(tmp_path: Path) -> None:
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.ema_denoiser = nn.Linear(2, 2)
    trainer.critics = nn.ModuleDict({str(axis): nn.Linear(2, 1) for axis in range(3)})
    trainer.connectivity_critic = nn.Linear(2, 1)
    trainer.step = Mock(side_effect=KeyboardInterrupt)

    with pytest.raises(KeyboardInterrupt):
        run_training(
            trainer,
            steps=1,
            save_every=1,
            run_dir=tmp_path,
        )

    assert (tmp_path / "generator.pt").is_file()
    assert tuple(path.name for path in sorted(tmp_path.glob("critic_*.pt"))) == (
        "critic_0.pt",
        "critic_1.pt",
        "critic_2.pt",
        "critic_c.pt",
    )


def test_fit_rejects_invalid_checkpoint_interval(tmp_path: Path) -> None:
    trainer = object.__new__(Trainer)

    with pytest.raises(ValueError, match="checkpoint_every"):
        run_training(
            trainer,
            steps=1,
            save_every=1,
            checkpoint_every=0,
            run_dir=tmp_path,
        )


def test_fit_keeps_latest_weights_and_sparse_numbered_checkpoints(
    tmp_path: Path,
) -> None:
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.ema_denoiser = nn.Linear(2, 2)
    trainer.critics = nn.ModuleDict({str(axis): nn.Linear(2, 1) for axis in range(3)})
    trainer.connectivity_critic = nn.Linear(2, 1)
    trainer.step = Mock(
        return_value=Metrics(
            generator=1.0,
            generator_total=1.0,
            critic=1.0,
            r1=0.0,
            transition=0,
            volume_size=8,
            domain=0,
            critic_axes=(1.0, 1.0, 1.0),
            anchor_planes=0,
            anchor_conflict_rate=0.0,
            anchor_loss=0.0,
            anchor_accuracy=0.0,
            generator_connectivity=0.0,
            critic_connectivity=0.0,
            connectivity_r1=0.0,
            anchor_ramp=0.0,
            connectivity_triplets=0,
            prior_volumes=0,
            prior_mebibytes=0.0,
            prior_ready=False,
        )
    )

    weights = run_training(
        trainer,
        steps=3,
        save_every=1,
        checkpoint_every=2,
        run_dir=tmp_path,
    )

    assert weights == tmp_path / "generator.pt"
    assert weights.is_file()
    checkpoints = tuple((tmp_path / "checkpoints").iterdir())
    assert tuple(path.name for path in checkpoints) == ("step_00000002",)
    assert (checkpoints[0] / "generator.pt").is_file()


def _parameters(model: nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in model.parameters())


def _changed(before: tuple[torch.Tensor, ...], model: nn.Module) -> bool:
    return any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.parameters(), strict=True)
    )


def _conditioning_trainer(
    *,
    anchored: bool,
    crop_size: int = 8,
    patch_size: int = 8,
    anchor_start_step: int = 0,
    anchor_ramp_steps: int = 0,
    connectivity_weight: float = 0.0,
    normal_transition_weight: float = 0.0,
    cfg_drop_each_probability: float = 0.0,
    axes: tuple[int, ...] = (0, 1, 2),
) -> tuple[Trainer, nn.Module, dict[int, _ConstantStream]]:
    data = DataConfig(
        domains={0: {axis: "." for axis in axes}},
        crop_size=crop_size,
        input_size=patch_size,
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
        denoiser_lr=1e-3,
        critic_lr=1e-3,
        beta1=0.0,
        beta2=0.9,
        r1_gamma=0.0,
        r1_interval=2,
        local_loss_weight=0.5,
    )
    cfg = _config(
        data,
        model,
        optim,
        anchor=AnchorConfig(
            train_prob=1.0 if anchored else 0.0,
            start_step=anchor_start_step,
            ramp_steps=anchor_ramp_steps,
        ),
        connectivity=ConnectivityConfig(
            weight=connectivity_weight,
            phase_transition_weight=normal_transition_weight,
        ),
        conditioning=ConditioningConfig(
            joint_each_prob=cfg_drop_each_probability,
        ),
    )
    denoiser, critics, connectivity_critic = build_models(cfg)
    ema = build_ema(denoiser)
    denoiser_optim, critic_optims, connectivity_optim = build_optimizers(
        denoiser,
        critics,
        connectivity_critic,
        cfg,
    )
    base = torch.arange(2 * crop_size * crop_size).reshape(
        2,
        crop_size,
        crop_size,
    )
    streams = {
        axis: _ConstantStream((base + axis).remainder(3).to(torch.long))
        for axis in axes
    }
    trainer = _make_trainer(
        cfg,
        denoiser=denoiser,
        ema_denoiser=ema,
        critics=critics,
        connectivity_critic=connectivity_critic,
        streams=streams,
        diffusion=Diffusion(2, beta_min=0.1, beta_max=2.0),
        denoiser_optim=denoiser_optim,
        critic_optims=critic_optims,
        connectivity_optim=connectivity_optim,
        device=torch.device("cpu"),
    )
    return trainer, denoiser, streams


def _make_trainer(
    cfg: TrainConfig,
    *,
    denoiser,
    ema_denoiser,
    critics,
    connectivity_critic,
    streams,
    streams_by_domain=False,
    diffusion,
    denoiser_optim,
    critic_optims,
    connectivity_optim,
    device,
) -> Trainer:
    use_amp = cfg.train.amp and device.type == "cuda"
    return Trainer(
        components=TrainerComponents(
            denoiser=denoiser,
            ema_denoiser=ema_denoiser,
            critics=critics,
            connectivity_critic=connectivity_critic,
            streams=streams if streams_by_domain else {0: streams},
            diffusion=diffusion,
            denoiser_optim=denoiser_optim,
            critic_optims=critic_optims,
            connectivity_optim=connectivity_optim,
            scaler=torch.amp.GradScaler("cuda", enabled=use_amp),
            device=device,
        ),
        settings=TrainerSettings(
            volume_batch_size=cfg.train.volume_batch_size,
            num_phases=cfg.data.num_phase,
            patch_size=cfg.data.input_size,
            slice_pairs_per_axis=cfg.train.pairs_per_axis,
            ema_decay=cfg.optim.ema_decay,
            r1_gamma=cfg.model.critic.r1_weight,
            r1_interval=cfg.model.critic.r1_interval,
            critic_local_weight=cfg.model.critic.local_loss_weight,
            anchor_training_probability=cfg.anchor.train_prob,
            anchor_start_step=cfg.anchor.start_step,
            anchor_ramp_steps=cfg.anchor.ramp_steps,
            anchor_shared_axis_probability=cfg.anchor.cross_domain_prob,
            anchor_pixel_loss_weight=cfg.anchor.pixel_weight,
            connectivity_weight=cfg.anchor.connectivity.weight,
            normal_transition_weight=(cfg.anchor.connectivity.phase_transition_weight),
            connectivity_bank_size=cfg.anchor.connectivity.volume_count,
            connectivity_refresh_steps=cfg.anchor.connectivity.refresh_every,
            vf_loss_weight=cfg.vf.weight,
            vf_target_average_max_samples=cfg.vf.max_samples,
            domain_dropout=1.0 - cfg.data.domain_prob,
            cfg_drop_each_probability=cfg.condition_dropout.joint_each_prob,
            latent_channels=cfg.model.generator.latent_channels,
            amp_enabled=use_amp,
        ),
    )


def _config(
    data: DataConfig,
    model: ModelConfig,
    optim: OptimConfig,
    *,
    anchor: AnchorConfig | None = None,
    connectivity: Config | None = None,
    vf: VfConfig | None = None,
    conditioning: Config | None = None,
) -> TrainConfig:
    anchor = AnchorConfig() if anchor is None else anchor
    anchor["connectivity"] = (
        ConnectivityConfig() if connectivity is None else connectivity
    )
    return TrainConfig(
        data=DataConfig(
            domains=data.domains,
            num_phase=data.num_phases,
            crop_partial=data.get("allow_partial_crop", False),
            crop_size=data.crop_size,
            input_size=data.input_size,
            augment=data.get("augment", False),
            augment_prob=data.get("augment_prob", 0.0),
            domain_prob=1.0 - data.get("domain_dropout", 0.0),
            batch_size=data.batch_size,
            num_workers=data.get("num_workers", 0),
        ),
        model=ModelConfig(
            grad_checkpoint=model.gradient_checkpointing,
            generator=Config(
                channels=[
                    model.base_channels * multiplier
                    for multiplier in model.channel_multipliers
                ],
                condition_channels=model.embedding_channels,
                latent_channels=model.latent_channels,
            ),
            critic=Config(
                channels=model.critic_channels,
                local_loss_weight=optim.local_loss_weight,
                r1_weight=optim.r1_gamma,
                r1_interval=optim.r1_interval,
            ),
        ),
        diffusion=DiffusionConfig(
            steps=2,
            beta_min=0.1,
            beta_max=2.0,
        ),
        anchor=anchor,
        condition_dropout=(
            ConditioningConfig() if conditioning is None else conditioning
        ),
        vf=(VfConfig(max_samples=1, weight=1.0) if vf is None else vf),
        optim=OptimConfig(
            generator_lr=optim.denoiser_lr,
            critic_lr=optim.critic_lr,
            adam_betas=[optim.beta1, optim.beta2],
            ema_decay=0.9,
        ),
        train=LoopConfig(
            init_weights=None,
            steps=1,
            volume_batch_size=1,
            pairs_per_axis=2,
            amp=False,
            update_weights_every=1,
            archive_every=1,
        ),
    )
