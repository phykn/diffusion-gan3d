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
from src.train.relation import RelationBank, RelationLoss
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
        training_probability=0.0,
        start_step=0,
        ramp_steps=0,
        multi_anchor_prob=0.0,
        max_density=0.05,
        min_spacing=2,
        mixed_axis_prob=0.5,
        teacher_bank_size_mib=1,
        loss_weight=0.0,
        shared_axis_probability=0.0,
    )
    cfg.update(values)
    return cfg


def VfConfig(**values):
    return Config(**values)


def ConnectivityConfig(**values):
    cfg = Config(
        loss_weight=0.0,
        normal_transition_loss_weight=0.0,
        replay_triplets_per_axis=1,
        replay_capacity_per_axis=4,
        max_triplets_per_step=4,
        reversal_invariant=True,
    )
    cfg.update(values)
    return cfg


def ConditioningConfig(**values):
    dropout = Config(
        drop_each_prob=0.0,
        single_condition_drop_prob=0.0,
    )
    dropout.update(values)
    return Config(cfg_dropout=dropout)


def get_vf_pool(batches, num_phases):
    trainer = object.__new__(Trainer)
    trainer.num_phases = num_phases
    return trainer.get_vf_pool(batches)


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
    )
    fake = TripletBatch(
        values=torch.ones(2, 3, 2, 1, 1),
        axes=axes,
        center_slots=fake_centers,
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


def test_get_vf_pool_preserves_each_axis_crop_as_an_empirical_target() -> None:
    batches = {
        0: torch.tensor([[[0, 0], [1, 1]], [[2, 2], [2, 2]]]),
        1: torch.tensor([[[1, 1], [1, 2]], [[0, 1], [2, 2]]]),
        2: torch.tensor([[[0, 0], [0, 0]], [[0, 1], [1, 2]]]),
    }
    expected = torch.tensor(
        (
            (0.5, 0.5, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.75, 0.25),
            (0.25, 0.25, 0.5),
            (1.0, 0.0, 0.0),
            (0.25, 0.5, 0.25),
        )
    )

    vfs = get_vf_pool(batches, 3)

    assert torch.equal(vfs, expected)


def test_empirical_vf_target_resamples_for_anchor_minima_and_rejects_no_match() -> None:
    trainer = object.__new__(Trainer)
    trainer.num_phases = 2
    trainer.volume_batch_size = 1
    trainer.device = torch.device("cpu")
    condition = build_anchors(
        (PlaneAnchor(torch.ones(2, 2, dtype=torch.uint8), axis=0, index=0),),
        batch_size=1,
        num_phases=2,
        volume_size=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        reconcile=False,
    )
    assert condition is not None
    pool = torch.tensor(((0.75, 0.25), (0.5, 0.5)))

    with patch(
        "src.train.engine.torch.randint",
        side_effect=(torch.tensor((0,)), torch.tensor(0)),
    ):
        target, resample_rate = trainer.sample_target_vf(pool, condition)

    assert torch.equal(target, pool[1:])
    assert resample_rate == 1.0

    with (
        patch("src.train.engine.torch.randint", return_value=torch.tensor((0,))),
        pytest.raises(ValueError, match="incompatible"),
    ):
        trainer.sample_target_vf(pool[:1], condition)


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
            training_probability=1.0,
            loss_weight=1.0,
            shared_axis_probability=1.0,
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
    trainer.relation_loss_weight = 0.02

    def shared_relation_loss(probs, condition, visible, *, domain):
        assert domain == 0
        assert bool(visible.all())
        assert not bool(condition.axis_masks[:, 0].any())
        assert bool(condition.axis_masks[:, 1:].any())
        zero = probs.sum() * 0.0
        return RelationLoss(
            loss=zero,
            phase=zero,
            support=zero,
            minus=zero,
            plus=zero,
            queries=1,
            matches=1,
            domain_matches=0,
            shared_matches=1,
            ood_rejections=0,
            missing_references=0,
            distance_weights=probs.new_zeros(probs.shape[2] - 1),
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
        with (
            patch.object(trainer, "sample_target_domain", return_value=0),
            patch.object(trainer.relation, "prior_ready", return_value=True),
            patch.object(trainer.relation, "needs_data", return_value=False),
            patch.object(
                trainer.relation,
                "loss",
                side_effect=shared_relation_loss,
            ),
        ):
            metrics = trainer.step(0, transition=1)
    finally:
        for hook in hooks:
            hook.remove()

    assert metrics.domain == 0
    assert metrics.anchor_shared
    assert metrics.relation_queries == metrics.relation_shared_matches == 1
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
            training_probability=1.0,
            loss_weight=1.0,
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
    original_update_critics = trainer.update_critics

    def track_vfs(batches):
        vf_batch_ids.extend(id(batches[axis]) for axis in (0, 1, 2))
        return get_vf_pool(batches, trainer.num_phases)

    def track_critics(transition, fake, batches, step, domain):
        critic_batch_ids.extend(id(batches[axis]) for axis in (0, 1, 2))
        return original_update_critics(transition, fake, batches, step, domain)

    with (
        patch.object(trainer, "get_vf_pool", side_effect=track_vfs),
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
    expected_pool = get_vf_pool(
        {axis: streams[axis].images for axis in (0, 1, 2)},
        3,
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

    assert selection.seeds[0].image.shape == (4, 8)
    assert int(selection.condition.mask.sum()) == 4 * 8


def test_incompatible_shared_anchor_falls_back_to_an_owned_axis() -> None:
    trainer = object.__new__(Trainer)
    trainer.volume_batch_size = 1
    trainer.num_phases = 2
    trainer.device = torch.device("cpu")
    trainer.anchor_shared_axis_probability = 1.0
    own = torch.zeros(1, 8, 8, dtype=torch.long)
    borrowed = torch.ones(1, 8, 8, dtype=torch.long)
    batches = {0: own, 1: borrowed}
    pool = trainer.get_vf_pool({0: own})

    shared = trainer.sample_real_anchor(
        batches,
        volume_size=8,
        owned_axes=(0,),
    )

    assert shared.source == "shared"
    assert not trainer.anchor_has_compatible_vf(shared.condition, pool)

    fallback = trainer.sample_real_anchor(
        {0: own},
        volume_size=8,
        owned_axes=(0,),
    )
    target, _ = trainer.resolve_target_vf(fallback, pool, fallback.condition)

    assert fallback.source == "real"
    assert target.tolist() == [[1.0, 0.0]]


def test_single_vf_condition_can_be_dropped_for_the_whole_batch() -> None:
    trainer, denoiser, streams = _conditioning_trainer(
        anchored=False,
        cfg_single_drop_probability=1.0,
    )

    with (
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
    trainer.cfg_single_drop_probability = 0.2

    with patch(
        "src.train.engine.torch.rand",
        return_value=torch.tensor((0.05, 0.15, 0.25, 0.35)),
    ):
        presence = trainer.sample_condition_presence(has_anchor=True)

    assert presence.anchor.tolist() == [False, False, True, True]
    assert presence.vf.tolist() == [False, True, False, True]
    assert presence.fractions() == (0.25, 0.25, 0.25, 0.25)


def test_anchor_specific_losses_stop_when_cfg_hides_the_anchor() -> None:
    trainer, _, _ = _conditioning_trainer(
        anchored=True,
        multi_probability=1.0,
        connectivity_weight=0.25,
    )
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
    assert metrics.relation_loss == 0.0
    assert metrics.teacher_volumes == 0
    assert metrics.connectivity_replay == 3


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
        patch.object(trainer, "sample_target_vf", return_value=(target, 0.0)),
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


def test_unconditional_warmup_populates_reference_before_anchor_connectivity() -> None:
    trainer, _, _ = _conditioning_trainer(
        anchored=True,
        anchor_start_step=1,
        connectivity_weight=0.25,
    )

    with patch.object(
        trainer.ema_denoiser,
        "forward",
        wraps=trainer.ema_denoiser.forward,
    ) as ema_forward:
        warmup = trainer.step(0, transition=0)
        connectivity_before = _parameters(trainer.connectivity_critic)
        anchored = trainer.step(1, transition=0)

    assert ema_forward.call_count == 0
    assert warmup.anchor_ramp == 0.0
    assert warmup.anchor_planes == 0
    assert warmup.connectivity_replay == 3
    assert warmup.generator_connectivity == 0.0
    assert warmup.critic_connectivity == 0.0
    assert anchored.anchor_ramp == 1.0
    assert anchored.anchor_planes == 1
    assert not anchored.anchor_teacher
    assert anchored.teacher_volumes == 0
    assert anchored.connectivity_triplets == 1
    assert anchored.generator_connectivity > 0.0
    assert anchored.critic_connectivity > 0.0
    assert _changed(connectivity_before, trainer.connectivity_critic)
    assert all(
        parameter.requires_grad
        for parameter in trainer.connectivity_critic.parameters()
    )
    assert math.isclose(
        anchored.generator_total,
        anchored.generator
        + 0.25 * anchored.generator_connectivity
        + anchored.anchor_loss
        + anchored.vf_loss,
        rel_tol=1e-5,
    )


@pytest.mark.parametrize("anchored", (False, True))
def test_full_anchor_free_ema_sample_populates_frozen_relation_bank(
    anchored: bool,
) -> None:
    trainer, _, _ = _conditioning_trainer(anchored=anchored)
    trainer.relation_loss_weight = 0.02
    trainer.relation_start_step = 0
    trainer.relation = RelationBank(
        num_domains=1,
        num_phases=3,
        axes=(0, 1, 2),
        capacity_per_axis=2,
        profiles_per_axis=2,
        neighbors=2,
        quantile_low=0.1,
        quantile_high=0.9,
    )

    with patch.object(
        trainer.ema_denoiser,
        "forward",
        wraps=trainer.ema_denoiser.forward,
    ) as ema_forward:
        metrics = trainer.step(0, transition=0)

    assert ema_forward.call_count == trainer.diffusion.timesteps
    assert metrics.relation_bank_entries == 12
    assert metrics.relation_ready_buckets == 6


def test_relation_reference_sampling_preserves_training_rng() -> None:
    trainer, _, _ = _conditioning_trainer(anchored=False)
    trainer.relation_loss_weight = 0.02
    trainer.relation_start_step = 0
    trainer.relation = RelationBank(
        num_domains=1,
        num_phases=3,
        axes=(0, 1, 2),
        capacity_per_axis=2,
        profiles_per_axis=2,
        neighbors=2,
        quantile_low=0.1,
        quantile_high=0.9,
    )
    prepared = trainer.prepare_step(0, transition=0)
    before = torch.random.get_rng_state().clone()

    trainer.record_relation_reference(step=0, prepared=prepared)

    assert torch.equal(torch.random.get_rng_state(), before)


def test_relation_prior_freezes_before_anchor_training_begins() -> None:
    trainer, _, _ = _conditioning_trainer(anchored=True)
    trainer.relation_loss_weight = 0.02
    trainer.relation_start_step = 0
    trainer.relation = RelationBank(
        num_domains=1,
        num_phases=3,
        axes=(0, 1, 2),
        capacity_per_axis=2,
        profiles_per_axis=2,
        neighbors=2,
        quantile_low=0.1,
        quantile_high=0.9,
    )
    captured = {}
    original_sample = trainer.diffusion.sample

    def capture_sample(model, initial, latent_channels, conditions):
        captured["conditions"] = conditions
        return original_sample(model, initial, latent_channels, conditions)

    with patch.object(trainer.diffusion, "sample", side_effect=capture_sample):
        prior_step = trainer.step(0, transition=0)
    anchored_step = trainer.step(1, transition=0)

    conditions = captured["conditions"]
    assert "anchor_image" not in conditions
    assert "anchor_mask" not in conditions
    assert not bool(conditions["vf_present"].any())
    assert torch.equal(conditions["vf"], torch.zeros_like(conditions["vf"]))
    assert prior_step.anchor_ramp == 0.0
    assert prior_step.anchor_planes == 0
    assert prior_step.relation_prior_ready
    assert anchored_step.anchor_ramp == 1.0
    assert anchored_step.anchor_planes == 1


def test_relation_bank_uses_one_fixed_ema_snapshot_until_complete() -> None:
    trainer, _, _ = _conditioning_trainer(anchored=True)
    trainer.relation_loss_weight = 0.02
    trainer.relation_start_step = 0
    trainer.relation = RelationBank(
        num_domains=1,
        num_phases=3,
        axes=(0, 1, 2),
        capacity_per_axis=3,
        profiles_per_axis=1,
        neighbors=2,
        quantile_low=0.1,
        quantile_high=0.9,
    )
    snapshot = _parameters(trainer.ema_denoiser)

    first = trainer.step(0, transition=0)
    second = trainer.step(1, transition=0)

    assert not first.relation_prior_ready
    assert not second.relation_prior_ready
    assert not _changed(snapshot, trainer.ema_denoiser)

    third = trainer.step(2, transition=0)

    assert third.relation_prior_ready
    assert _changed(snapshot, trainer.ema_denoiser)


def test_relation_loss_uses_diffusion_signal_weight_in_generator_total() -> None:
    trainer, _, _ = _conditioning_trainer(anchored=True)
    trainer.relation_loss_weight = 0.02

    def relation_loss(probs, condition, visible, *, domain):
        del condition, visible, domain
        zero = probs.sum() * 0.0
        return RelationLoss(
            loss=zero + 2.0,
            phase=zero + 1.5,
            support=zero + 0.5,
            minus=zero + 0.75,
            plus=zero + 1.25,
            queries=1,
            matches=1,
            domain_matches=0,
            shared_matches=1,
            ood_rejections=0,
            missing_references=0,
            distance_weights=probs.new_tensor((0.75, 0.25, 0, 0, 0, 0, 0)),
        )

    with (
        patch.object(trainer.relation, "prior_ready", return_value=True),
        patch.object(trainer.relation, "needs_data", return_value=False),
        patch.object(trainer.relation, "loss", side_effect=relation_loss),
    ):
        metrics = trainer.step(0, transition=0)

    temporal = float(trainer.diffusion.alpha_bars[1])
    expected = (
        metrics.generator
        + metrics.anchor_loss
        + metrics.vf_loss
        + 0.02 * temporal * metrics.relation_loss
    )
    assert metrics.relation_queries == metrics.relation_matches == 1
    assert metrics.relation_shared_matches == 1
    assert math.isclose(
        metrics.relation_weighted_loss,
        0.02 * temporal * metrics.relation_loss,
        rel_tol=1e-6,
    )
    assert metrics.relation_phase_loss == 1.5
    assert metrics.relation_support_loss == 0.5
    assert metrics.relation_minus_loss == 0.75
    assert metrics.relation_plus_loss == 1.25
    assert metrics.relation_distance_weights[:2] == (0.75, 0.25)
    assert math.isclose(metrics.generator_total, expected, rel_tol=1e-5)


def test_single_real_prediction_is_promoted_only_on_a_later_step() -> None:
    trainer, _, _ = _conditioning_trainer(
        anchored=True,
        multi_probability=1.0,
    )
    trainer.connect.teacher_min_entries = 1

    first = trainer.step(0, transition=0)
    second = trainer.step(1, transition=0)

    assert first.anchor_planes == 1
    assert not first.anchor_teacher
    assert first.teacher_volumes == 1
    assert second.anchor_planes >= 2
    assert second.anchor_teacher
    assert second.teacher_volumes == 1
    assert math.isclose(sum(second.target_vfs), 1.0, rel_tol=1e-6)


def test_nonterminal_prediction_does_not_enter_teacher_bank() -> None:
    trainer, _, _ = _conditioning_trainer(
        anchored=True,
        multi_probability=1.0,
    )
    trainer.connect.teacher_min_entries = 1

    metrics = trainer.step(0, transition=1)

    assert metrics.anchor_planes == 1
    assert metrics.teacher_volumes == 0


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
            connectivity_replay=0,
            anchor_teacher=False,
            teacher_volumes=0,
            teacher_mebibytes=0.0,
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
    multi_probability: float = 0.0,
    connectivity_weight: float = 0.0,
    normal_transition_weight: float = 0.0,
    cfg_drop_each_probability: float = 0.0,
    cfg_single_drop_probability: float = 0.0,
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
            training_probability=1.0 if anchored else 0.0,
            start_step=anchor_start_step,
            ramp_steps=anchor_ramp_steps,
            multi_anchor_prob=multi_probability,
            loss_weight=1.0 if anchored else 0.0,
        ),
        connectivity=ConnectivityConfig(
            loss_weight=connectivity_weight,
            normal_transition_loss_weight=normal_transition_weight,
        ),
        conditioning=ConditioningConfig(
            drop_each_prob=cfg_drop_each_probability,
            single_condition_drop_prob=cfg_single_drop_probability,
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
    use_amp = cfg.train.mixed_precision and device.type == "cuda"
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
            num_phases=cfg.data.num_phases,
            patch_size=cfg.data.input_size,
            slice_pairs_per_axis=cfg.train.slice_pairs_per_axis,
            ema_decay=cfg.train.ema_decay,
            r1_gamma=cfg.optim.r1_gamma,
            r1_interval=cfg.optim.r1_interval,
            critic_local_weight=cfg.optim.local_loss_weight,
            anchor_training_probability=cfg.anchor.training_probability,
            anchor_start_step=cfg.anchor.start_step,
            anchor_ramp_steps=cfg.anchor.ramp_steps,
            anchor_multi_probability=cfg.anchor.multi_anchor_prob,
            anchor_max_density=cfg.anchor.max_density,
            anchor_min_spacing=cfg.anchor.min_spacing,
            anchor_mixed_axis_probability=cfg.anchor.mixed_axis_prob,
            anchor_teacher_bank_mebibytes=cfg.anchor.teacher_bank_size_mib,
            anchor_loss_weight=cfg.anchor.loss_weight,
            anchor_shared_axis_probability=cfg.anchor.get(
                "shared_axis_probability",
                0.0,
            ),
            connectivity_weight=cfg.connectivity.loss_weight,
            normal_transition_weight=(cfg.connectivity.normal_transition_loss_weight),
            connectivity_replay_triplets_per_axis=(
                cfg.connectivity.replay_triplets_per_axis
            ),
            connectivity_replay_capacity_per_axis=(
                cfg.connectivity.replay_capacity_per_axis
            ),
            connectivity_max_triplets_per_step=(cfg.connectivity.max_triplets_per_step),
            vf_loss_weight=cfg.vf.loss_weight,
            domain_dropout=cfg.data.get("domain_dropout", 0.0),
            cfg_drop_each_probability=(cfg.conditioning.cfg_dropout.drop_each_prob),
            cfg_single_drop_probability=(
                cfg.conditioning.cfg_dropout.single_condition_drop_prob
            ),
            latent_channels=cfg.model.latent_channels,
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
    return TrainConfig(
        data=data,
        model=model,
        diffusion=DiffusionConfig(
            timesteps=2,
            beta_min=0.1,
            beta_max=2.0,
        ),
        anchor=AnchorConfig() if anchor is None else anchor,
        connectivity=(ConnectivityConfig() if connectivity is None else connectivity),
        conditioning=(ConditioningConfig() if conditioning is None else conditioning),
        vf=(
            VfConfig(
                loss_weight=1.0,
            )
            if vf is None
            else vf
        ),
        optim=optim,
        train=LoopConfig(
            total_steps=1,
            volume_batch_size=1,
            slice_pairs_per_axis=2,
            mixed_precision=False,
            ema_decay=0.9,
            save_every_steps=1,
        ),
    )
