from __future__ import annotations

import copy
import math
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from src.anchor import PlaneAnchor, encode_anchors
from src.build.model import build_models
from src.build.trainer import build_optimizers
from src.data.augment import CriticAugment
from src.data.slice import TripletBatch
from src.data.slice import sample_pairs as sample_volume_pairs
from src.evaluate.label import compute_vf
from src.model.diffusion import Diffusion
from src.model.layers import NULL_DOMAIN
from src.plane import PLANE_DIRECTIONS, PLANES
from src.prepare.resize import phase_channels
from src.train.anchor_bank import AnchorBank
from src.train.batch import ConditionPresence, RealBatch
from src.train.ema import build_ema
from src.train.loss.gan import get_critic_r1
from src.train.metrics import Metrics
from src.train.run.loop import run_train
from src.train.trainer import (
    Trainer,
    TrainerComponents,
    TrainerSettings,
)


class Config(dict):
    def __getattr__(self, name):
        return self[name]


def _anchor_config(**values):
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


def _connectivity_config(**values):
    cfg = Config(
        weight=0.0,
        phase_transition_weight=0.0,
    )
    cfg.update(values)
    return cfg


def _conditioning_config(**values):
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
    condition = None
    if axis_masks is not None:
        planes = []
        for normal in range(3):
            for _, z, y, x in axis_masks[:, normal].nonzero().tolist():
                coords = [z, y, x]
                planes.append(
                    PlaneAnchor(
                        torch.zeros(1, 1, dtype=torch.long),
                        normal,
                        coords[normal],
                        tuple(c for a, c in enumerate(coords) if a != normal),
                    )
                )
        condition = encode_anchors(
            planes,
            previous.shape[0],
            2,
            previous.shape[-3:],
            previous.device,
            previous.dtype,
        )
    return sample_volume_pairs(
        previous,
        current,
        axis,
        count,
        patch_size if crop_shape is None else crop_shape,
        anchor=condition,
    )


def test_connectivity_augmentation_preserves_triplet_center_slots() -> None:
    trainer = object.__new__(Trainer)
    trainer.profile_settings = {"enabled": False}
    trainer.num_phases = 2
    trainer.patch_size = 1
    trainer.device = torch.device("cpu")
    trainer.connectivity_weight = 1.0
    trainer.normal_transition_weight = 1.0
    trainer.diagnostics = {}
    real_centers = torch.tensor((1, 1))
    fake_centers = torch.tensor((0, 2))
    axes = torch.tensor((0, 0))
    real = TripletBatch(
        values=torch.zeros(2, 3, 2, 1, 1),
        axes=axes,
        gaps=torch.ones(2, dtype=torch.long),
        center_slots=real_centers,
    )
    fake = TripletBatch(
        values=torch.ones(2, 3, 2, 1, 1),
        axes=axes,
        gaps=torch.ones(2, dtype=torch.long),
        center_slots=fake_centers,
    )
    anchor = encode_anchors(
        (PlaneAnchor(torch.zeros(3, 3, dtype=torch.uint8), axis=0, index=1),),
        batch_size=1,
        num_phases=2,
        volume_size=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        reconcile=False,
    )
    assert anchor is not None
    trainer.anchor_triplets = Mock()
    trainer.anchor_triplets.sample.return_value = (real, fake)
    trainer.critic_augment = Mock()
    trainer.critic_augment.apply_together.return_value = (
        real.values + 2.0,
        fake.values + 3.0,
    )

    augmented_real, augmented_fake = trainer.make_connectivity_triplets(
        torch.zeros(1, 2, 3, 3, 3),
        torch.zeros(1, 2, 3, 3, 3),
        anchor,
        transition=0,
        source="multi",
    )

    assert augmented_real.center_slots.tolist() == [1, 1]
    assert augmented_fake.center_slots.tolist() == [0, 2]
    assert torch.equal(augmented_real.values, real.values + 2.0)
    assert torch.equal(augmented_fake.values, fake.values + 3.0)


def test_anchor_transitions_prioritize_the_final_step() -> None:
    trainer = object.__new__(Trainer)
    trainer.profile_settings = {"enabled": False}
    trainer.diffusion = Diffusion(11)

    with (
        patch("src.train.trainer.torch.rand", return_value=torch.tensor(0.1)),
        patch("src.train.trainer.torch.randint") as randint,
    ):
        assert trainer.sample_transition(anchored=True) == 0
        randint.assert_not_called()

    with (
        patch("src.train.trainer.torch.rand", return_value=torch.tensor(0.5)),
        patch(
            "src.train.trainer.torch.randint", return_value=torch.tensor(7)
        ) as randint,
    ):
        assert trainer.sample_transition(anchored=True) == 7
        randint.assert_called_once_with(1, 11, ())

    with patch(
        "src.train.trainer.torch.randint",
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

    def next(self) -> dict:
        self.calls += 1
        return {"image": self.images.clone()}


def test_get_batches_uses_one_domain_for_all_axes() -> None:
    trainer = object.__new__(Trainer)
    trainer.height_data = None
    trainer.profile_settings = {"enabled": False}
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

    assert all(bool((batch == 1).all()) for batch in batches.images.values())
    assert all(stream.calls == 0 for stream in streams[0].values())
    assert all(stream.calls == 1 for stream in streams[1].values())


def test_missing_axes_borrow_from_axis_providers() -> None:
    trainer = object.__new__(Trainer)
    trainer.height_data = None
    trainer.profile_settings = {"enabled": False}
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
    assert [int(batches.images[axis][0, 0, 0]) for axis in (0, 1, 2)] == [10, 21, 22]
    assert critic_domains == {0: 0, 1: NULL_DOMAIN, 2: NULL_DOMAIN}


def test_connectivity_uses_axis_critic_domain_for_shared_context() -> None:
    triplets = TripletBatch(
        values=torch.zeros(3, 3, 2, 4, 4),
        axes=torch.tensor((0, 1, 2)),
        gaps=torch.ones(3, dtype=torch.long),
        center_slots=torch.ones(3, dtype=torch.long),
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
    trainer.profile_settings = {"enabled": False}
    trainer.domain_dropout = 0.0
    assert trainer.sample_domain_condition(2) == 2

    trainer.domain_dropout = 1.0
    assert trainer.sample_domain_condition(2) == NULL_DOMAIN


def test_anchor_training_alternates_external_and_multi_anchor_modes() -> None:
    trainer = object.__new__(Trainer)
    trainer.profile_settings = {"enabled": False}
    trainer.anchor_training_probability = 0.5
    trainer.use_multi_anchor_next = False
    trainer.anchor_bank = AnchorBank()
    trainer.volume_batch_size = 1
    trainer.device = torch.device("cpu")
    trainer.sample_real_anchor = Mock(return_value=Mock(source="real"))

    with patch("src.train.trainer.torch.rand", return_value=torch.tensor(0.9)):
        assert trainer.sample_anchor(RealBatch({}), 2, owned_axes=()) is None
    assert not trainer.use_multi_anchor_next

    trainer.anchor_training_probability = 1.0
    sources = [
        trainer.sample_anchor(RealBatch({}), 2, owned_axes=()).source for _ in range(3)
    ]

    assert sources == [
        "real",
        "real",
        "real",
    ]  # Replay waits for a measured-conditioned sample.


def test_training_step_uses_null_critics_for_borrowed_axes() -> None:
    data = Config(
        domains={0: {0: "."}, 1: {0: ".", 1: ".", 2: "."}},
        crop_size=8,
        input_size=8,
        num_phases=2,
        batch_size=2,
        domain_dropout=0.0,
    )
    model = Config(
        base_channels=4,
        channel_multipliers=(1, 2),
        embedding_channels=8,
        latent_channels=4,
        critic_channels=(4, 8),
        gradient_checkpointing=False,
    )
    optim = Config(
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
        anchor=_anchor_config(
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
        0: {0: _ConstantStream(phase_channels(images, data.num_phases))},
        1: {
            axis: _ConstantStream(phase_channels(images, data.num_phases))
            for axis in (0, 1, 2)
        },
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
        critics[PLANES[axis]].register_forward_pre_hook(
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
    assert streams[0][0].calls == 1
    assert streams[1][0].calls == 0
    assert streams[1][1].calls == 1
    assert streams[1][2].calls == 1
    assert all(set(values) == {0} for values in observed_domains[0])
    assert all(set(values) == {NULL_DOMAIN} for values in observed_domains[1])
    assert all(set(values) == {NULL_DOMAIN} for values in observed_domains[2])


def test_training_step_updates_denoiser_and_all_critics() -> None:
    data = Config(
        domains={0: {0: ".", 1: ".", 2: "."}},
        crop_size=8,
        input_size=8,
        num_phases=3,
        batch_size=2,
    )
    model = Config(
        base_channels=4,
        channel_multipliers=(1, 2),
        embedding_channels=8,
        latent_channels=4,
        critic_channels=(4, 8),
        gradient_checkpointing=False,
    )
    optim = Config(
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
    streams = {
        axis: _ConstantStream(phase_channels(images, data.num_phases))
        for axis in (0, 1, 2)
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
    trainer.critic_augment = Mock(
        wraps=CriticAugment(
            planes={
                plane: {"flip_axes": list(PLANE_DIRECTIONS[plane]), "rotate_90": True}
                for plane in PLANES
            },
            prob=1.0,
        )
    )
    denoiser_before = _parameters(denoiser)
    critic_before = {axis: _parameters(critics[PLANES[axis]]) for axis in (0, 1, 2)}
    local_before = {
        axis: _parameters(critics[PLANES[axis]].local_output) for axis in (0, 1, 2)
    }

    r1_values = []

    def track_r1(scores, inputs):
        penalties = get_critic_r1(scores, inputs)
        r1_values.append(float(penalties.combine(optim.local_loss_weight).detach()))
        return penalties

    with patch("src.train.trainer.get_critic_r1", side_effect=track_r1):
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
    assert trainer.critic_augment.apply_together.call_count == 6
    assert math.isclose(metrics.r1, sum(r1_values) / len(r1_values), rel_tol=1e-6)
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
    assert all(
        _changed(critic_before[axis], critics[PLANES[axis]]) for axis in (0, 1, 2)
    )
    assert all(
        _changed(local_before[axis], critics[PLANES[axis]].local_output)
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


def test_anchor_training_uses_real_plane_and_updates_adapter() -> None:
    data = Config(
        domains={0: {0: ".", 1: ".", 2: "."}},
        crop_size=8,
        input_size=8,
        num_phases=3,
        batch_size=2,
    )
    model = Config(
        base_channels=4,
        channel_multipliers=(1, 2),
        embedding_channels=8,
        latent_channels=4,
        critic_channels=(4, 8),
        gradient_checkpointing=False,
    )
    optim = Config(
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
        anchor=_anchor_config(
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
        streams={
            axis: _ConstantStream(phase_channels(images, data.num_phases))
            for axis in (0, 1, 2)
        },
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
            "compute_logits",
            wraps=denoiser.compute_logits,
        ) as compute_logits,
    ):
        metrics = trainer.step(0, transition=0)

    assert metrics.anchor_planes == 1
    assert metrics.anchor_conflict_rate == 0.0
    assert forward.call_count == 1
    assert compute_logits.call_count == 2
    assert all(
        call.kwargs["anchor_image"] is not None
        and call.kwargs["anchor_mask"] is not None
        for call in compute_logits.call_args_list
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
    original_sample_vf = trainer.sample_vf
    original_update_critics = trainer.update_critics
    measured = []

    def track_vfs(batches, selection, owned_axes):
        vf_batch_ids.extend(id(batches.images[axis]) for axis in (0, 1, 2))
        measured.append(selection.measured)
        return original_sample_vf(batches, selection, owned_axes)

    def track_critics(transition, fake, batches, step, domain, **kwargs):
        critic_batch_ids.extend(id(batches.images[axis]) for axis in (0, 1, 2))
        return original_update_critics(
            transition,
            fake,
            batches,
            step,
            domain,
            **kwargs,
        )

    with (
        patch.object(trainer, "sample_vf", side_effect=track_vfs),
        patch.object(trainer, "update_critics", side_effect=track_critics),
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
    assert len(forward.call_args_list) == 1
    assert len(compute_logits.call_args_list) == 2
    assert all(vf is vfs[0] for vf in vfs)
    assert vfs[0].shape == (1, 3)
    region = measured[0].regions[0]
    image = (measured[0].image.select(region.axis + 2, region.index) + 1) * 0.5
    expected_vf = compute_vf({region.axis: image}, num_phases=3)
    assert torch.allclose(vfs[0][0], expected_vf)
    assert vf_batch_ids == critic_batch_ids

    gradient = denoiser.vf_mlp[-1].weight.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all())
    assert float(gradient.abs().sum()) > 0.0


@pytest.mark.parametrize("anchored", (False, True))
def test_vf_conditions_follow_individual_fractional_crops(anchored):
    trainer, _, streams = _conditioning_trainer(anchored=anchored)
    trainer.volume_batch_size = 2
    fractions = torch.tensor([[0.8, 0.2, 0.0], [0.1, 0.3, 0.6]])
    for stream in streams.values():
        stream.images = fractions[:, :, None, None].expand(-1, -1, 4, 8)

    prepared = trainer.prepare_step(0, 0)

    expected = fractions[fractions[:, 0].argsort()]
    observed = prepared.target_vf[prepared.target_vf[:, 0].argsort()]
    torch.testing.assert_close(observed, expected)
    assert prepared.model_conditions["vf"] is prepared.target_vf
    if anchored:
        measured = prepared.selection.measured
        region = measured.regions[0]
        image = measured.image.select(region.axis + 2, region.index)
        crop = image[..., region.row : region.row + 4, region.col : region.col + 8]
        torch.testing.assert_close(prepared.target_vf, ((crop + 1) * 0.5).mean((2, 3)))


def test_replay_vf_uses_only_its_measured_root():
    trainer, _, streams = _conditioning_trainer(anchored=True)
    trainer.volume_batch_size = 2
    fractions = torch.tensor([0.2, 0.3, 0.5])
    measured = encode_anchors(
        [PlaneAnchor(fractions[:, None, None].expand(-1, 4, 4), 0, 2)],
        1,
        3,
        8,
        torch.device("cpu"),
        torch.float32,
    )
    prediction = torch.full((1, 3, 8, 8, 8), -1.0)
    prediction[:, 0] = 1.0
    trainer.anchor_bank.add(0, prediction, measured, torch.tensor([True]))
    trainer.use_multi_anchor_next = True
    for stream in streams.values():
        stream.images.zero_()
        stream.images[:, 1] = 1.0

    prepared = trainer.prepare_step(0, 0)

    assert prepared.selection.source == "multi"
    torch.testing.assert_close(prepared.target_vf, fractions.expand(2, -1))


def test_unanchored_vf_supports_more_volumes_than_real_crops():
    trainer, _, streams = _conditioning_trainer(anchored=False)
    trainer.volume_batch_size = 5
    fractions = torch.tensor([[0.8, 0.2, 0.0], [0.1, 0.3, 0.6]])
    for stream in streams.values():
        stream.images = fractions[:, :, None, None].expand(-1, -1, 8, 8)

    prepared = trainer.prepare_step(0, 0)

    assert prepared.target_vf.shape == (5, 3)
    for row in prepared.target_vf:
        assert any(torch.allclose(row, crop) for crop in fractions)
    torch.testing.assert_close(prepared.target_vf[:2], fractions)


def test_unanchored_vf_and_height_use_the_same_owned_crops():
    trainer, _, _ = _conditioning_trainer(anchored=False)
    trainer.volume_batch_size = 2
    trainer.height_data = {"crop_size": 8, "height_extents": {0: 32}}
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
    with patch.object(trainer, "get_batches", return_value=batches):
        prepared = trainer.prepare_step(0, 0)

    torch.testing.assert_close(prepared.target_vf, fractions)
    torch.testing.assert_close(
        prepared.model_conditions["height"],
        trainer.volume_height(batches.origins[1], 0, batches.extents[1]),
    )


def test_profile_keeps_priority_over_crop_vf():
    trainer, _, _ = _conditioning_trainer(anchored=False)
    trainer.profile_settings = {"enabled": True}
    profile = torch.tensor([[[0.2, 0.4], [0.3, 0.2], [0.5, 0.4]]])
    batches = RealBatch(
        images={axis: torch.full((2, 3, 8, 8), 1 / 3) for axis in (0, 1, 2)},
        profiles={1: profile},
    )
    with patch.object(trainer, "get_batches", return_value=batches):
        prepared = trainer.prepare_step(0, 0)
    torch.testing.assert_close(prepared.target_vf, profile.mean(-1))
    assert prepared.model_conditions["vf"] is prepared.target_vf


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
        patch("src.train.trainer.torch.randn", return_value=initial.clone()),
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
            "compute_logits",
            wraps=denoiser.compute_logits,
        ) as compute_logits:
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
        for call in compute_logits.call_args_list
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


def test_real_anchor_preserves_a_rectangular_observation() -> None:
    trainer, _, _ = _conditioning_trainer(anchored=True)
    batches = RealBatch({axis: torch.randint(0, 3, (2, 4, 8)) for axis in (0, 1, 2)})

    selection = trainer.sample_real_anchor(batches, volume_size=8)

    condition = selection.condition
    coords = condition.mask[0, 0].nonzero()
    spans = tuple(
        int(coords[:, dim].max() - coords[:, dim].min() + 1) for dim in range(3)
    )
    assert condition.image.shape == (1, 3, 8, 8, 8)
    assert sorted(spans) == [1, 4, 8]
    assert int(condition.mask.sum()) == 4 * 8


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
            "compute_logits",
            wraps=denoiser.compute_logits,
        ) as compute_logits,
    ):
        metrics = trainer.step(0, transition=0)

    assert all(stream.calls == 1 for stream in streams.values())
    assert not metrics.vf_active
    assert metrics.vf_loss == 0.0
    calls = [*forward.call_args_list, *compute_logits.call_args_list]
    assert len(forward.call_args_list) == 1
    assert len(compute_logits.call_args_list) == 2
    assert all(call.kwargs["vf"] is not None for call in calls)
    assert all(not bool(call.kwargs["vf_present"].any()) for call in calls)
    assert all(
        parameter.grad is None or not bool(parameter.grad.any())
        for parameter in denoiser.vf_mlp.parameters()
    )


def test_joint_cfg_dropout_uses_four_categorical_anchor_vf_states() -> None:
    trainer = object.__new__(Trainer)
    trainer.profile_settings = {"enabled": False}
    trainer.volume_batch_size = 4
    trainer.device = torch.device("cpu")
    trainer.cfg_drop_each_probability = 0.1

    with patch(
        "src.train.trainer.torch.rand",
        return_value=torch.tensor((0.05, 0.15, 0.25, 0.35)),
    ):
        presence = trainer.sample_condition_presence(has_anchor=True)

    assert presence.anchor.tolist() == [False, False, True, True]
    assert presence.vf.tolist() == [False, True, False, True]


def test_single_condition_dropout_matches_joint_marginal_visibility() -> None:
    trainer = object.__new__(Trainer)
    trainer.profile_settings = {"enabled": False}
    trainer.volume_batch_size = 2
    trainer.device = torch.device("cpu")
    trainer.cfg_drop_each_probability = 0.1

    with patch(
        "src.train.trainer.torch.rand",
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
    observed = encode_anchors(
        (PlaneAnchor(torch.zeros(8, 8, dtype=torch.uint8), axis=0, index=4),),
        batch_size=1,
        num_phases=3,
        volume_size=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert observed is not None
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
    prediction = prediction + 0.0 * next(trainer.denoiser.parameters()).sum()
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
        patch.object(trainer, "sample_vf", return_value=target),
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


def test_exception_inside_step_does_not_publish_partial_weights(tmp_path: Path) -> None:
    trainer = object.__new__(Trainer)
    trainer.profile_settings = {"enabled": False}
    trainer.device = torch.device("cpu")
    trainer.cfg = {"stage": "low_res", "data": {}}
    trainer.streams = {}
    trainer.ema_denoiser = nn.Linear(2, 2)
    trainer.critics = nn.ModuleDict(
        {PLANES[axis]: nn.Linear(2, 1) for axis in range(3)}
    )
    trainer.connectivity_critic = nn.Linear(2, 1)
    trainer.connectivity_weight = 0.0
    trainer.normal_transition_weight = 0.0
    trainer.real_transition_weight = 0.0
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


def test_fit_keeps_latest_weights_and_sparse_numbered_checkpoints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("src.train.run.loop.save_training", lambda *args: None)
    trainer = object.__new__(Trainer)
    trainer.profile_settings = {"enabled": False}
    trainer.device = torch.device("cpu")
    trainer.cfg = {"stage": "low_res", "data": {}}
    trainer.streams = {}
    trainer.ema_denoiser = nn.Linear(2, 2)
    trainer.critics = nn.ModuleDict(
        {PLANES[axis]: nn.Linear(2, 1) for axis in range(3)}
    )
    trainer.connectivity_critic = nn.Linear(2, 1)
    trainer.connectivity_weight = 0.0
    trainer.normal_transition_weight = 0.0
    trainer.real_transition_weight = 0.0
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
        )
    )

    weights = run_train(
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
    data = Config(
        domains={0: {axis: "." for axis in axes}},
        crop_size=crop_size,
        input_size=patch_size,
        num_phases=3,
        batch_size=2,
    )
    model = Config(
        base_channels=4,
        channel_multipliers=(1, 2),
        embedding_channels=8,
        latent_channels=4,
        critic_channels=(4, 8),
        gradient_checkpointing=False,
    )
    optim = Config(
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
        anchor=_anchor_config(
            train_prob=1.0 if anchored else 0.0,
            start_step=anchor_start_step,
            ramp_steps=anchor_ramp_steps,
        ),
        connectivity=_connectivity_config(
            weight=connectivity_weight,
            phase_transition_weight=normal_transition_weight,
        ),
        conditioning=_conditioning_config(
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
        axis: _ConstantStream(phase_channels((base + axis).remainder(3).long(), 3))
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
    cfg: Config,
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
            patch_size=cfg.data.lo_res_size,
            slice_pairs_per_axis=cfg.train.slice_pairs_per_plane,
            ema_decay=cfg.optim.ema_decay,
            r1_gamma=cfg.loss.r1_weight,
            r1_interval=cfg.loss.r1_every_steps,
            critic_local_weight=cfg.loss.critic_local_weight,
            anchor_training_probability=cfg.conditioning.anchor.probability,
            anchor_start_step=cfg.conditioning.anchor.start_step,
            anchor_ramp_steps=cfg.conditioning.anchor.ramp_steps,
            anchor_shared_axis_probability=cfg.conditioning.anchor.borrowed_plane_probability,
            anchor_pixel_loss_weight=cfg.loss.anchor_pixel_weight,
            connectivity_weight=cfg.loss.connectivity.adversarial_weight,
            normal_transition_weight=(cfg.loss.connectivity.normal_transition_weight),
            vf_loss_weight=cfg.loss.volume_fraction_weight,
            domain_dropout=1.0 - cfg.conditioning.domain_keep_probability,
            cfg_drop_each_probability=cfg.conditioning.dropout_probability_per_case,
            latent_channels=cfg.model.generator.latent_channels,
            amp_enabled=use_amp,
            connectivity_ramp_steps=0,
            connectivity_windows_per_plane=1,
        ),
    )


def _config(
    data: Config,
    model: Config,
    optim: Config,
    *,
    anchor: Config | None = None,
    connectivity: Config | None = None,
    vf: Config | None = None,
    conditioning: Config | None = None,
) -> Config:
    anchor = _anchor_config() if anchor is None else anchor
    anchor["connectivity"] = (
        _connectivity_config() if connectivity is None else connectivity
    )
    conditioning = _conditioning_config() if conditioning is None else conditioning
    connectivity = anchor["connectivity"]
    vf = Config(weight=1.0) if vf is None else vf
    return Config(
        data=Config(
            domains={
                domain: {PLANES[axis]: paths for axis, paths in planes.items()}
                for domain, planes in data.domains.items()
            },
            num_phases=data.num_phases,
            crop_size=data.crop_size,
            lo_res_size=data.input_size,
        ),
        model=Config(
            gradient_checkpointing=model.gradient_checkpointing,
            generator=Config(
                channels=[
                    model.base_channels * multiplier
                    for multiplier in model.channel_multipliers
                ],
                embedding_channels=model.embedding_channels,
                latent_channels=model.latent_channels,
                anchor_multiscale_input=anchor.multiscale_input,
            ),
            critic=Config(
                channels=model.critic_channels,
                plane_groups=[
                    [plane]
                    for axis, plane in enumerate(PLANES)
                    if any(axis in planes for planes in data.domains.values())
                ],
            ),
            diffusion=Config(num_steps=2, beta_min=0.1, beta_max=2.0),
        ),
        conditioning=Config(
            anchor=Config(
                probability=anchor.train_prob,
                start_step=anchor.start_step,
                ramp_steps=anchor.ramp_steps,
                borrowed_plane_probability=anchor.cross_domain_prob,
            ),
            domain_keep_probability=1.0 - data.get("domain_dropout", 0),
            dropout_probability_per_case=conditioning.joint_each_prob,
        ),
        loss=Config(
            critic_local_weight=optim.local_loss_weight,
            r1_weight=optim.r1_gamma,
            r1_every_steps=optim.r1_interval,
            anchor_pixel_weight=anchor.pixel_weight,
            volume_fraction_weight=vf.weight,
            connectivity=Config(
                adversarial_weight=connectivity.weight,
                normal_transition_weight=connectivity.phase_transition_weight,
            ),
        ),
        optim=Config(
            generator_lr=optim.denoiser_lr,
            critic_lr=optim.critic_lr,
            adam_betas=[optim.beta1, optim.beta2],
            ema_decay=0.9,
        ),
        train=Config(
            initial_weights=None,
            total_steps=1,
            volume_batch_size=1,
            slice_pairs_per_plane=2,
            mixed_precision=False,
            weights_every_steps=1,
            archive_every_steps=1,
            real_batch_size=data.batch_size,
            num_workers=0,
        ),
    )


@pytest.mark.parametrize("size,count", [(8, 4), (16, 8)])
def test_replay_keeps_measurement_and_plane_density(size, count):
    bank = AnchorBank(capacity=1, plane_spacing=2)
    image = torch.stack((torch.full((size, size), 0.2), torch.full((size, size), 0.8)))
    measured = encode_anchors(
        [PlaneAnchor(image, 0, 2)], 1, 2, size, torch.device("cpu"), torch.float32
    )
    prediction = torch.zeros(1, 2, size, size, size)
    bank.add(0, prediction, measured, torch.tensor([True]))
    replay = bank.sample(0, 2, torch.device("cpu"))
    condition, target, reference = replay.condition, replay.measured, replay.reference
    assert condition.planes == count
    torch.testing.assert_close(
        condition.image[:, :, 2], measured.image[:, :, 2].expand(2, -1, -1, -1)
    )
    torch.testing.assert_close(reference, prediction.expand(2, -1, -1, -1, -1))
    assert not torch.equal(reference[:, :, 2], target.image[:, :, 2])
    assert target.regions == measured.regions
    assert bank.sample(1, 1, torch.device("cpu")) is None


def test_replay_continuity_reference_excludes_pasted_measurement_jump():
    from src.data.slice import AnchorTripletSampler
    from src.train.loss.connectivity import compute_transition_loss

    size = 5
    prediction = torch.empty(1, 2, size, size, size)
    prediction[:, 0], prediction[:, 1] = -0.6, 0.6
    image = torch.stack((torch.full((size, size), 0.8), torch.full((size, size), 0.2)))
    measured = encode_anchors(
        [PlaneAnchor(image, 0, 2)], 1, 2, size, torch.device("cpu"), torch.float32
    )
    bank = AnchorBank(capacity=1)
    bank.add(0, prediction, measured, torch.tensor([True]))
    replay = bank.sample(0, 1, torch.device("cpu"))
    condition, target, reference = replay.condition, replay.measured, replay.reference
    torch.testing.assert_close(reference, prediction)
    torch.testing.assert_close(condition.image[:, :, 2], measured.image[:, :, 2])
    sampler = AnchorTripletSampler(max_gap=1, windows_per_plane=1)
    adapted = -prediction
    real, fake = sampler.sample(adapted, reference, target)
    assert compute_transition_loss(real, fake) == 0
    pasted = torch.where(target.mask, target.image, reference)
    real, fake = sampler.sample(pasted, reference, target)
    assert compute_transition_loss(real, fake) > 0


def test_generator_input_diagnostics_preserve_training_updates_and_rng():
    trainer, _, _ = _conditioning_trainer(anchored=False, axes=(0,))
    trainer.r1_interval = 1
    control = copy.deepcopy(trainer)
    control.r1_interval = 3  # R1/R2 weights are zero; only diagnostics differ.
    with torch.random.fork_rng(devices=[]):
        rng = torch.get_rng_state()
        actual = trainer.step(0, transition=0)
        after = torch.get_rng_state()
        torch.set_rng_state(rng)
        expected = control.step(0, transition=0)
        assert torch.equal(after, torch.get_rng_state())
    prefix = "generator_input_gradient/xy/0/t0/"
    assert actual.diagnostics[prefix + "previous"] > 0
    assert actual.diagnostics[prefix + "current"] > 0
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
    trainer.r1_interval = 3
    metrics = trainer.step(1, transition=1)
    assert not any(
        key.startswith("generator_input_gradient/") for key in metrics.diagnostics
    )
