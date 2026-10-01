from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from src.build.trainer import build_trainer
from src.config.train import load_train_config
from src.prepare.resize import phase_channels, resize_phases
from src.train.batch import DenoiserBatch
from src.train.loss.denoiser import DenoiserLossSettings, denoiser_objective
from src.train.loss.group_statistics import compute_group_statistics_loss
from src.train.loss.transition import compute_real_transition_loss
from src.train.state import (
    resume_sr_training,
    resume_training,
    save_sr_training,
    save_training,
)

ALL = {"xy_xz_yz": (0, 1, 2)}
SIDES = {"xy": (0,), "xz_yz": (1, 2)}


def lamella(axis, size=8):
    shape = [1, 1, 1, 1]
    shape[axis + 1] = size
    labels = torch.arange(size).reshape(shape).expand(1, size, size, size) // 2 % 2
    return phase_channels(labels, 2)


@pytest.mark.parametrize("axis", range(3))
def test_lamella_is_penalized_only_by_equivalent_normal_axes(axis):
    probs = lamella(axis)
    assert compute_group_statistics_loss(probs, ALL, 3, 0)[0] > 0
    sides = compute_group_statistics_loss(probs, SIDES, 3, 0)[0]
    assert (sides == 0) if axis == 0 else (sides > 0)
    singleton, logs = compute_group_statistics_loss(
        probs, {"xy": (0,), "xz": (1,), "yz": (2,)}, 3, 0
    )
    assert singleton == 0 and logs == {}


def test_opposite_sample_anisotropy_cannot_cancel_in_a_batch():
    probs = torch.cat((lamella(1), lamella(2)))
    assert (
        compute_group_statistics_loss(probs.mean(0, keepdim=True), SIDES, 3, 0)[0] == 0
    )
    separate = [compute_group_statistics_loss(p[None], SIDES, 3, 0)[0] for p in probs]
    loss, _ = compute_group_statistics_loss(probs, SIDES, 3, 0)
    torch.testing.assert_close(loss, torch.stack(separate).mean())
    assert loss > 0


def test_height_rows_cannot_cancel_and_z_structure_is_preserved():
    probs = lamella(1)
    probs[:, :, 4:] = lamella(2)[:, :, 4:]
    assert compute_group_statistics_loss(probs, SIDES, 3, 0)[0] == 0
    assert compute_group_statistics_loss(probs, ALL, 3, 0, preserve_height=True)[0] > 0
    assert (
        compute_group_statistics_loss(lamella(0), ALL, 3, 0, preserve_height=True)[0]
        == 0
    )


def test_isotropic_geometry_and_axis_reversal_have_the_expected_losses():
    z, y, x = torch.meshgrid(*(torch.arange(8) - 3.5 for _ in range(3)), indexing="ij")
    sphere = phase_channels(((z * z + y * y + x * x) < 9)[None].long(), 2)
    assert compute_group_statistics_loss(sphere, ALL, 3, 0)[0] == 0
    probs = lamella(2)[:, :, :5, :6, :7]
    original = compute_group_statistics_loss(probs, ALL, 8, 0)[0]
    flipped = compute_group_statistics_loss(probs.flip((-1, -2, -3)), ALL, 8, 0)[0]
    torch.testing.assert_close(original, flipped)


def test_tolerance_and_inactive_anchors_have_zero_gradient():
    probs = lamella(2).requires_grad_()
    for tolerance, active in [(1, None), (0, torch.tensor([False]))]:
        loss, logs = compute_group_statistics_loss(
            probs, ALL, 3, tolerance, active=active
        )
        assert loss == 0
        loss.backward()
        assert torch.equal(probs.grad, torch.zeros_like(probs))
        probs.grad = None
        assert all(not value.requires_grad for value in logs.values())
    active = torch.tensor([True, False])
    batch = torch.cat((lamella(0), probs.detach())).requires_grad_()
    loss, _ = compute_group_statistics_loss(batch, ALL, 3, 0, active=active)
    expected, _ = compute_group_statistics_loss(batch[:1], ALL, 3, 0)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert batch.grad[0].abs().sum() > 0 and batch.grad[1].abs().sum() == 0


def test_sr_coarse_anisotropy_is_an_allowance_and_never_receives_gradients():
    coarse = lamella(2, 4).requires_grad_()
    probs = resize_phases(coarse.detach(), (8, 8, 8)).requires_grad_()
    assert compute_group_statistics_loss(probs, SIDES, 3, 0)[0] > 0
    loss, _ = compute_group_statistics_loss(probs, SIDES, 3, 0, coarse=coarse)
    assert loss == 0
    loss.backward()
    assert coarse.grad is None
    isotropic = torch.full_like(coarse, 0.5)
    assert compute_group_statistics_loss(probs, SIDES, 3, 0, coarse=isotropic)[0] > 0


def test_prior_has_finite_gradients_and_can_reduce_directional_difference():
    logits = torch.nn.Parameter((lamella(2) * 0.8 + 0.1).log())
    optimizer = torch.optim.Adam([logits], lr=0.1)
    before = compute_group_statistics_loss(logits.softmax(1), ALL, 3, 0.01)[0].item()
    for _ in range(20):
        optimizer.zero_grad()
        loss, _ = compute_group_statistics_loss(logits.softmax(1), ALL, 3, 0.01)
        if loss.item() == 0:
            break
        loss.backward()
        assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
        optimizer.step()
    assert loss.item() < before * 0.5
    probs = logits.softmax(1).detach()
    torch.testing.assert_close(probs.mean((0, 2, 3, 4)), torch.tensor([0.5, 0.5]))


def test_equal_axis_pairs_do_not_prove_full_rotational_isotropy():
    z, y, x = torch.meshgrid(*(torch.arange(8) for _ in range(3)), indexing="ij")
    labels = ((z + y + x) % 8 < 4).long()
    probs = phase_channels(labels[None], 2)
    assert compute_group_statistics_loss(probs, ALL, 3, 0)[0] == 0
    a = labels[:, :-1, :-1] == 1
    diagonal = (a & (labels[:, 1:, 1:] == 1)).float().mean()
    opposite = ((labels[:, 1:, :-1] == 1) & (labels[:, :-1, 1:] == 1)).float().mean()
    assert diagonal != opposite


def test_symmetry_alone_accepts_uniform_fractions_but_real_statistics_reject_them():
    observed = lamella(2)[:, :, 0]
    uniform = torch.full((1, 2, 8, 8, 8), 0.5)
    assert compute_group_statistics_loss(uniform, ALL, 3, 0)[0] == 0
    loss, _ = compute_real_transition_loss({0: observed}, {0: uniform[:, :, 0]}, 3)
    assert loss > 0


def test_pair_gradients_do_not_retain_volume_copies_for_each_gap():
    probs = torch.rand(1, 3, 8, 8, 8).softmax(1).requires_grad_()
    saved = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        saved[storage.data_ptr()] = storage.nbytes()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss, _ = compute_group_statistics_loss(probs, ALL, 6, 0)
    loss.backward()
    assert sum(saved.values()) < probs.untyped_storage().nbytes() * 2


def objective_batch():
    probs = lamella(2).requires_grad_()
    batch = DenoiserBatch(
        transition=0,
        critic_domains={},
        fake={
            axis: (torch.empty(0, 2, 8, 8), torch.empty(0, 2, 8, 8))
            for axis in range(3)
        },
        logits=probs,
        clean_probs=probs,
        anchor=None,
        anchor_observed_mask=None,
        anchor_observed_axis_masks=None,
        anchor_present=torch.tensor([False]),
        anchor_ramp=1,
        target_vf=torch.tensor([[0.5, 0.5]]),
        vf_present=torch.tensor([True]),
        observed_axes=(0, 1, 2),
        group_statistics_ramp=0.5,
    )
    settings = DenoiserLossSettings(
        local_weight=0,
        vf_weight=0,
        real_transition_weight=0,
        profile_bins=4,
        profile_weight=0,
        profile_gradient_weight=0,
        consistency_weight=0,
        consistency_tolerance=0,
        num_phases=2,
        group_statistics_weight=0.2,
        group_statistics_max_gap=3,
        group_statistics_tolerance=0,
    )
    return batch, settings


def test_objective_adds_the_ramped_prior_and_preserves_the_real_loss():
    batch, settings = objective_batch()
    real_loss = batch.clean_probs.sum() * 0 + 0.7
    batch = replace(batch, real_transition_loss=real_loss, connectivity_ramp=0.25)
    settings = replace(settings, real_transition_weight=0.4)
    update, logs = denoiser_objective(batch, {}, ALL, None, settings)
    prior, _ = compute_group_statistics_loss(batch.clean_probs, ALL, 3, 0)
    torch.testing.assert_close(update.total, 0.1 * prior + 0.1 * real_loss)
    assert logs["loss/group_statistics"] > 0
    update.total.backward()
    assert batch.clean_probs.grad.abs().sum() > 0


@pytest.mark.parametrize("change", [{"transition": 1}, {"group_statistics_ramp": 0}])
def test_objective_skips_nonfinal_transitions_and_inactive_schedule(change):
    batch, settings = objective_batch()
    batch = replace(batch, **change)
    update, logs = denoiser_objective(batch, {}, ALL, None, settings)
    assert update.total == 0 and "loss/group_statistics" not in logs


def test_objective_excludes_borrowed_domains_and_active_anchor_cases():
    batch, settings = objective_batch()
    update, _ = denoiser_objective(
        replace(batch, observed_axes=(2,)), {}, ALL, None, settings
    )
    assert update.total == 0
    batch = replace(batch, anchor=object(), anchor_present=torch.tensor([True]))

    def anchor_loss(*args):
        zero = batch.clean_probs.sum() * 0
        return SimpleNamespace(total=zero, coarse=zero, pixel=zero, accuracy=zero)

    update, logs = denoiser_objective(batch, {}, ALL, anchor_loss, settings)
    assert update.total == 0 and logs["loss/group_statistics"] == 0
    assert logs["group_statistics/active_fraction"] == 0


def small_trainer(tmp_path, stage, start=0, ramp=0):
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    Image.new("L", (8, 8)).save(images / "sample.png")
    cfg = load_train_config(f"tests/fixtures/config/train/{stage}.yaml", stage)
    cfg["data"].update(
        domains={0: {plane: [str(images)] for plane in ("xy", "xz", "yz")}},
        crop_size=8,
        lo_res_size=8,
        hi_res_size=12,
    )
    cfg["model"]["generator"].update(
        channels=[4, 8], embedding_channels=8, latent_channels=4
    )
    cfg["model"]["critic"]["channels"] = [4, 8]
    cfg["model"]["gradient_checkpointing"] = False
    cfg["model"]["diffusion"]["num_steps"] = 2
    cfg["augmentation"]["probability"] = 0
    cfg["loss"]["group_statistics"].update(
        weight=0.2, max_gap=2, tolerance=0, start_step=start, ramp_steps=ramp
    )
    cfg["train"].update(
        total_steps=10,
        num_workers=0,
        volume_batch_size=1,
        real_batch_size=2,
        slice_pairs_per_plane=2,
        mixed_precision=False,
    )
    if stage == "low_res":
        cfg["conditioning"]["anchor"]["probability"] = 0
    bank = None if stage == "low_res" else {0: torch.rand(2, 2, 8, 8, 8).softmax(1)}
    return build_trainer(cfg, torch.device("cpu"), bank=bank), cfg, bank


@pytest.mark.parametrize("stage", ["low_res", "sr"])
def test_trainer_schedule_is_independent_of_real_transitions_and_roundtrips_checkpoints(
    tmp_path, stage
):
    torch.set_num_threads(1)
    trainer, cfg, bank = small_trainer(tmp_path, stage)
    metrics = trainer.step(0, transition=0)
    assert "loss/group_statistics" in metrics.diagnostics
    assert metrics.diagnostics["group_statistics/ramp"] == 1
    assert metrics.diagnostics["group_statistics/active_fraction"] == 1
    assert torch.isfinite(torch.tensor(metrics.generator_total))
    path = tmp_path / "training.pt"
    if stage == "low_res":
        save_training(path, trainer)
    else:
        save_sr_training(trainer, path)
    payload = torch.load(path, weights_only=True)
    restored = build_trainer(cfg, torch.device("cpu"), bank=bank)
    resume = resume_training if stage == "low_res" else resume_sr_training
    resume(restored, payload)
    assert restored.completed_steps == 1
    assert restored.group_statistics_weight == 0.2
    restored.cfg["loss"]["group_statistics"]["weight"] = 0
    with pytest.raises(ValueError, match="saved"):
        resume(restored, payload)


@pytest.mark.parametrize("stage", ["low_res", "sr"])
def test_trainer_respects_group_statistics_start_and_ramp(tmp_path, stage):
    trainer, _, _ = small_trainer(tmp_path, stage, start=2, ramp=4)
    for step, expected in ((0, 0), (2, 0.25), (3, 0.5), (5, 1)):
        prepared = trainer.prepare_step(step, transition=0)
        assert prepared.group_statistics_ramp == expected
        assert prepared.domain == 0
