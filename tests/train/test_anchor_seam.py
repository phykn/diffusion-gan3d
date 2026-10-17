import copy

import numpy as np
import pytest
import torch
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.anchor import PlaneAnchor, encode_anchors
from src.build.model import build_seam_critic
from src.build.trainer import build_trainer
from src.config.train import load_train_config
from src.data.slice import sample_seam_bands
from src.model.critic import SeamCritic2D
from src.model.layers import NULL_DOMAIN
from src.train.batch import AnchorSelection
from src.train.run.loop import run_train, save_weights
from src.train.state import resume_training, save_training
from src.train.trainer import Trainer


def coordinates(size=8, batch=1):
    grid = torch.stack(
        torch.meshgrid(*(torch.arange(size).float(),) * 3, indexing="ij")
    )
    return grid.expand(batch, -1, -1, -1, -1).clone()


def condition(anchors, size=8, batch=1):
    return encode_anchors(
        [
            PlaneAnchor(
                torch.zeros(batch, *shape, dtype=torch.long), axis, index, position
            )
            for axis, index, shape, position in anchors
        ],
        batch,
        3,
        size,
        torch.device("cpu"),
        torch.float32,
    )


def test_near_bands_end_one_row_before_the_anchor():
    volume = coordinates().requires_grad_()
    far, near = sample_seam_bands(
        volume, condition([(0, 5, (8, 8), None)]), 2, torch.Generator().manual_seed(0)
    )
    assert near.values.shape == (16, 3, 2, 8)
    assert near.axes.tolist() == [0] * 16
    assert near.values[:, 0].amin((0, 2)).tolist() == [3.0, 4.0]
    assert near.values[:, 0].amax((0, 2)).tolist() == [3.0, 4.0]
    assert near.values.requires_grad
    assert len(far) == len(near) and not far.values.requires_grad
    far_rows = {tuple(rows) for rows in far.values[:, 0, :, 0].tolist()}
    assert far_rows <= {(0.0, 1.0), (1.0, 2.0)}


def test_bands_with_fixed_voxels_are_discarded():
    anchors = condition([(0, 5, (8, 8), None), (2, 1, (8, 8), None)])
    far, near = sample_seam_bands(coordinates(), anchors, 2)
    seam = near.axes == 0
    assert int(seam.sum()) == 7
    assert set(near.values[seam, 2].unique().tolist()) == {0, 2, 3, 4, 5, 6, 7}
    for bands in (near, far):
        for axis in (0, 2):
            values = bands.values[bands.axes == axis]
            assert not (values[:, axis] == anchors.regions[axis // 2].index).any()
            assert not (
                values[:, 2 - axis] == anchors.regions[1 - axis // 2].index
            ).any()


def test_far_bands_keep_more_than_n_rows_from_every_same_axis_anchor():
    anchors = condition([(0, 2, (12, 12), None), (0, 9, (12, 12), None)], size=12)
    far, near = sample_seam_bands(coordinates(12), anchors, 2)
    assert len(near) == 48 and len(far) == 24
    assert far.values[:, 0].amin((0, 2)).tolist() == [5.0, 6.0]
    assert far.values[:, 0].amax((0, 2)).tolist() == [5.0, 6.0]
    near_rows = {tuple(row) for row in near.values[:, 0, :, 0].tolist()}
    assert near_rows == {(0.0, 1.0), (7.0, 8.0)}


def test_partial_anchor_selects_only_crossing_sections():
    anchors = condition([(1, 4, (2, 3), (1, 5))])
    _, near = sample_seam_bands(coordinates(), anchors, 1)
    assert len(near) == 5
    assert near.values[:, 1].unique().tolist() == [3.0]
    z = near.values[:, 0, 0]
    x = near.values[:, 2, 0]
    across_z = (z == z[:, :1]).all(1)
    assert set(z[across_z, 0].tolist()) == {1.0, 2.0}
    assert set(x[~across_z, 0].tolist()) == {5.0, 6.0, 7.0}


def test_anchor_without_enough_rows_or_inactive_batch_has_no_bands():
    volume = coordinates(batch=2)
    far, near = sample_seam_bands(volume, condition([(0, 1, (8, 8), None)], batch=2), 2)
    assert len(far) == len(near) == 0
    assert near.values.shape == (0, 3, 2, 8)
    anchors = condition([(0, 5, (8, 8), None)], batch=2)
    _, both = sample_seam_bands(volume, anchors, 2)
    anchors = copy.copy(anchors)
    object.__setattr__(anchors, "active_batches", (1,))
    _, one = sample_seam_bands(volume, anchors, 2)
    assert len(both) == 2 * len(one)
    _, none = sample_seam_bands(volume, None, 2)
    assert len(none) == 0


def test_axis_without_far_bands_contributes_no_near_bands():
    anchors = condition([(0, 6, (8, 8), None), (1, 4, (8, 8), None)])
    far, near = sample_seam_bands(coordinates(), anchors, 4)
    assert len(far) == len(near) == 0
    far, near = sample_seam_bands(coordinates(), anchors, 2)
    assert near.axes.unique().tolist() == far.axes.unique().tolist() == [0, 1]


@pytest.mark.parametrize("rows", (0, -1, 1.5, True))
def test_band_rows_must_be_a_positive_integer(rows):
    with pytest.raises(ValueError, match="rows"):
        sample_seam_bands(coordinates(), None, rows)


def test_seam_critic_scores_bands_by_axis_and_domain():
    torch.manual_seed(0)
    critic = SeamCritic2D(3, (4, 8), 8, 2)
    bands = torch.randn(3, 3, 4, 8)
    scores = critic(bands, torch.tensor([0, 1, 2]), torch.tensor([0, 1, NULL_DOMAIN]))
    assert scores.logits_global.shape == (3,) and scores.logits_local is None
    moved = critic(bands, torch.tensor([1, 1, 2]), torch.tensor([0, 1, NULL_DOMAIN]))
    assert moved.logits_global[0] != scores.logits_global[0]
    torch.testing.assert_close(moved.logits_global[1:], scores.logits_global[1:])
    with pytest.raises(ValueError, match="bands"):
        critic(bands[0], torch.tensor([0]), torch.tensor([0]))
    with pytest.raises(ValueError, match="axes"):
        critic(bands, torch.tensor([0]), torch.tensor([0, 0, 0]))


def test_axis_domain_helper_maps_any_axis_tensor():
    domains = Trainer.get_axis_domains({0: 1, 2: 0}, torch.tensor([2, 1, 0, 0]))
    assert domains.tolist() == [0, NULL_DOMAIN, 1, 1]


def small_config(tmp_path, weight=0.25):
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    for index in range(4):
        labels = ((np.indices((12, 12)).sum(0) + index) % 3).astype(np.uint8)
        Image.fromarray(labels).save(images / f"{index}.png")
    cfg = load_train_config("tests/fixtures/config/train/low_res.yaml")
    cfg["data"].update(
        domains={0: {p: [str(images)] for p in ("xy", "xz", "yz")}},
        num_phases=3,
        crop_size=8,
        lo_res_size=8,
    )
    cfg["model"]["generator"].update(
        channels=[4, 8], embedding_channels=8, latent_channels=4
    )
    cfg["model"]["critic"].update(channels=[4, 8])
    cfg["model"]["gradient_checkpointing"] = False
    cfg["model"]["diffusion"].update(num_steps=2)
    cfg["conditioning"]["anchor"].update(probability=1.0, ramp_steps=0)
    cfg["loss"].update(
        r1_weight=0.01, r1_every_steps=2, anchor_seam_weight=weight, anchor_seam_rows=1
    )
    cfg["train"].update(
        total_steps=4, real_batch_size=2, slice_pairs_per_plane=2, mixed_precision=False
    )
    return cfg


def fixed_anchor(trainer, axis=0, index=4):
    def sample_anchor(*args, **kwargs):
        anchor = condition(
            [(axis, index, (8, 8), None)], batch=trainer.settings.volume_batch_size
        )
        return AnchorSelection(
            condition=anchor,
            observed_mask=anchor.mask,
            observed_axis_masks=anchor.axis_masks,
            source="real",
            measured=anchor,
        )

    trainer.conditions.sample_anchor = sample_anchor


def test_lr_trainer_builds_the_seam_critic_even_when_disabled(tmp_path):
    cfg = small_config(tmp_path, weight=0.0)
    trainer = build_trainer(cfg, torch.device("cpu"))
    assert isinstance(trainer.seam_critic, SeamCritic2D)
    assert trainer.updates["seam"] == 0
    before = copy.deepcopy(trainer.seam_critic.state_dict())
    fixed_anchor(trainer)
    metrics = trainer.step(0, transition=0)
    assert trainer.updates["seam"] == 0
    assert not any("anchor_seam" in key for key in metrics.diagnostics)
    for name, value in trainer.seam_critic.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    sr = load_train_config("tests/fixtures/config/train/sr.yaml", "sr")
    assert build_seam_critic(sr) is None


def test_seam_critic_and_generator_loss_train_from_the_selected_transition(
    tmp_path, monkeypatch
):
    torch.manual_seed(0)
    trainer = build_trainer(small_config(tmp_path), torch.device("cpu"))
    fixed_anchor(trainer)
    before = copy.deepcopy(trainer.seam_critic.state_dict())
    sampled = []

    def record(volume, *args):
        sampled.append(volume)
        return sample_seam_bands(volume, *args)

    monkeypatch.setattr("src.train.trainer.sample_seam_bands", record)
    metrics = trainer.step(0, transition=1)
    assert sampled and sampled[0].requires_grad
    diagnostics = metrics.diagnostics
    assert trainer.updates["seam"] == 1
    assert diagnostics["anchor_seam/near_bands"] == 16
    assert diagnostics["anchor_seam/far_bands"] == 16
    for key in (
        "loss/critic_anchor_seam",
        "loss/anchor_seam",
        "anchor_seam/near_score",
        "anchor_seam/far_score",
        "anchor_seam/contribution",
    ):
        assert np.isfinite(diagnostics[key])
    assert diagnostics["anchor_seam/contribution"] == pytest.approx(
        0.25 * diagnostics["loss/anchor_seam"]
    )
    assert any(
        not torch.equal(value, before[name])
        for name, value in trainer.seam_critic.state_dict().items()
    )


def test_seam_loss_reaches_the_generator(tmp_path):
    torch.manual_seed(0)
    trainer = build_trainer(small_config(tmp_path), torch.device("cpu"))
    prediction = torch.randn(1, 3, 8, 8, 8, requires_grad=True)
    anchor = condition([(0, 4, (8, 8), None)])
    _, fake = trainer.make_seam_bands(prediction, anchor, torch.tensor([True]))
    domains = trainer.get_axis_domains({0: 0}, fake.axes)
    scores = trainer.seam_critic(fake.values, fake.axes, domains)
    scores.logits_global.sum().backward()
    assert prediction.grad[:, :, 3].abs().sum() > 0
    assert prediction.grad[:, :, :3].abs().sum() == 0
    assert prediction.grad[:, :, 4:].abs().sum() == 0
    _, hidden = trainer.make_seam_bands(prediction, anchor, torch.tensor([False]))
    assert len(hidden) == 0


def test_tensorboard_records_seam_terms(tmp_path):
    torch.manual_seed(0)
    trainer = build_trainer(small_config(tmp_path), torch.device("cpu"))
    fixed_anchor(trainer)
    run_dir = tmp_path / "run"
    run_train(trainer, steps=1, save_every=1, run_dir=run_dir)
    tags = set(
        EventAccumulator(str(run_dir / "tensorboard")).Reload().Tags()["scalars"]
    )
    assert {
        "loss/critic_anchor_seam",
        "loss/anchor_seam",
        "anchor_seam/near_score",
        "anchor_seam/far_score",
        "anchor_seam/contribution",
        "anchor_seam/near_bands",
        "anchor_seam/far_bands",
    } <= tags
    assert (run_dir / "critic_s.pt").is_file()


def test_weights_export_and_initial_weights_include_an_optional_seam_critic(tmp_path):
    cfg = small_config(tmp_path)
    trainer = build_trainer(cfg, torch.device("cpu"))
    with torch.no_grad():
        for parameter in trainer.seam_critic.parameters():
            parameter.fill_(0.5)
    weights = tmp_path / "weights"
    save_weights(trainer, weights)
    assert (weights / "critic_s.pt").is_file()
    cfg["train"]["initial_weights"] = str(weights)
    loaded = build_trainer(copy.deepcopy(cfg), torch.device("cpu"))
    for parameter in loaded.seam_critic.parameters():
        assert (parameter == 0.5).all()
    (weights / "critic_s.pt").unlink()
    fresh = build_trainer(copy.deepcopy(cfg), torch.device("cpu"))
    assert not all((p == 0.5).all() for p in fresh.seam_critic.parameters())


def test_checkpoint_restores_the_seam_critic(tmp_path):
    torch.manual_seed(0)
    cfg = small_config(tmp_path)
    trainer = build_trainer(cfg, torch.device("cpu"))
    fixed_anchor(trainer)
    trainer.step(0, transition=0)
    path = tmp_path / "training.pt"
    save_training(path, trainer)
    payload = torch.load(path, weights_only=True)
    assert {"seam", "seam_optim"} <= payload.keys()
    restored = build_trainer(cfg, torch.device("cpu"))
    resume_training(restored, payload)
    assert restored.updates == trainer.updates
    torch.testing.assert_close(
        restored.seam_critic.state_dict(), trainer.seam_critic.state_dict()
    )
    torch.testing.assert_close(
        restored.seam_optim.state_dict(), trainer.seam_optim.state_dict()
    )


def test_checkpoint_from_before_the_seam_critic_starts_a_new_one(tmp_path):
    cfg = small_config(tmp_path)
    for key in ("anchor_seam_weight", "anchor_seam_rows"):
        cfg["loss"].pop(key)
    trainer = build_trainer(cfg, torch.device("cpu"))
    path = tmp_path / "training.pt"
    save_training(path, trainer)
    payload = torch.load(path, weights_only=True)
    payload.pop("seam")
    payload.pop("seam_optim")
    payload["updates"].pop("seam")
    for key in ("anchor_seam_weight", "anchor_seam_rows"):
        payload["config"]["loss"].pop(key)
    torch.manual_seed(1)
    restored = build_trainer(cfg, torch.device("cpu"))
    initial = copy.deepcopy(restored.seam_critic.state_dict())
    resume_training(restored, payload)
    assert restored.updates["seam"] == 0
    for name, value in restored.seam_critic.state_dict().items():
        torch.testing.assert_close(value, initial[name], rtol=0, atol=0)


@pytest.mark.parametrize("missing", ["seam", "seam_optim", "counter"])
def test_resume_rejects_incomplete_seam_state(tmp_path, missing):
    cfg = small_config(tmp_path)
    trainer = build_trainer(cfg, torch.device("cpu"))
    path = tmp_path / "training.pt"
    save_training(path, trainer)
    payload = torch.load(path, weights_only=True)
    if missing == "counter":
        payload["updates"].pop("seam")
    else:
        payload.pop(missing)
    with pytest.raises(ValueError, match="Seam training state is incomplete"):
        resume_training(build_trainer(cfg, torch.device("cpu")), payload)


def test_active_seam_requires_saved_seam_state(tmp_path):
    cfg = small_config(tmp_path)
    trainer = build_trainer(cfg, torch.device("cpu"))
    path = tmp_path / "training.pt"
    save_training(path, trainer)
    payload = torch.load(path, weights_only=True)
    payload.pop("seam")
    payload.pop("seam_optim")
    payload["updates"].pop("seam")
    with pytest.raises(ValueError, match="Seam training state is missing"):
        resume_training(build_trainer(cfg, torch.device("cpu")), payload)
