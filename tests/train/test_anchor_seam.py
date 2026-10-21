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


def bands(volume, anchors, rows=2, width=4, count=1000, seed=0, directed=None):
    return sample_seam_bands(
        volume,
        anchors,
        rows,
        width,
        count,
        directed,
        torch.Generator().manual_seed(seed),
    )


def section_axes(values):
    flat = values.flatten(2)
    return (flat == flat[:, :, :1]).all(2).float().argmax(1)


def section_rows(values):
    rows = values[:, :, 1, 0] - values[:, :, 0, 0]
    return rows.abs().argmax(1)


def test_near_bands_start_next_to_the_anchor_on_both_sides():
    volume = coordinates().requires_grad_()
    far, near = bands(volume, condition([(0, 5, (8, 8), None)]))
    assert near.values.shape == (160, 3, 2, 4)
    near_rows = [tuple(rows) for rows in near.values[:, 0, :, 0].tolist()]
    assert near_rows.count((4.0, 3.0)) == near_rows.count((6.0, 7.0)) == 80
    assert (near.values[:, 0] == near.values[:, 0, :, :1]).all()
    assert near.values.requires_grad
    assert not far.values.requires_grad
    far_rows = {tuple(rows) for rows in far.values[:, 0, :, 0].tolist()}
    assert far_rows <= {(0.0, 1.0), (1.0, 2.0), (1.0, 0.0), (2.0, 1.0)}
    assert {rows[0] < rows[1] for rows in far_rows} == {True, False}


def test_band_labels_follow_the_section_plane():
    anchors = condition(
        [(0, 5, (12, 12), None), (1, 2, (12, 12), None), (2, 9, (12, 12), None)],
        size=12,
    )
    far, near = bands(coordinates(12), anchors)
    for batch in (near, far):
        assert batch.axes.unique().tolist() == [0, 1, 2]
        assert torch.equal(section_axes(batch.values), batch.axes)


def test_bands_span_half_a_patch_from_a_random_start_and_mirror():
    far, near = bands(
        coordinates(12), condition([(0, 5, (12, 12), None)], size=12), width=6
    )
    for batch in (near, far):
        assert batch.values.shape[-1] == 6
        across = 3 - batch.axes
        lines = batch.values[torch.arange(len(batch)), across, 0]
        steps = lines.diff(dim=1)
        assert ((steps == 1).all(1) | (steps == -1).all(1)).all()
        assert set(steps[:, 0].tolist()) == {-1.0, 1.0}
        assert set(lines.amin(1).tolist()) == set(range(7))


def test_each_step_keeps_a_random_subset_of_bands_per_section_plane():
    anchors = condition([(0, 5, (8, 8), None), (1, 2, (8, 8), None)])
    far, near = bands(coordinates(), anchors, count=3)
    for batch in (near, far):
        assert batch.axes.tolist() == [0] * 3 + [1] * 3 + [2] * 3
    _, other = bands(coordinates(), anchors, count=3, seed=1)
    assert not torch.equal(near.values, other.values)
    _, single = bands(coordinates(), condition([(0, 5, (8, 8), None)]), count=3)
    assert single.axes.tolist() == [1] * 3 + [2] * 3


def composition(batch):
    pairs = torch.stack((section_rows(batch.values), batch.axes), 1)
    return sorted(map(tuple, pairs.tolist()))


def test_near_and_far_bands_share_section_planes_and_row_directions():
    anchors = condition(
        [(0, 8, (16, 16), None), (2, 8, (8, 8), None), (1, 3, (16, 16), None)],
        size=16,
    )
    far, near = bands(coordinates(16), anchors, rows=3, width=8, count=20)
    assert len(near) and composition(far) == composition(near)
    assert len(set(composition(near))) > 3


def test_section_planes_without_near_bands_have_no_far_bands():
    far, near = bands(
        coordinates(16), condition([(0, 8, (16, 6), (0, 4))], size=16), width=8
    )
    assert near.axes.unique().tolist() == far.axes.unique().tolist() == [2]
    assert len(far) == len(near)


def test_directed_axis_rows_are_read_upward_on_both_sides():
    anchors = condition([(0, 5, (8, 8), None), (1, 2, (8, 8), None)])
    far, near = bands(coordinates(), anchors, directed=0)
    for batch in (near, far):
        along_z = section_rows(batch.values) == 0
        assert along_z.any()
        steps = batch.values[along_z, 0, 1, 0] - batch.values[along_z, 0, 0, 0]
        assert (steps == 1).all()
    near_rows = {
        tuple(row)
        for row in near.values[:, 0, :, 0][section_rows(near.values) == 0].tolist()
    }
    assert near_rows == {(3.0, 4.0), (6.0, 7.0)}
    _, undirected = bands(coordinates(), anchors)
    steps = undirected.values[:, 0, 1, 0] - undirected.values[:, 0, 0, 0]
    assert set(steps[section_rows(undirected.values) == 0].tolist()) == {-1.0, 1.0}


def test_directed_axis_is_never_mirrored_across_a_band():
    anchors = condition([(1, 5, (8, 8), None), (2, 2, (8, 8), None)])
    far, near = bands(coordinates(), anchors, directed=0)
    for batch in (near, far):
        across = 3 - batch.axes - section_rows(batch.values)
        lines = batch.values[torch.arange(len(batch)), across, 0]
        steps = lines[:, 1] - lines[:, 0]
        assert (steps[across == 0] == 1).all()
        assert set(steps[across != 0].tolist()) == {-1.0, 1.0}
        assert (across == 0).any()


def test_bands_with_fixed_voxels_are_discarded():
    anchors = condition([(0, 5, (8, 8), None), (2, 1, (8, 8), None)])
    far, near = bands(coordinates(), anchors)
    assert len(near) and len(far)
    for batch in (near, far):
        assert not (batch.values[:, 0] == 5).any()
        assert not (batch.values[:, 2] == 1).any()


def test_far_bands_keep_more_than_n_rows_from_every_same_axis_anchor():
    anchors = condition([(0, 2, (12, 12), None), (0, 9, (12, 12), None)], size=12)
    far, near = bands(coordinates(12), anchors)
    far_rows = {tuple(row) for row in far.values[:, 0, :, 0].tolist()}
    assert far_rows == {(5.0, 6.0), (6.0, 5.0)}
    near_rows = {tuple(row) for row in near.values[:, 0, :, 0].tolist()}
    assert near_rows == {(1.0, 0.0), (3.0, 4.0), (8.0, 7.0), (10.0, 11.0)}


def test_partial_anchor_keeps_near_bands_inside_the_region():
    anchors = condition([(1, 4, (2, 6), (1, 2))])
    far, near = bands(coordinates(), anchors)
    assert len(near) == 12
    assert near.axes.unique().tolist() == [0]
    assert set(near.values[:, 0].unique().tolist()) == {1.0, 2.0}
    assert near.values[:, 2].amin() == 2.0 and near.values[:, 2].amax() == 7.0
    rows = {tuple(row) for row in near.values[:, 1, :, 0].tolist()}
    assert rows == {(3.0, 2.0), (5.0, 6.0)}
    assert len(far) == 12 and far.axes.unique().tolist() == [0]
    assert torch.equal(section_rows(far.values), section_rows(near.values))


def test_partial_anchor_narrower_than_a_band_has_no_near_bands():
    far, near = bands(coordinates(), condition([(1, 4, (2, 3), (1, 5))]))
    assert len(far) == len(near) == 0


def test_anchor_without_enough_rows_or_inactive_batch_has_no_bands():
    volume = coordinates(batch=2)
    far, near = bands(volume, condition([(0, 3, (8, 8), None)], batch=2), rows=4)
    assert len(far) == len(near) == 0
    assert near.values.shape == (0, 3, 4, 4)
    anchors = condition([(0, 5, (8, 8), None)], batch=2)
    _, both = bands(volume, anchors)
    anchors = copy.copy(anchors)
    object.__setattr__(anchors, "active_batches", (1,))
    _, one = bands(volume, anchors)
    assert len(both) == 2 * len(one)
    _, none = bands(volume, None)
    assert len(none) == 0


def test_axis_without_far_bands_contributes_no_near_bands():
    anchors = condition([(0, 6, (8, 8), None), (1, 4, (8, 8), None)])
    far, near = bands(coordinates(), anchors, rows=4)
    assert len(far) == len(near) == 0
    far, near = bands(coordinates(), anchors)
    assert len(near) and len(far)


def test_near_and_far_bands_without_a_shared_orientation_are_empty():
    # The axis-0 anchor has near and far bands, but in different section
    # planes, so none can be matched.
    anchors = [(0, 8, (2, 16), (7, 0))]
    anchors += [(2, index, (6, 16), (0, 0)) for index in (1, 4, 7, 10, 13)]
    anchors += [(2, index, (5, 16), (11, 0)) for index in (2, 5, 8, 11, 14)]
    far, near = bands(
        torch.randn(1, 3, 16, 16, 16), condition(anchors, size=16), count=100
    )
    assert len(far) == len(near) == 0


@pytest.mark.parametrize("name", ("rows", "width", "count"))
@pytest.mark.parametrize("value", (0, -1, 1.5, True))
def test_band_sizes_must_be_positive_integers(name, value):
    sizes = {"rows": 2, "width": 4, "count": 2, name: value}
    with pytest.raises(ValueError, match=name):
        sample_seam_bands(coordinates(), None, **sizes)


def test_seam_critic_scores_bands_by_axis_time_and_domain():
    torch.manual_seed(0)
    critic = SeamCritic2D(3, (4, 8), 8, 2, time_scale=0.5)
    bands = torch.randn(3, 3, 4, 8)
    axes = torch.tensor([0, 1, 2])
    time = torch.tensor([0, 1, 1])
    domains = torch.tensor([0, 1, NULL_DOMAIN])
    scores = critic(bands, axes, time, domains)
    assert scores.logits_global.shape == (3,) and scores.logits_local is None
    moved = critic(bands, torch.tensor([1, 1, 2]), time, domains)
    assert moved.logits_global[0] != scores.logits_global[0]
    torch.testing.assert_close(moved.logits_global[1:], scores.logits_global[1:])
    later = critic(bands, axes, torch.tensor([1, 1, 0]), domains)
    assert later.logits_global[0] != scores.logits_global[0]
    assert later.logits_global[2] != scores.logits_global[2]
    torch.testing.assert_close(later.logits_global[1], scores.logits_global[1])
    with pytest.raises(ValueError, match="bands"):
        critic(bands[0], torch.tensor([0]), torch.tensor([0]), torch.tensor([0]))
    with pytest.raises(ValueError, match="axes"):
        critic(bands, torch.tensor([0]), time, torch.tensor([0, 0, 0]))


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
    times = []
    trainer.seam_critic.register_forward_pre_hook(
        lambda module, args: times.append(args[2])
    )

    def record(volume, *args):
        sampled.append(volume)
        assert args[2:] == (4, 2, None)
        return sample_seam_bands(volume, *args)

    monkeypatch.setattr("src.train.trainer.sample_seam_bands", record)
    metrics = trainer.step(0, transition=1)
    assert sampled and sampled[0].requires_grad
    assert len(times) == 3
    assert all(time.tolist() == [1] * 4 for time in times)
    diagnostics = metrics.diagnostics
    assert trainer.updates["seam"] == 1
    assert diagnostics["anchor_seam/near_bands"] == 4
    assert diagnostics["anchor_seam/far_bands"] == 4
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
    time = trainer.make_time(0, len(fake))
    scores = trainer.seam_critic(fake.values, fake.axes, time, domains)
    scores.logits_global.sum().backward()
    assert prediction.grad[:, :, [3, 5]].abs().sum() > 0
    assert prediction.grad[:, :, :3].abs().sum() == 0
    assert prediction.grad[:, :, 4].abs().sum() == 0
    assert prediction.grad[:, :, 6:].abs().sum() == 0
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


def test_weights_export_and_initial_weights_include_the_seam_critic(tmp_path):
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


def test_checkpoint_restores_the_seam_critic(tmp_path):
    torch.manual_seed(0)
    cfg = small_config(tmp_path)
    trainer = build_trainer(cfg, torch.device("cpu"))
    fixed_anchor(trainer)
    trainer.step(0, transition=0)
    path = tmp_path / "training.pt"
    save_training(trainer, path)
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


@pytest.mark.parametrize("missing", ["seam", "seam_optim", "counter"])
def test_resume_rejects_incomplete_seam_state(tmp_path, missing):
    cfg = small_config(tmp_path)
    trainer = build_trainer(cfg, torch.device("cpu"))
    path = tmp_path / "training.pt"
    save_training(trainer, path)
    payload = torch.load(path, weights_only=True)
    if missing == "counter":
        payload["updates"].pop("seam")
    else:
        payload.pop(missing)
    message = "update counters" if missing == "counter" else "is missing"
    with pytest.raises(ValueError, match=message):
        resume_training(build_trainer(cfg, torch.device("cpu")), payload)
