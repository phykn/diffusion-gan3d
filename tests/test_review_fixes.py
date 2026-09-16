from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.anchor import PlaneAnchor, encode_anchors
from src.build.sr import build_sr_trainer
from src.build.trainer import build_trainer
from src.config import load_train_config, normalize_train_config
from src.model.diffusion import Diffusion
from src.predict.generator import Generator
from src.predict.scale import ScaledGenerator
from src.prepare.resize import resize_crop
from src.train.loss.anchor import SoftAnchorLoss
from src.train.loss.connect import AnchorTripletSampler, anchor_boundary_metrics
from src.train.loss.vf import compute_vf_loss
from src.train.sr import SRTrainer
from src.train.sr_loss import consistency_loss
from src.train.sr_run import file_hash, refresh_bank


@pytest.mark.parametrize(
    "section,key",
    [
        ("train", "seed"),
        ("train", "stability_version"),
        ("optim", "generatr_lr"),
        ("data", "input_size"),
    ],
)
def test_obsolete_and_unknown_config_keys_fail_with_full_path(section, key):
    with pytest.raises(ValueError, match=rf"{section}\.{key}"):
        normalize_train_config({section: {key: 1}})


def test_subpixel_phase_fraction_reaches_anchor_and_soft_loss():
    labels = torch.zeros(4, 4, dtype=torch.long)
    labels[:, 0] = 1
    fractions = resize_crop(labels, 2, 2)
    torch.testing.assert_close(fractions[1], torch.tensor([[0.5, 0], [0.5, 0]]))
    assert fractions[1].mean() == 0.25
    condition = encode_anchors(
        (PlaneAnchor(fractions, 0, 0),), 1, 2, 2, torch.device("cpu"), torch.float32
    )
    torch.testing.assert_close((condition.image[0, :, 0] + 1) / 2, fractions)
    logits = torch.zeros(1, 2, 2, 2, 2, requires_grad=True)
    result = SoftAnchorLoss(1, 1)(logits, condition, torch.tensor([True]))
    result.total.backward()
    assert logits.grad[0, 0, 0, 0, 0] == 0
    assert logits.grad[0, 1, 0, 0, 0] == 0
    assert logits.grad[0, 1, 0, 0, 1] > 0


class AnchorModel(torch.nn.Module):
    num_domains = 1
    downsample_factor = 1

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        current,
        time,
        latent,
        domain,
        vf=None,
        anchor_image=None,
        anchor_mask=None,
    ):
        self.calls.append((time.clone(), anchor_mask))
        result = torch.full_like(current, -1)
        result[:, 0] = 1
        if anchor_mask is not None:
            result = torch.where(anchor_mask, anchor_image, result)
        return result


def generator(model):
    return Generator(model, Diffusion(3), torch.device("cpu"), 8, 2, 4, False)


def test_full_strength_anchor_uses_one_model_evaluation_per_transition():
    model = AnchorModel()
    gen = generator(model)
    anchor = PlaneAnchor(torch.ones(8, 8, dtype=torch.long), 0, 5)
    vol = gen.generate(anchors=(anchor,), margin=0)
    assert len(model.calls) == 3
    assert all(mask is not None for _, mask in model.calls)
    assert torch.equal(vol[5], anchor.image.to(torch.uint8))


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("partial", [False, True])
def test_global_anchor_crosses_tiles_and_keeps_rectangular_coordinates(axis, partial):
    model = AnchorModel()
    scaled = ScaledGenerator(generator(model))
    shape = (10, 12, 14)
    plane_shape = tuple(n for i, n in enumerate(shape) if i != axis)
    image_shape = (7, 9) if partial else plane_shape
    labels = (torch.arange(np.prod(image_shape)).reshape(image_shape) % 2).long()
    position = (2, 3) if partial else None
    expected = torch.zeros(plane_shape, dtype=torch.uint8)
    row, col = position or (0, 0)
    expected[row : row + image_shape[0], col : col + image_shape[1]] = labels.to(
        torch.uint8
    )
    index = shape[axis] - 2
    result = scaled.generate(
        shape=shape,
        overlap=2,
        anchors=(PlaneAnchor(labels, axis, index, position),),
        progress=False,
    )
    assert result.shape == shape
    assert torch.equal(result.select(axis, index), expected)
    assert sum(mask is not None for _, mask in model.calls) > 3


def test_connectivity_samples_every_parallel_anchor_and_includes_adjacent_gap():
    anchors = tuple(
        PlaneAnchor(torch.zeros(8, 8, dtype=torch.long), 0, i) for i in (2, 5)
    )
    condition = encode_anchors(anchors, 1, 2, 8, torch.device("cpu"), torch.float32)
    volume = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
    sampler = AnchorTripletSampler(max_gap=3, windows_per_plane=4)
    located = sampler._sample_anchor_triplets(volume, condition)
    for index in (2, 5):
        slots = [i for i, pos in enumerate(located.locations) if pos == (0, 0, index)]
        assert len(slots) == 4
        assert located.triplets.gaps[slots[0]] == 1
    assert located.triplets.values.shape[-2:] == (4, 4)
    real, fake = sampler.sample(volume, volume.detach(), condition)
    torch.testing.assert_close(real.values, fake.values)
    fake.values.sum().backward()
    assert volume.grad.abs().sum() > 0


def test_boundary_metric_detects_a_detached_anchor_plane():
    condition = encode_anchors(
        (PlaneAnchor(torch.zeros(6, 6, dtype=torch.long), 0, 3),),
        1,
        2,
        6,
        torch.device("cpu"),
        torch.float32,
    )
    reference = torch.ones(1, 2, 6, 6, 6)
    reference[:, 1] = -1
    prediction = reference.clone()
    prediction[:, :, 2] *= -1
    values = anchor_boundary_metrics(prediction, reference, condition)
    assert values["anchor/neighbor_agreement"] == 0.5
    assert values["anchor/neighbor_excess_jump"] == 0.5


def test_nonfinal_real_anchor_does_not_generate_unused_reference(tmp_path):
    torch.set_num_threads(1)
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(tmp_path / "image.png")
    cfg = load_train_config("config/train/low_res.yaml")
    cfg["data"].update(
        crop_size=8,
        lo_res_size=8,
        domains={0: {p: [str(tmp_path)] for p in ("xy", "xz", "yz")}},
    )
    cfg["model"]["generator"].update(
        channels=[4, 8], embedding_channels=8, latent_channels=4
    )
    cfg["model"]["critic"]["channels"] = [4, 8]
    cfg["model"]["diffusion"]["num_steps"] = 3
    cfg["model"]["gradient_checkpointing"] = False
    cfg["train"].update(
        real_batch_size=1, slice_pairs_per_plane=2, mixed_precision=False
    )
    cfg["conditioning"]["anchor"].update(probability=1, ramp_steps=2)
    cfg["loss"]["connectivity"]["ramp_steps"] = 20
    trainer = build_trainer(cfg, torch.device("cpu"))
    with patch.object(
        trainer, "generate_pair", wraps=trainer.generate_pair
    ) as generate:
        metrics = trainer.step(0, transition=1)
    assert generate.call_count == 1
    assert metrics.anchor_ramp == 0.5
    assert metrics.connectivity_ramp == 0.05


def test_sr_corruption_does_not_modify_clean_coarse_target():
    trainer = object.__new__(SRTrainer)
    trainer.cfg = {
        "conditioning": {
            "coarse_corruption_probability": 1,
            "coarse_corruption_strength": 1,
        }
    }
    low = torch.zeros(8, 3, 8, 8, 8)
    low[:, 0] = 1
    original = low.clone()
    corrupted = trainer.corrupt_coarse(low)
    assert torch.equal(low, original)
    assert not torch.equal(corrupted, low)
    torch.testing.assert_close(corrupted.sum(1), torch.ones_like(corrupted[:, 0]))


def test_sr_loss_targets_clean_coarse_while_model_receives_corrupted_input(tmp_path):
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(tmp_path / "sample.png")
    cfg = load_train_config("config/train/sr.yaml", "sr")
    cfg["data"].update(crop_size=8, lo_res_size=8, domains={0: {"xy": [str(tmp_path)]}})
    cfg["model"]["generator"].update(channels=4, blocks=1, scale_factor=1)
    cfg["model"]["critic"].update(channels=[4, 8], plane_groups=[["xy"]])
    cfg["train"].update(
        mixed_precision=False, critic_updates_per_step=1, slices_per_plane=1
    )
    trainer = build_sr_trainer(
        cfg, {0: torch.zeros(2, 8, 8, 8, dtype=torch.uint8)}, torch.device("cpu")
    )
    with (
        patch.object(trainer, "corrupt_coarse", side_effect=lambda low: low.flip(1)),
        patch.object(trainer.model, "forward", wraps=trainer.model.forward) as forward,
        patch("src.train.sr.consistency_loss", wraps=consistency_loss) as consistency,
    ):
        trainer.train_step()
    assert all(call.args[0][:, 1].eq(1).all() for call in forward.call_args_list)
    assert consistency.call_args.args[1][:, 0].eq(1).all()


def test_anchor_and_vf_losses_do_not_read_device_scalars_for_control_flow():
    condition = encode_anchors(
        (PlaneAnchor(torch.zeros(4, 4, dtype=torch.long), 0, 2),),
        1,
        2,
        4,
        torch.device("cpu"),
        torch.float32,
    )
    logits = torch.randn(1, 2, 4, 4, 4, requires_grad=True)
    visible = torch.tensor([False])
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        loss = SoftAnchorLoss(2, 0.05)(logits, condition, visible).total
        loss = loss + compute_vf_loss(
            logits.softmax(1), torch.tensor([[0.5, 0.5]]), visible
        )
        loss.backward()
    assert not any(
        event.key == "aten::_local_scalar_dense" for event in profile.key_averages()
    )


def test_bank_refresh_saves_new_bank_without_overwriting_resume_source(
    tmp_path, monkeypatch
):
    source = tmp_path / "generator.pt"
    source.write_bytes(b"frozen weights")
    source_config = tmp_path / "train.yaml"
    source_config.write_text("stage: low_res\n", encoding="utf-8")
    old_bank = tmp_path / "lr_bank.pt"
    old_bank.write_bytes(b"original bank")
    cfg = {
        "data": {},
        "source": {
            "weights": str(source),
            "weights_sha256": file_hash(source),
            "config_sha256": file_hash(source_config),
            "bank": str(old_bank),
        },
        "lr_bank": {"refresh_every_steps": 2, "guidance": 1},
    }
    trainer = SimpleNamespace(
        cfg=cfg,
        step=2,
        device=torch.device("cpu"),
        bank={0: torch.zeros(2, 8, 8, 8, dtype=torch.uint8)},
    )
    monkeypatch.setattr(
        "src.train.sr_run.load_generator",
        lambda *args: SimpleNamespace(
            generate=lambda **kwargs: torch.ones(8, 8, 8, dtype=torch.uint8)
        ),
    )
    refresh_bank(trainer, tmp_path)
    assert old_bank.read_bytes() == b"original bank"
    assert trainer.bank[0][0].eq(1).all()
    assert trainer.bank[0][1].eq(0).all()
    assert cfg["source"]["bank_sha256"] == file_hash(
        tmp_path / "lr_bank_step_00000002.pt"
    )
    source_config.write_text("stage: sr\n", encoding="utf-8")
    with pytest.raises(ValueError, match="configuration changed"):
        refresh_bank(trainer, tmp_path)
