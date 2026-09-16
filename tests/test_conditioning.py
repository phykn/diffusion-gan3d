from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.anchor import PlaneAnchor, encode_anchors
from src.build.data import build_datasets, resolve_height_metadata
from src.build.predict import load_generator
from src.build.sr import build_sr_trainer
from src.build.trainer import build_trainer
from src.config import load_train_config, save_yaml
from src.model.diffusion import Diffusion
from src.predict.generator import Generator
from src.predict.sr import SuperResolutionAPI
from src.predict.tiled import TiledGenerator
from src.prepare.resize import resize_crop
from src.train.loss.anchor import SoftAnchorLoss
from src.train.sr import export_sr
from src.train.state import resume_training, save_training


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
            result = torch.lerp(result, anchor_image, anchor_mask.float())
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
    scaled = TiledGenerator(generator(model))
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


def configuration(tmp_path, stage="low_res", height=False):
    folder = tmp_path / "images"
    folder.mkdir(exist_ok=True)
    labels = (np.indices((24, 24)).sum(0) // 3 % 2).astype(np.uint8)
    Image.fromarray(labels).save(folder / "sample.png")
    cfg = load_train_config(f"config/train/{stage}.yaml", stage)
    cfg["data"].update(
        crop_size=8,
        lo_res_size=8,
        num_phases=2,
        domains={0: {"xy": [str(folder)], "xz": [str(folder)], "yz": [str(folder)]}},
    )
    cfg["conditioning"]["height_enabled"] = height
    cfg["model"]["critic"].update(
        channels=[4, 8], plane_groups=[["xy"], ["xz", "yz"]], pyramid_min_size=4
    )
    cfg["train"].update(
        mixed_precision=False,
        volume_batch_size=1,
        total_steps=4,
        structure_every_steps=1,
    )
    if stage == "low_res":
        cfg["model"]["generator"].update(
            channels=[4, 8], embedding_channels=8, latent_channels=4
        )
        cfg["model"]["diffusion"]["num_steps"] = 2
        cfg["model"]["gradient_checkpointing"] = False
        cfg["conditioning"]["anchor"].update(
            probability=1, ramp_steps=0, plane_spacing=2
        )
        cfg["conditioning"]["dropout_probability_per_case"] = 0
        cfg["train"].update(real_batch_size=2, slice_pairs_per_plane=2, num_workers=0)
        cfg["loss"]["r1_weight"] = 0
        cfg["loss"]["connectivity"].update(
            adversarial_weight=0.1, normal_transition_weight=0.1, ramp_steps=0
        )
    else:
        cfg["model"]["generator"].update(
            channels=[4, 8], embedding_channels=8, latent_channels=4
        )
        cfg["model"]["diffusion"]["num_steps"] = 2
        cfg["model"]["gradient_checkpointing"] = False
        cfg["data"]["hi_res_size"] = 16
        cfg["train"].update(real_batch_size=2, slice_pairs_per_plane=2)
    return cfg


def test_replay_training_never_generates_reference_and_survives_resume(tmp_path):
    cfg = configuration(tmp_path)
    trainer = build_trainer(cfg, torch.device("cpu"))
    with patch.object(
        trainer, "generate_pair", wraps=trainer.generate_pair
    ) as generate:
        first = trainer.step(0, transition=0)
        second = trainer.step(1, transition=0)
    assert generate.call_count == 2
    assert first.anchor_planes == 1 and second.anchor_planes == 4
    assert second.anchor_loss > 0 and second.generator_connectivity > 0
    assert second.diagnostics["sampling/reference_passes"] == 0
    assert first.anchor_neighbor_agreement is not None
    assert any("two_point_mae" in key for key in second.diagnostics)
    save_training(tmp_path / "last.pt", trainer)
    restored = build_trainer(cfg, torch.device("cpu"))
    payload = torch.load(tmp_path / "last.pt", weights_only=True)
    resume_training(restored, payload)
    assert len(restored.anchor_bank.entries[0]) == len(trainer.anchor_bank.entries[0])
    assert not any("rng" in key for key in payload)


def test_height_origin_survives_loader_training_export_and_tiles(tmp_path):
    cfg = configuration(tmp_path, height=True)
    resolve_height_metadata(cfg)
    assert cfg["data"]["height_extents"] == {0: 24}
    dataset = build_datasets(cfg)[0][1]
    with patch("src.data.dataset.np.random.randint", side_effect=[7, 2]):
        sample = dataset[tmp_path / "images/sample.png"]
    assert sample["height_origin"] == 7
    trainer = build_trainer(cfg, torch.device("cpu"))
    trainer.step(0, transition=0)
    assert trainer.denoiser.height_input.weight.grad.abs().sum() > 0
    save_yaml(tmp_path / "train.yaml", trainer.cfg)
    torch.save(trainer.denoiser.state_dict(), tmp_path / "generator.pt")
    generator = load_generator(tmp_path / "generator.pt", torch.device("cpu"))
    heights = []
    original = generator.predict

    def record(*args, **kwargs):
        heights.append(kwargs["height"].clone())
        return original(*args, **kwargs)

    scaled = TiledGenerator(generator)
    with patch.object(generator, "predict", side_effect=record):
        result = scaled.generate_probs(
            (12, 8, 8), overlap=1, margin=0, height_origin=3, progress=False
        )
    assert result.shape == (2, 12, 8, 8)
    torch.testing.assert_close(heights[0][:, :, 4:], heights[1][:, :, :4])
    assert heights[0][0, 0, 0, 0, 0] == pytest.approx(2 * 3.5 / 24 - 1)


def test_height_conditioned_sr_uses_fractional_bank_and_zero_level_at_inference(
    tmp_path,
):
    cfg = configuration(tmp_path, stage="sr", height=True)
    bank = {0: torch.rand(2, 2, 8, 8, 8).softmax(1)}
    trainer = build_sr_trainer(
        cfg, bank, torch.device("cpu"), {0: torch.tensor([2.0, 8.0])}
    )
    high_height = trainer.volume_height(torch.tensor([2.0]), 0)
    assert high_height[0, 0, 0, 0, 0] == pytest.approx(2 * 2.25 / 24 - 1)
    metrics = trainer.step(0)
    assert np.isfinite(metrics.generator_total)
    assert trainer.denoiser.height_input.weight.grad.abs().sum() > 0
    export_sr(trainer, tmp_path / "model.pt")
    api = SuperResolutionAPI(tmp_path / "model.pt")
    with patch.object(api.model, "forward", wraps=api.model.forward) as forward:
        result = api.predict_probs(bank[0][0], height_origin=2, margin=0)
    assert result.shape == (2, 16, 16, 16)
    assert forward.call_args.kwargs["corruption_level"].eq(0).all()
    assert forward.call_count == 2
    assert forward.call_args.kwargs["height"][0, 0, 0, 0, 0] == pytest.approx(
        2 * 2.25 / 24 - 1
    )


def test_partial_anchor_with_cfg_uses_two_forwards_per_transition(tmp_path):
    trainer = build_trainer(configuration(tmp_path), torch.device("cpu"))
    generator = Generator(
        trainer.denoiser.eval(), trainer.diffusion, trainer.device, 8, 2, 4, False
    )
    anchor = PlaneAnchor(torch.zeros(8, 8, dtype=torch.long), 0, 3)
    with patch.object(
        trainer.denoiser, "compute_logits", wraps=trainer.denoiser.compute_logits
    ) as logits:
        generator.generate_probs(
            anchors=[anchor], guidance=1.5, anchor_strength=0.8, margin=0
        )
    assert logits.call_count == 2 * trainer.diffusion.timesteps
    masks = [
        call.kwargs["anchor_mask"]
        for call in logits.call_args_list
        if call.kwargs.get("anchor_mask") is not None
    ]
    assert all(mask.max() == 0.8 for mask in masks)
