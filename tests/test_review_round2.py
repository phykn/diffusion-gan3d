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
from src.evaluate.structure import structure_metrics
from src.model.critic import PairCritic2D
from src.model.diffusion import Diffusion
from src.model.layers import embed_domain
from src.predict.generator import Generator
from src.predict.scale import ScaledGenerator
from src.predict.sr import SuperResolutionAPI
from src.prepare.height import height_field
from src.train.anchor_bank import AnchorBank
from src.train.loss.connect import AnchorTripletSampler
from src.train.loss.gan import get_critic_loss, get_generator_loss
from src.train.sr_loss import consistency_loss
from src.train.state import resume_training, save_training
from src.train.trainer import Trainer


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
        cfg["model"]["generator"].update(channels=4, blocks=1, scale_factor=2)
        cfg["train"].update(slices_per_plane=2, critic_updates_per_step=1)
    return cfg


@pytest.mark.parametrize("axis", range(3))
def test_measured_parallel_plane_is_excluded_from_adversarial_gradients(axis):
    trainer = object.__new__(Trainer)
    trainer.slice_pairs_per_axis = 100
    trainer.patch_size = 5
    volume = torch.randn(1, 2, 5, 5, 5, requires_grad=True)
    condition = encode_anchors(
        [PlaneAnchor(torch.zeros(5, 5, dtype=torch.long), axis, 2)],
        1,
        2,
        5,
        torch.device("cpu"),
        torch.float32,
    )
    previous, _ = trainer.sample_pairs(volume, volume, axis, measured=condition)
    previous.sum().backward()
    assert volume.grad.select(axis + 2, 2).count_nonzero() == 0
    assert volume.grad.sum() > 0


@pytest.mark.parametrize("size,count", [(8, 4), (16, 8)])
def test_replay_keeps_measurement_and_plane_density(size, count):
    bank = AnchorBank(capacity=1, plane_spacing=2)
    image = torch.stack((torch.full((size, size), 0.2), torch.full((size, size), 0.8)))
    measured = encode_anchors(
        [PlaneAnchor(image, 0, 2)], 1, 2, size, torch.device("cpu"), torch.float32
    )
    prediction = torch.zeros(1, 2, size, size, size)
    bank.add(0, prediction, measured, torch.tensor([True]))
    condition, target, reference, _ = bank.sample(0, 2, torch.device("cpu"))
    assert condition.planes == count
    torch.testing.assert_close(
        condition.image[:, :, 2], measured.image[:, :, 2].expand(2, -1, -1, -1)
    )
    torch.testing.assert_close(reference[:, :, 2], target.image[:, :, 2])
    assert target.regions == measured.regions
    assert bank.sample(1, 1, torch.device("cpu")) is None


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


def test_pyramid_averages_losses_after_nonlinearity():
    critic = PairCritic2D(2, (4, 8), 8, 1)
    critic.pyramid_min_size = 4
    previous = torch.randn(2, 2, 16, 16, requires_grad=True)
    current = torch.randn_like(previous)
    scores = critic(
        previous,
        current,
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
    )
    assert len(scores.levels) == 3
    assert [s.logits_local.shape[-1] for s in scores.levels] == [8, 4, 2]
    loss = get_generator_loss(scores)
    expected = torch.stack(
        [get_generator_loss(s).global_loss for s in scores.levels]
    ).mean()
    torch.testing.assert_close(loss.global_loss, expected)
    get_critic_loss(scores, scores).combine(0.5).backward()
    assert torch.isfinite(previous.grad).all()


def test_fractional_consistency_does_not_sharpen_coarse():
    low = torch.empty(1, 2, 4, 4, 4)
    low[:, 0], low[:, 1] = 0.3, 0.7
    high = torch.nn.functional.interpolate(low, scale_factor=2).requires_grad_()
    loss, error = consistency_loss(high, low, 0)
    assert float(loss.detach()) < 1e-12 and float(error.detach()) < 1e-12


def test_structure_metrics_detect_disconnection_and_match_measured_planes():
    labels = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    labels[:, :, 3, 3] = 1
    probs = torch.nn.functional.one_hot(labels, 2).movedim(-1, 1).float()
    real = {0: probs[:, :, 2]}
    metrics = structure_metrics(probs, real, {"xy": (0,)})
    assert metrics["structure/phase_1/percolation_xy"] == 1
    assert metrics["structure/phase_1/percolation_lower_bound_xy"] == 1
    assert metrics["structure/phase_1/percolation_xz"] == 0
    assert metrics["structure/xy/two_point_mae"] == 0
    assert metrics["structure/xy/chord_tv"] == 0
    probs[:, :, 4] = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1)
    assert (
        structure_metrics(probs, real, {"xy": (0,)})["structure/phase_1/percolation_xy"]
        == 0
    )


def test_height_origin_survives_loader_training_export_and_tiles(tmp_path):
    cfg = configuration(tmp_path, height=True)
    resolve_height_metadata(cfg)
    assert cfg["data"]["height_extents"] == {0: 24}
    dataset = build_datasets(cfg)[0][1]
    with patch("src.data.real.np.random.randint", side_effect=[7, 2]):
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

    scaled = ScaledGenerator(generator)
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
    low_height = trainer.volume_height(bank[0][:1], torch.tensor([2.0]), 0)
    _, high_height = trainer.slices_with_height(
        torch.zeros(1, 2, 16, 16, 16), low_height, 1, 1, 0
    )
    assert high_height[0, 0, 0, 0] == pytest.approx(2 * 2.25 / 24 - 1)
    metrics = trainer.train_step()
    assert np.isfinite(metrics["generator"])
    assert trainer.model.height_input.weight.grad.abs().sum() > 0
    trainer.export(tmp_path / "model.pt")
    api = SuperResolutionAPI(tmp_path / "model.pt")
    with patch.object(api.model, "forward", wraps=api.model.forward) as forward:
        result = api.predict_probs(bank[0][0], height_origin=2)
    assert result.shape == (2, 16, 16, 16)
    assert forward.call_args.args[3].eq(0).all()
    assert forward.call_args.kwargs["height"][0, 0, 0, 0, 0] == pytest.approx(
        2 * 2.5 / 24 - 1
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_posterior_and_domain_validation_do_not_read_host_scalars():
    device = torch.device("cuda")
    diffusion = Diffusion(2).to(device)
    current = torch.randn(1, 2, 4, 4, 4, device=device)
    embedding = torch.nn.Embedding(2, 4).to(device)
    time = torch.tensor([1], device=device)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        diffusion.sample_posterior(current, current, time)
        embed_domain(embedding, time, torch.float32)
    assert not any(
        event.key == "aten::_local_scalar_dense" for event in profile.key_averages()
    )


def test_triplet_sampler_uses_integer_regions_without_dense_mask_scan():
    prediction = torch.zeros(1, 2, 8, 8, 8)
    condition = encode_anchors(
        [PlaneAnchor(torch.zeros(8, 8, dtype=torch.long), 0, 3)],
        1,
        2,
        8,
        torch.device("cpu"),
        torch.float32,
    )
    with (
        patch.object(
            torch.Tensor, "nonzero", side_effect=AssertionError("dense mask scan")
        ),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("host transfer")),
    ):
        real, fake = AnchorTripletSampler(windows_per_plane=4).sample(
            prediction, prediction, condition
        )
    assert len(real) == len(fake) == 12


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


def test_height_field_uses_cell_centers_and_physical_pixel_scale():
    height = height_field((4, 4, 4), 1, [8, 16], 2, 32)
    torch.testing.assert_close(
        height[:, 0, 0, :, 0],
        torch.tensor(
            [[-0.4375, -0.3125, -0.1875, -0.0625], [0.0625, 0.1875, 0.3125, 0.4375]]
        ),
    )
