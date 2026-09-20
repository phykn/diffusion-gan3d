import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.anchor import PlaneAnchor, encode_anchors
from src.build.data import build_datasets
from src.build.predict import load_generator
from src.build.sr import build_sr_trainer
from src.build.trainer import build_trainer
from src.config.files import save_yaml
from src.config.train import load_train_config
from src.data.source import infer_height_extents
from src.model.diffusion import Diffusion
from src.predict.generator import Generator
from src.predict.sr.inference import SuperResolutionAPI
from src.predict.tiling.sampler import TiledGenerator
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
    cfg = load_train_config("tests/fixtures/config/train/low_res.yaml")
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
    cfg = load_train_config(f"tests/fixtures/config/train/{stage}.yaml", stage)
    cfg["data"].update(
        crop_size=8,
        lo_res_size=8,
        num_phases=2,
        domains={0: {"xy": [str(folder)], "xz": [str(folder)], "yz": [str(folder)]}},
    )
    cfg["conditioning"]["height_enabled"] = height
    if height:
        cfg["data"]["thickness_axis"] = "z"
        for plane, flips in (("xz", ["x"]), ("yz", ["y"])):
            cfg["augmentation"]["planes"][plane] = {
                "flip_axes": flips,
                "rotate_90": False,
            }
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


def test_profile_training_replay_and_conditional_critics(tmp_path):
    cfg = configuration(tmp_path, height=True)
    cfg["conditioning"]["spatial_profile"].update(
        enabled=True, num_bins=4, critic_enabled=True
    )
    Image.fromarray(np.zeros((32, 24), dtype=np.uint8)).save(
        tmp_path / "images/second.png"
    )
    trainer = build_trainer(cfg, torch.device("cpu"))
    assert trainer.cfg["data"]["height_extents"] == {0: None}
    first = trainer.step(0, transition=0)
    assert np.isfinite(first.generator_total)
    assert trainer.denoiser.profile_input.weight.grad.abs().sum() > 0
    original = trainer.anchor_bank.entries[0][0]
    assert original["profile"] is not None
    assert original["geometry"]["height_extent"] in (24, 32)
    assert Path(original["geometry"]["image_id"]).is_file()
    second = trainer.step(1, transition=0)
    assert second.generator_connectivity > 0
    assert trainer.connectivity_critic.height_input.weight.grad.abs().sum() > 0
    assert trainer.connectivity_critic.profile_input.weight.grad.abs().sum() > 0
    assert "profile/label_mae" in second.diagnostics
    save_training(tmp_path / "last.pt", trainer)
    restored = build_trainer(cfg, torch.device("cpu"))
    resume_training(restored, torch.load(tmp_path / "last.pt", weights_only=True))
    torch.testing.assert_close(
        restored.anchor_bank.entries[0][0]["profile"], original["profile"]
    )


def test_profile_cfg_removes_profile_and_keeps_height(tmp_path):
    cfg = configuration(tmp_path, height=True)
    cfg["conditioning"]["spatial_profile"]["enabled"] = True
    trainer = build_trainer(cfg, torch.device("cpu"))
    model = trainer.denoiser.eval()
    current = torch.randn(1, 2, 8, 8, 8)
    latent = torch.randn(1, 4)
    domain = torch.zeros(1, dtype=torch.long)
    height = trainer.volume_height([2], 0)
    profile = torch.rand(1, 2, 8).softmax(1)
    with (
        torch.no_grad(),
        patch.object(model, "compute_logits", wraps=model.compute_logits) as logits,
    ):
        model.apply_guidance_logits(
            current,
            domain,
            latent,
            2,
            domain,
            vf=profile.mean(-1),
            height=height,
            profile=profile,
        )
    assert len(logits.call_args_list) == 2
    assert "profile" not in logits.call_args_list[0].kwargs
    assert logits.call_args_list[0].kwargs["height"] is height
    assert logits.call_args_list[1].kwargs["profile"] is profile
    with torch.no_grad():
        dropped = model.compute_logits(
            current,
            domain,
            latent,
            domain,
            vf=profile.mean(-1),
            vf_present=torch.tensor([False]),
            height=height,
            profile=profile,
            profile_present=torch.tensor([False]),
        )
        absent = model.compute_logits(current, domain, latent, domain, height=height)
    torch.testing.assert_close(dropped, absent)


def test_profile_public_api_responds_to_equal_mean_profiles_with_same_seed(tmp_path):
    from src.api import InferenceAPI

    cfg = configuration(tmp_path, height=True)
    cfg["conditioning"]["spatial_profile"]["enabled"] = True
    trainer = build_trainer(cfg, torch.device("cpu"))
    save_yaml(tmp_path / "train.yaml", trainer.cfg)
    torch.save(trainer.denoiser.state_dict(), tmp_path / "generator.pt")
    api = InferenceAPI(tmp_path / "generator.pt", device="cpu")
    first = {
        "axis": "z",
        "points": [[0, [0.2, 0.8]], [1, [0.8, 0.2]]],
        "interpolation": "linear",
    }
    second = {
        "axis": "z",
        "points": [[0, [0.8, 0.2]], [1, [0.2, 0.8]]],
        "interpolation": "linear",
    }
    a = api.generate_probs(size=8, seed=3, height_extent=8, vf_profile=first)
    b = api.generate_probs(size=8, seed=3, height_extent=8, vf_profile=second)
    repeated = api.generate_probs(size=8, seed=3, height_extent=8, vf_profile=first)
    torch.testing.assert_close(a, repeated)
    assert not torch.allclose(a, b)


def test_profile_tiles_use_global_coordinates_and_local_means(tmp_path):
    cfg = configuration(tmp_path, height=True)
    cfg["conditioning"]["spatial_profile"]["enabled"] = True
    trainer = build_trainer(cfg, torch.device("cpu"))
    save_yaml(tmp_path / "train.yaml", trainer.cfg)
    torch.save(trainer.denoiser.state_dict(), tmp_path / "generator.pt")
    generator = load_generator(tmp_path / "generator.pt", torch.device("cpu"))
    spec = {
        "axis": "z",
        "points": [[0, [0.1, 0.9]], [1, [0.9, 0.1]]],
        "interpolation": "linear",
    }
    with patch.object(generator, "predict", wraps=generator.predict) as predict:
        result = TiledGenerator(generator).generate_probs(
            (12, 8, 8),
            overlap=1,
            margin=0,
            height_origin=3,
            height_extent=32,
            vf_profile=spec,
            progress=False,
        )
    assert result.shape == (2, 12, 8, 8)
    calls = [call.kwargs for call in predict.call_args_list]
    torch.testing.assert_close(
        calls[0]["profile"][..., 4:], calls[1]["profile"][..., :4]
    )
    for call in calls:
        torch.testing.assert_close(call["vf"], call["profile"].mean(-1))
    assert not torch.equal(calls[0]["vf"], calls[1]["vf"])
    with pytest.raises(ValueError, match="profile mean"):
        generator.generate(vf_profile=spec, vf=[1, 0], height_extent=32)


def test_profile_rejects_impossible_anchor_and_preserved_base(tmp_path):
    cfg = configuration(tmp_path, height=True)
    cfg["conditioning"]["spatial_profile"]["enabled"] = True
    trainer = build_trainer(cfg, torch.device("cpu"))
    save_yaml(tmp_path / "train.yaml", trainer.cfg)
    torch.save(trainer.denoiser.state_dict(), tmp_path / "generator.pt")
    generator = load_generator(tmp_path / "generator.pt", torch.device("cpu"))
    spec = {
        "axis": "z",
        "points": [[0, [1, 0]], [1, [1, 0]]],
        "interpolation": "constant",
    }
    anchor = PlaneAnchor(torch.ones(8, 8, dtype=torch.long), 1, 2)
    with pytest.raises(ValueError, match="profile conflicts"):
        generator.generate(anchors=[anchor], vf_profile=spec)
    with pytest.raises(ValueError, match="profile conflicts"):
        TiledGenerator(generator).generate(
            shape=(12, 8, 8),
            overlap=1,
            base=torch.ones(8, 8, 8, dtype=torch.long),
            preserve_base=True,
            vf_profile=spec,
            progress=False,
        )


def test_sr_bank_records_2d_profile_and_per_image_extent(tmp_path):
    from src.train.run.bank import sample_bank_condition

    cfg = configuration(tmp_path, height=True)
    cfg["conditioning"]["spatial_profile"].update(enabled=True, num_bins=4)
    record = sample_bank_condition(cfg, 0, build_datasets(cfg))
    assert record["height_extent"] == 24
    assert record["source_shape"] == [24, 24]
    assert "vf_profile" in record
    assert record["vf_profile"]["points"][0][0] == 0
    assert record["vf_profile"]["points"][-1][0] == 1
    sr_cfg = configuration(tmp_path, stage="sr", height=True)
    Image.fromarray(np.zeros((32, 24), dtype=np.uint8)).save(
        tmp_path / "images/second.png"
    )
    bank = {0: torch.rand(2, 2, 8, 8, 8).softmax(1)}
    trainer = build_sr_trainer(
        sr_cfg,
        bank,
        torch.device("cpu"),
        {0: torch.tensor([2.0, 8.0])},
        {0: torch.tensor([24.0, 32.0])},
    )
    assert np.isfinite(trainer.step(0).generator_total)
    export_sr(trainer, tmp_path / "sr.pt")
    api = SuperResolutionAPI(tmp_path / "sr.pt")
    with pytest.raises(ValueError, match="height_extent is required"):
        api.predict_probs(bank[0][0])
    with patch.object(api, "_height", wraps=api._height) as height:
        output = api.predict_probs(
            bank[0][0], height_origin=8, height_extent=32, tile_size=16, overlap=2
        )
    assert output.shape == (2, 16, 16, 16)
    assert height.call_args.args[-1] == 32


def test_profile_sr_training_refresh_and_geometry_artifacts(tmp_path):
    from src.train.run.sr import run_sr_train

    torch.set_num_threads(1)
    cfg = configuration(tmp_path, height=True)
    cfg["conditioning"]["spatial_profile"].update(enabled=True, num_bins=4)
    Image.fromarray(np.zeros((32, 24), dtype=np.uint8)).save(
        tmp_path / "images/second.png"
    )
    trainer = build_trainer(cfg, torch.device("cpu"))
    source = tmp_path / "source"
    source.mkdir()
    save_yaml(source / "train.yaml", trainer.cfg)
    torch.save(trainer.denoiser.state_dict(), source / "generator.pt")
    sr_cfg = configuration(tmp_path, stage="sr", height=True)
    sr_cfg["train"].update(total_steps=2, weights_every_steps=1)
    sr_cfg["lr_bank"].update(samples_per_domain=1, refresh_every_steps=1)
    save_yaml(tmp_path / "sr.yaml", sr_cfg)
    run = run_sr_train(
        config=tmp_path / "sr.yaml",
        base_weights=source / "generator.pt",
        run_dir=tmp_path / "trained",
        device="cpu",
    )
    assert (run / "data_manifest.json").is_file()
    banks = sorted(run.glob("lr_bank/step_*.pt"))
    assert len(banks) == 2
    for path in banks:
        saved = torch.load(path, weights_only=True)
        assert saved["conditions"][0][0]["vf_profile"]["axis"] == "z"
        assert saved["height_extents"][0][0] in (24, 32)
    payload = torch.load(run / "checkpoints/last.pt", weights_only=True)
    assert payload["step"] == 2
    assert payload["data_fingerprint"]


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
    original = copy.deepcopy(cfg)
    extents = infer_height_extents(cfg["data"])
    assert extents == {0: 24}
    assert cfg == original
    cfg["data"]["height_extents"] = extents
    dataset = build_datasets(cfg)[0][1]
    with patch("src.data.dataset.np.random.randint", side_effect=[7, 2]):
        sample = dataset[tmp_path / "images/sample.png"]
    assert sample["height_origin"] == 7
    trainer = build_trainer(cfg, torch.device("cpu"))
    trainer.step(0, transition=0)
    assert trainer.denoiser.height_input.weight.grad.abs().sum() > 0
    batches = trainer.get_batches(0)
    with patch("src.train.trainer.torch.randint", return_value=torch.tensor(0)):
        selected = trainer.sample_real_anchor(batches, volume_size=8)
    assert selected.condition.regions[0].axis != 0
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
    assert "profile/sr_coarse_soft_mae" in metrics.diagnostics
    assert "profile/sr_coarse_label_mae" in metrics.diagnostics
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
