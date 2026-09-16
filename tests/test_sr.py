import copy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.build.data import build_datasets
from src.build.model import build_sr_model
from src.build.sr import build_sr_trainer
from src.config import (
    get_sizes,
    get_sr_sizes,
    load_train_config,
)
from src.data.slice import sample_slices
from src.predict.inference import InferenceAPI
from src.predict.sr import SuperResolutionAPI
from src.prepare.resize import (
    downsample,
    phase_channels,
    resize_crop,
    resize_labels,
    scaled_size,
)
from src.serve.app import create_app
from src.train.loss.sr import consistency_loss
from src.train.sr import corrupt_coarse, export_sr, resume_sr_training, save_sr_training
from src.train.trainer import Trainer


def sr_config(tmp_path, scale=1.5, phases=3):
    folder = tmp_path / "images"
    folder.mkdir(exist_ok=True)
    labels = (np.indices((24, 24)).sum(0) // 3 % phases).astype(np.uint8)
    Image.fromarray(labels).save(folder / "sample.png")
    cfg = load_train_config("config/train/sr.yaml", "sr")
    cfg["data"].update(
        crop_size=16,
        lo_res_size=8,
        hi_res_size=int(8 * scale),
        num_phases=phases,
        domains={0: {"xy": [str(folder)], "yz": [str(folder)]}},
    )
    cfg["model"]["generator"].update(
        channels=[4, 8], embedding_channels=8, latent_channels=4
    )
    cfg["model"]["diffusion"]["num_steps"] = 2
    cfg["model"]["gradient_checkpointing"] = False
    cfg["model"]["critic"].update(channels=[4, 8], plane_groups=[["xy"], ["yz"]])
    cfg["optim"].update(
        generator_lr=0.001, critic_lr=0.001, adam_betas=[0.0, 0.9], ema_decay=0.9
    )
    cfg["train"].update(
        total_steps=2,
        volume_batch_size=1,
        real_batch_size=2,
        slice_pairs_per_plane=2,
        checkpoint_every_steps=1,
        mixed_precision=False,
    )
    cfg["lr_bank"]["samples_per_domain"] = 2
    return cfg


def export_model(path, cfg):
    model = build_sr_model(cfg)
    torch.save(
        {"format": "diffusion-gan3d.sr", "config": cfg, "model": model.state_dict()},
        path,
    )
    return model


@pytest.mark.parametrize(
    "crop,low,scale,high", [(256, 64, 1.5, 96), (128, 64, 2, 128), (256, 64, 4, 256)]
)
def test_independent_crop_and_fractional_scale(crop, low, scale, high):
    assert get_sizes({"crop_size": crop, "lo_res_size": low, "hi_res_size": high}) == (
        crop,
        low,
        high,
    )


@pytest.mark.parametrize(
    "size,scale", [(3, 1.5), (64, 0.5), (True, 2), (64, float("nan")), (64, True)]
)
def test_invalid_scale_never_silently_rounds(size, scale):
    with pytest.raises(ValueError):
        scaled_size(size, scale)


def test_old_resolution_keys_are_rejected():
    with pytest.raises(ValueError, match="lo_res_size"):
        get_sizes({"crop_size": 16, "input_size": 8, "allow_part": True})


@pytest.mark.parametrize("scale", [1.5, 2, 4])
def test_sr_scale_changes_hr_without_changing_lr_data_or_preparation(tmp_path, scale):
    labels = torch.randint(3, (16, 16), generator=torch.Generator().manual_seed(71))
    Image.fromarray(labels.numpy().astype(np.uint8)).save(tmp_path / "sample.png")
    cfg = load_train_config("config/train/low_res.yaml")
    cfg["data"].update(
        crop_size=16,
        lo_res_size=8,
        hi_res_size=8,
        num_phases=3,
        domains={0: {"xy": [str(tmp_path)]}},
    )
    assert "scale_factor" not in cfg["data"]
    assert get_sizes(cfg["data"]) == (16, 8, 8)
    dataset = build_datasets(cfg)[0][0]
    expected_lr = resize_crop(labels, 8, 3)
    assert torch.equal(dataset[tmp_path / "sample.png"], expected_lr)

    sr_cfg = load_train_config("config/train/sr.yaml", "sr")
    sr_cfg["data"] = copy.deepcopy(cfg["data"])
    sr_cfg["data"]["hi_res_size"] = int(8 * scale)
    high_size = int(8 * scale)
    assert get_sr_sizes(sr_cfg) == (16, 8, high_size)
    high_ds = build_datasets(sr_cfg, high=True)[0][0]
    assert torch.equal(
        high_ds[tmp_path / "sample.png"], resize_crop(labels, high_size, 3)
    )
    assert torch.equal(dataset[tmp_path / "sample.png"], expected_lr)
    assert sr_cfg["data"]["lo_res_size"] == cfg["data"]["lo_res_size"]

    api = object.__new__(InferenceAPI)
    api.data = cfg["data"]
    api._crop_size = 16
    api.generator = SimpleNamespace(num_phases=3, patch_size=8)
    assert torch.equal(api.prepare_image(labels), expected_lr)


@pytest.mark.parametrize("high", [None, False, "128", 32, float("inf"), 96.5])
def test_invalid_sr_grid_is_rejected_before_model_construction(high):
    cfg = load_train_config("config/train/sr.yaml", "sr")
    cfg["data"]["hi_res_size"] = high
    with pytest.raises(ValueError):
        build_sr_model(cfg)


def test_sr_requires_its_own_grid(tmp_path):
    cfg = load_train_config("config/train/sr.yaml", "sr")
    del cfg["data"]["hi_res_size"]
    with pytest.raises(ValueError, match="data.hi_res_size"):
        build_sr_model(cfg)


def test_crop_hr_lr_pipeline_and_phase_ids(tmp_path):
    cfg = sr_config(tmp_path)
    low_ds = build_datasets(cfg)[0][0]
    high_ds = build_datasets(cfg, high=True)[0][0]
    path = low_ds.path_groups[0][0]
    np.random.seed(12)
    low = low_ds[path]
    np.random.seed(12)
    high = high_ds[path]
    assert low.shape == (3, 8, 8) and high.shape == (3, 12, 12)
    torch.testing.assert_close(low.sum(0), torch.ones(8, 8))
    torch.testing.assert_close(high.sum(0), torch.ones(12, 12))
    assert torch.all(low[1] >= 0)
    labels = torch.tensor([[0, 2], [2, 2]])
    assert (
        resize_labels(labels, 1, 3).item() == 2
    )  # Averaging IDs would invent phase 1.
    cfg["data"]["crop_size"] = 32
    ds = build_datasets(cfg)[0][0]
    with pytest.raises(ValueError, match="crop size must fit"):
        ds[path]


def test_web_anchor_preparation_matches_new_dataset(tmp_path):
    cfg = sr_config(tmp_path)
    api = object.__new__(InferenceAPI)
    api.data = cfg["data"]
    api._crop_size = 16
    api.generator = SimpleNamespace(num_phases=3, patch_size=8)
    crop = torch.arange(16 * 16).reshape(16, 16) % 3
    expected = resize_crop(crop, 8, 3)
    with TestClient(create_app(inference=api)) as client:
        response = client.post("/prepare", json={"image": crop.tolist()})
        assert response.status_code == 200
        assert response.json()["image"] == expected.tolist()
        assert client.post("/prepare", json={"image": [[0]]}).status_code == 422


def test_consistency_has_gradient_and_a_dead_zone():
    logits = torch.randn(1, 3, 12, 12, 12, requires_grad=True)
    low = phase_channels(torch.randint(3, (1, 8, 8, 8)), 3)
    loss, error = consistency_loss(logits.softmax(1), low, 0.005)
    loss.backward()
    assert error > 0 and logits.grad.abs().sum() > 0
    constant = torch.zeros(1, 2, 8, 8, 8)
    constant[:, 0] = 1
    loss, _ = consistency_loss(
        torch.nn.functional.interpolate(constant, scale_factor=2), constant, 0.005
    )
    assert loss == 0
    assert torch.allclose(
        downsample(constant, (4, 4, 4)).sum(1), torch.ones(1, 4, 4, 4)
    )


def test_axis_slicing_keeps_channel_and_plane_semantics():
    volume = torch.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6).float()
    for axis, shape in enumerate(((5, 6), (4, 6), (4, 5))):
        planes = sample_slices(volume, axis, 7)
        assert planes.shape == (7, 3, *shape)
        assert torch.equal(planes[:, 1] - planes[:, 0], torch.full((7, *shape), 120.0))


def test_sr_training_restores_state_without_rng(tmp_path):
    torch.set_num_threads(1)
    cfg = sr_config(tmp_path)
    bank = {0: torch.rand(2, 3, 8, 8, 8).softmax(1)}
    trainer = build_sr_trainer(cfg, bank, torch.device("cpu"))
    before = copy.deepcopy(trainer.denoiser.state_dict())
    critic_before = copy.deepcopy(trainer.critics.state_dict())
    assert type(trainer) is Trainer
    first = trainer.step(0)
    assert np.isfinite(first.generator_total)
    assert any(
        not torch.equal(before[k], v) for k, v in trainer.denoiser.state_dict().items()
    )
    assert any(
        not torch.equal(critic_before[k], v)
        for k, v in trainer.critics.state_dict().items()
    )
    checkpoint = tmp_path / "last.pt"
    save_sr_training(trainer, checkpoint)
    expected = copy.deepcopy(trainer.denoiser.state_dict())
    expected_critics = copy.deepcopy(trainer.critics.state_dict())
    restored = build_sr_trainer(cfg, bank, torch.device("cpu"))
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["format"] == "diffusion-gan3d.sr.train"
    assert not {"torch_rng", "cuda_rng", "numpy_rng"} & payload.keys()
    assert set(trainer.critics) == {"0_xy", "0_yz"}
    rng = torch.get_rng_state().clone()
    resume_sr_training(restored, payload)
    assert torch.equal(rng, torch.get_rng_state())
    assert restored.completed_steps == 1
    torch.testing.assert_close(
        restored.denoiser_optim.state_dict(), trainer.denoiser_optim.state_dict()
    )
    for group in trainer.critic_optims:
        torch.testing.assert_close(
            restored.critic_optims[group].state_dict(),
            trainer.critic_optims[group].state_dict(),
        )
    for key, value in expected.items():
        torch.testing.assert_close(
            restored.denoiser.state_dict()[key], value, rtol=0, atol=0
        )
    for key, value in expected_critics.items():
        torch.testing.assert_close(
            restored.critics.state_dict()[key], value, rtol=0, atol=0
        )
    actual_metrics = restored.step(restored.completed_steps)
    assert np.isfinite(actual_metrics.generator_total)
    assert restored.completed_steps == 2
    path = tmp_path / "model.pt"
    export_sr(restored, path)
    api = SuperResolutionAPI(path)
    assert api.super_resolve(bank[0][0]).shape == (12, 12, 12)


@pytest.mark.parametrize("scale", [1.5, 2, 4])
def test_predict_fractional_shape_seed_and_tiled_coverage(tmp_path, scale):
    torch.set_num_threads(1)
    cfg = sr_config(tmp_path, scale)
    path = tmp_path / "model.pt"
    export_model(path, cfg)
    api = SuperResolutionAPI(path)
    low = torch.randint(3, (12, 14, 16))
    rng = torch.get_rng_state().clone()
    expected = tuple(int(n * scale) for n in low.shape)
    full = api.predict_probs(low, seed=11)
    if scale == 1.5:
        with pytest.raises(ValueError, match="integer HR/LR scale"):
            api.predict_probs(low, seed=11, tile_size=8, overlap=2)
        tiled = api.predict_probs(low, seed=11, tile_size=24, overlap=2)
    else:
        tiled = api.predict_probs(low, seed=11, tile_size=16, overlap=4)
    assert full.shape == (3, *expected) and tiled.shape == full.shape
    assert torch.isfinite(tiled).all()
    torch.testing.assert_close(tiled.sum(0), torch.ones(expected))
    torch.testing.assert_close(
        tiled,
        api.predict_probs(
            low,
            seed=11,
            tile_size=24 if scale == 1.5 else 16,
            overlap=2 if scale == 1.5 else 4,
        ),
        rtol=0,
        atol=0,
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert not torch.equal(full, api.predict_probs(low, seed=12))


def test_multidomain_and_invalid_inference_contract(tmp_path):
    cfg = sr_config(tmp_path)
    cfg["data"]["domains"][1] = {"xy": cfg["data"]["domains"][0]["xy"]}
    path = tmp_path / "model.pt"
    export_model(path, cfg)
    api = SuperResolutionAPI(path)
    low = torch.zeros(8, 8, 8, dtype=torch.uint8)
    with pytest.raises(ValueError, match="domain is required"):
        api.super_resolve(low)
    assert api.super_resolve(low, domain=1).shape == (12, 12, 12)
    with pytest.raises(ValueError, match="integer dtype"):
        api.super_resolve(low.float(), domain=0)
    with pytest.raises(ValueError, match="integer HR/LR scale"):
        api.super_resolve(low, domain=0, tile_size=5, overlap=1)


@pytest.mark.parametrize("fractions", [False, True])
def test_sr_rejects_memory_budget_before_converting_coarse_or_predicting(
    tmp_path, monkeypatch, fractions
):
    from types import SimpleNamespace

    import src.predict.sr as sr_module
    from src.predict import sr_memory

    cfg = sr_config(tmp_path, scale=2)
    path = tmp_path / "model.pt"
    export_model(path, cfg)
    api = SuperResolutionAPI(path)
    low = torch.zeros(8, 8, 8, dtype=torch.uint8)
    if fractions:
        low = torch.ones(3, 8, 8, 8) / 3
    monkeypatch.setattr(
        sr_memory.psutil, "virtual_memory", lambda: SimpleNamespace(available=1)
    )

    def fail(*args, **kwargs):
        pytest.fail("allocated before SR preflight")

    monkeypatch.setattr(sr_module, "phase_channels", fail)
    monkeypatch.setattr(sr_module, "resize_phases", fail)
    monkeypatch.setattr(api, "_predict", fail)
    with pytest.raises(MemoryError, match="SR RAM"):
        api.predict_probs(low)


def test_sr_tiled_refinement_uses_bounded_shared_fusion(tmp_path, monkeypatch):
    import src.predict.tiled as tiled_module

    cfg = sr_config(tmp_path, scale=2)
    path = tmp_path / "model.pt"
    export_model(path, cfg)
    api = SuperResolutionAPI(path)
    low = torch.zeros(8, 8, 8, dtype=torch.uint8)
    buffers = []
    original = tiled_module.make_fusion

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        buffers.append(result)
        return result

    monkeypatch.setattr(tiled_module, "make_fusion", record)
    monkeypatch.setattr(
        api.generator,
        "predict",
        lambda values, *args, **kwargs: torch.full_like(values, -1 / 3),
    )
    result = api.predict_probs(low, tile_size=8, overlap=2, margin=2)
    assert len(buffers) == 1 and buffers[0].pred_sum.shape == (1, 3, 12, 20, 20)
    torch.testing.assert_close(result, torch.full_like(result, 1 / 3))


@pytest.mark.parametrize("tiled", [False, True])
def test_sr_labels_match_probabilities_and_use_correct_budget(
    tmp_path, monkeypatch, tiled
):
    import src.predict.sr as sr_module

    cfg = sr_config(tmp_path, scale=2)
    path = tmp_path / "model.pt"
    export_model(path, cfg)
    api = SuperResolutionAPI(path)
    low = torch.randint(3, (6, 6, 6))
    options = dict(seed=13, margin=2, tile_size=8 if tiled else None, overlap=2)
    expected = api.predict_probs(low, **options).argmax(0).to(torch.uint8)
    original = sr_module.estimate_sr_memory
    modes = []

    def record(*args, **kwargs):
        modes.append(kwargs["output_kind"])
        return original(*args, **kwargs)

    monkeypatch.setattr(sr_module, "estimate_sr_memory", record)
    if not tiled:

        def fail(*args, **kwargs):
            pytest.fail("single-block labels materialized CPU probabilities")

        monkeypatch.setattr(api, "_predict", fail)
        monkeypatch.setattr(api, "predict_probs", fail)
    actual = api.super_resolve(low, **options)
    assert modes == ["labels"]
    torch.testing.assert_close(actual, expected)


def test_sr_corruption_does_not_modify_clean_coarse_target():
    low = torch.zeros(8, 3, 8, 8, 8)
    low[:, 0] = 1
    original = low.clone()
    corrupted, level = corrupt_coarse(low, 1, 1)
    assert torch.equal(low, original)
    assert not torch.equal(corrupted, low)
    torch.testing.assert_close(corrupted.sum(1), torch.ones_like(corrupted[:, 0]))


def test_sr_loss_targets_clean_coarse_while_model_receives_corrupted_input(tmp_path):
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(tmp_path / "sample.png")
    cfg = load_train_config("config/train/sr.yaml", "sr")
    cfg["data"].update(
        crop_size=8, lo_res_size=8, hi_res_size=8, domains={0: {"xy": [str(tmp_path)]}}
    )
    cfg["model"]["generator"].update(
        channels=[4, 8], embedding_channels=8, latent_channels=4
    )
    cfg["model"]["diffusion"]["num_steps"] = 2
    cfg["model"]["gradient_checkpointing"] = False
    cfg["model"]["critic"].update(channels=[4, 8], plane_groups=[["xy"]])
    cfg["train"].update(
        mixed_precision=False, real_batch_size=1, slice_pairs_per_plane=1
    )
    trainer = build_sr_trainer(
        cfg,
        {
            0: torch.nn.functional.one_hot(torch.zeros(2, 8, 8, 8, dtype=torch.long), 2)
            .movedim(-1, 1)
            .float()
        },
        torch.device("cpu"),
    )
    with (
        patch(
            "src.train.trainer.corrupt_coarse",
            side_effect=lambda low, *args: (low.flip(1), low.new_ones(len(low))),
        ),
        patch.object(
            trainer.denoiser, "compute_logits", wraps=trainer.denoiser.compute_logits
        ) as forward,
        patch(
            "src.train.trainer.consistency_loss", wraps=consistency_loss
        ) as consistency,
    ):
        trainer.step(0)
    assert all(
        call.kwargs["coarse"][:, 1].eq(1).all() for call in forward.call_args_list
    )
    assert consistency.call_args.args[1][:, 0].eq(1).all()


def test_fractional_consistency_does_not_sharpen_coarse():
    low = torch.empty(1, 2, 4, 4, 4)
    low[:, 0], low[:, 1] = 0.3, 0.7
    high = torch.nn.functional.interpolate(low, scale_factor=2).requires_grad_()
    loss, error = consistency_loss(high, low, 0)
    assert float(loss.detach()) < 1e-12 and float(error.detach()) < 1e-12


@pytest.mark.parametrize("guidance,count", [(0.0, 1), (1.0, 1), (1.7, 2)])
def test_sr_cfg_keeps_coarse_and_height_in_both_branches(tmp_path, guidance, count):
    from src.model.layers import NULL_DOMAIN

    cfg = sr_config(tmp_path, scale=2)
    model = build_sr_model(cfg)
    current = torch.randn(1, 3, 16, 16, 16)
    coarse = torch.rand_like(current).softmax(1)
    height = torch.rand(1, 1, 16, 16, 16)
    level = torch.tensor([0.2])
    with patch.object(
        model, "compute_logits", return_value=torch.zeros_like(current)
    ) as logits:
        model.apply_guidance_logits(
            current,
            torch.tensor([1]),
            torch.zeros(1, 4),
            guidance,
            torch.tensor([0]),
            coarse=coarse,
            height=height,
            corruption_level=level,
        )
    assert logits.call_count == count
    for call in logits.call_args_list:
        assert call.kwargs["coarse"] is coarse
        assert call.kwargs["height"] is height
        assert call.kwargs["corruption_level"] is level
    domains = [int(call.args[3].item()) for call in logits.call_args_list]
    assert domains == (
        [0] if guidance == 1 else [NULL_DOMAIN] if guidance == 0 else [NULL_DOMAIN, 0]
    )


@pytest.mark.parametrize(
    "condition", ["vf", "vf_present", "anchor_image", "anchor_mask"]
)
def test_sr_rejects_untrained_anchor_and_vf_conditions(tmp_path, condition):
    model = build_sr_model(sr_config(tmp_path))
    current = torch.zeros(1, 3, 12, 12, 12)
    for guidance in (0, 1, 2):
        with pytest.raises(
            ValueError, match="do not accept anchors or volume fractions"
        ):
            model.apply_guidance_logits(
                current,
                torch.tensor([1]),
                torch.zeros(1, 4),
                guidance,
                torch.tensor([0]),
                coarse=current,
                **{condition: torch.ones(1)},
            )
    with pytest.raises(ValueError, match="requires coarse"):
        model(current, torch.tensor([1]), torch.zeros(1, 4), torch.tensor([0]))


@pytest.mark.parametrize("start", [(3, 6, 3), (-3, 0, 0), (12, 12, 12)])
def test_coarse_halo_matches_global_trilinear_including_edges(start):
    from src.prepare.resize import coarse_region

    low = torch.rand(1, 3, 6, 7, 8).softmax(1)
    full = torch.nn.functional.interpolate(
        low, scale_factor=3, mode="trilinear", align_corners=False
    )
    padded = torch.nn.functional.pad(full, (6,) * 6, mode="replicate")
    shape = (9, 9, 12)
    region = (
        slice(None),
        slice(None),
        *(slice(6 + s, 6 + s + n) for s, n in zip(start, shape)),
    )
    torch.testing.assert_close(
        coarse_region(low, start, shape, 3), padded[region], atol=2e-7, rtol=1e-6
    )


def test_tiled_coarse_and_height_share_global_coordinates_with_margin(tmp_path):
    cfg = sr_config(tmp_path, scale=2)
    cfg["conditioning"]["height_enabled"] = True
    cfg["data"]["thickness_axis"] = "z"
    cfg["data"]["height_extents"] = {0: 40}
    path = tmp_path / "sr.pt"
    export_model(path, cfg)
    api = SuperResolutionAPI(path)
    low = torch.rand(3, 12, 8, 8).softmax(0)
    heights = []

    def identity(values, time, latent, **conditions):
        heights.append(conditions["height"].clone())
        return 2 * conditions["coarse"] - 1

    with patch.object(api.generator, "predict", side_effect=identity):
        actual = api.predict_probs(
            low, tile_size=16, overlap=4, margin=2, height_origin=3
        )
    expected = torch.nn.functional.interpolate(
        low[None], scale_factor=2, mode="trilinear", align_corners=False
    )[0]
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=0)
    assert len(heights) == 2 * api.generator.diffusion.timesteps
    torch.testing.assert_close(heights[0][:, :, 8:], heights[1][:, :, :12])
    assert heights[0][0, 0, 0, 0, 0] == pytest.approx(2 * (3 - 2 + 0.5) / 40 - 1)


@pytest.mark.parametrize("guidance", [1.0, 1.7])
def test_sr_tiles_share_current_latent_and_update_each_voxel_once(
    tmp_path, monkeypatch, guidance
):
    from src.predict.tile import VolumeState
    from src.predict.tiled import TiledGenerator

    cfg = sr_config(tmp_path, scale=2)
    cfg["model"]["diffusion"]["num_steps"] = 3
    path = tmp_path / "sr.pt"
    export_model(path, cfg)
    api = SuperResolutionAPI(path)
    calls, writes = [], {}
    active = None
    original_step, original_write = TiledGenerator.step, VolumeState.write

    def step(sampler, *args, **kwargs):
        nonlocal active
        active = int(args[3].item())
        writes[active] = torch.zeros(16, 12, 12, dtype=torch.int32)
        original_step(sampler, *args, **kwargs)
        active = None

    def write(state, region, values):
        if active is not None:
            writes[active][region] += 1
        original_write(state, region, values)

    def predict(values, time, latent, **conditions):
        assert conditions.get("vf") is None
        assert conditions.get("anchor_image") is None
        assert conditions["corruption_level"].eq(0).all()
        assert conditions["guidance"] == guidance
        calls.append(
            (
                int(time.item()),
                values.clone(),
                latent.clone(),
                conditions["coarse"].clone(),
            )
        )
        return 2 * (values + conditions["coarse"]).softmax(1) - 1

    def forbidden(*args, **kwargs):
        pytest.fail("an independent tile reverse chain was started")

    monkeypatch.setattr(TiledGenerator, "step", step)
    monkeypatch.setattr(VolumeState, "write", write)
    monkeypatch.setattr(api.generator, "predict", predict)
    monkeypatch.setattr(api.generator.diffusion, "sample", forbidden)
    low = torch.rand(3, 6, 4, 4).softmax(0)
    output = api.predict_probs(low, tile_size=8, overlap=2, margin=2, guidance=guidance)
    assert [call[0] for call in calls] == [2, 2, 1, 1, 0, 0]
    for first, second in zip(calls[::2], calls[1::2]):
        torch.testing.assert_close(
            first[1][:, :, 4:], second[1][:, :, :8], atol=0, rtol=0
        )
        torch.testing.assert_close(first[2], second[2], atol=0, rtol=0)
        torch.testing.assert_close(first[3][:, :, 4:], second[3][:, :, :8])
    assert all(
        torch.equal(counts, torch.ones_like(counts)) for counts in writes.values()
    )
    torch.testing.assert_close(output.sum(0), torch.ones(12, 8, 8))


@pytest.mark.parametrize("tile,overlap,margin", [(10, 2, 4), (12, 1, 4), (16, 4, 1)])
def test_tiled_sr_rejects_misaligned_hr_lattice(tmp_path, tile, overlap, margin):
    path = tmp_path / "sr.pt"
    export_model(path, sr_config(tmp_path, scale=4))
    api = SuperResolutionAPI(path)
    with pytest.raises(ValueError, match="multiples of the LR/HR scale"):
        api.predict_probs(
            torch.zeros(8, 8, 8, dtype=torch.uint8),
            tile_size=tile,
            overlap=overlap,
            margin=margin,
        )


def test_sr_runs_full_reverse_chain_and_training_uses_matching_hr_slices(tmp_path):
    cfg = sr_config(tmp_path, scale=1.5)
    cfg["model"]["diffusion"]["num_steps"] = 3
    bank = {0: torch.rand(2, 3, 8, 8, 8).softmax(1)}
    trainer = build_sr_trainer(cfg, bank, torch.device("cpu"))
    with patch.object(
        trainer.denoiser, "compute_logits", wraps=trainer.denoiser.compute_logits
    ) as forward:
        metrics = trainer.step(0, transition=0)
    assert [int(call.args[1][0]) for call in forward.call_args_list] == [2, 1, 0]
    assert all(
        call.kwargs["coarse"].shape == (1, 3, 12, 12, 12)
        for call in forward.call_args_list
    )
    assert all(
        call.kwargs.get("vf") is None and call.kwargs.get("anchor_image") is None
        for call in forward.call_args_list
    )
    assert all(
        stream.next().shape[-2:] == (12, 12) for stream in trainer.streams[0].values()
    )
    assert not metrics.vf_active and metrics.anchor_planes == 0
    assert trainer.denoiser.coarse_input.weight.grad.abs().sum() > 0
    path = tmp_path / "sr.pt"
    export_sr(trainer, path)
    api = SuperResolutionAPI(path)
    with patch.object(api.model, "forward", wraps=api.model.forward) as forward:
        api.predict_probs(bank[0][0], margin=0)
    assert [int(call.args[1][0]) for call in forward.call_args_list] == [2, 1, 0]
    assert all(
        call.kwargs["corruption_level"].eq(0).all() for call in forward.call_args_list
    )


def test_wrong_artifact_kind_and_explicit_scale_setting_are_rejected(tmp_path):
    cfg = sr_config(tmp_path)
    cfg["model"]["generator"]["scale_factor"] = 1.5
    with pytest.raises(ValueError, match="unknown training setting"):
        build_sr_model(cfg)
    path = tmp_path / "training.pt"
    torch.save({"format": "diffusion-gan3d.sr.train"}, path)
    with pytest.raises(ValueError, match="use exported SR weights"):
        SuperResolutionAPI(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_sr_cuda_amp_checkpointing_and_cfg_inference(tmp_path):
    cfg = sr_config(tmp_path, scale=2)
    cfg["train"]["mixed_precision"] = True
    cfg["model"]["gradient_checkpointing"] = True
    cfg["loss"].update(r1_weight=0.1, r1_every_steps=1)
    trainer = build_sr_trainer(
        cfg, {0: torch.rand(2, 3, 8, 8, 8).softmax(1)}, torch.device("cuda")
    )
    trainer.scaler = torch.amp.GradScaler("cuda", init_scale=128)
    for step, transition in enumerate((1, 0)):
        metrics = trainer.step(step, transition=transition)
        assert np.isfinite(metrics.generator_total)
        assert np.isfinite(metrics.diagnostics["gradient/generator"])
        sensitivities = {
            key: value
            for key, value in metrics.diagnostics.items()
            if key.startswith("generator_input_gradient/")
        }
        assert sensitivities and all(
            np.isfinite(value) for value in sensitivities.values()
        )
        assert all(f"/t{transition}/" in key for key in sensitivities)
        assert any(
            value > 0
            for key, value in sensitivities.items()
            if key.endswith("/previous")
        )
    assert trainer.updates["generator"] == 2
    assert all(trainer.updates[group] == 2 for group in trainer.critics)
    path = tmp_path / "sr.pt"
    export_sr(trainer, path)
    api = SuperResolutionAPI(path, "cuda")
    for guidance in (1.0, 1.7):
        probs = api.predict_probs(trainer.bank[0][0], guidance=guidance, margin=0)
        assert torch.isfinite(probs).all()
        torch.testing.assert_close(
            probs.sum(0), torch.ones(16, 16, 16), atol=2e-6, rtol=0
        )
    tiled = api.predict_probs(
        trainer.bank[0][0], guidance=1.7, tile_size=12, overlap=2, margin=2
    )
    assert torch.isfinite(tiled).all()
    torch.testing.assert_close(tiled.sum(0), torch.ones(16, 16, 16), atol=2e-6, rtol=0)
