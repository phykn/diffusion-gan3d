import copy
from types import SimpleNamespace

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
    normalize_train_config,
    save_yaml,
)
from src.plane import PLANE_AXES
from src.predict.inference import InferenceAPI
from src.predict.sr import SuperResolutionAPI
from src.prepare.resize import downsample, phase_channels, resize_labels, scaled_size
from src.serve.app import create_app
from src.train.sr_loss import consistency_loss, sample_slices


def sr_config(tmp_path, scale=1.5, phases=3):
    folder = tmp_path / "images"
    folder.mkdir(exist_ok=True)
    labels = (np.indices((24, 24)).sum(0) // 3 % phases).astype(np.uint8)
    Image.fromarray(labels).save(folder / "sample.png")
    return {
        "data": {
            "crop_size": 16,
            "lo_res_size": 8,
            "scale_factor": scale,
            "num_phase": phases,
            "domains": {0: {0: [str(folder)], 2: [str(folder)]}},
            "augment": "anisotropic",
            "augment_prob": 0.5,
        },
        "model": {"channels": 4, "blocks": 1, "noise_channels": 1},
        "critic": {"channels": [4, 8]},
        "optim": {
            "generator_lr": 0.001,
            "critic_lr": 0.001,
            "betas": [0.0, 0.9],
            "ema_decay": 0.9,
        },
        "train": {
            "steps": 2,
            "seed": 0,
            "bank_size": 2,
            "batch_size": 1,
            "critic_steps": 1,
            "slices_per_axis": 2,
            "save_every": 1,
            "amp": False,
            "gradient_penalty": 10.0,
            "lr_weight": 10.0,
            "lr_tolerance": 0.005,
            "temperature": 0.05,
        },
    }


def export_model(path, cfg):
    model = build_sr_model(cfg)
    torch.save(
        {"format": "diffusion-gan3d.sr.v1", "config": cfg, "model": model.state_dict()},
        path,
    )
    return model


@pytest.mark.parametrize(
    "crop,low,scale,high", [(256, 64, 1.5, 96), (128, 64, 2, 128), (256, 64, 4, 256)]
)
def test_independent_crop_and_fractional_scale(crop, low, scale, high):
    assert get_sizes(
        {"crop_size": crop, "lo_res_size": low, "scale_factor": scale}
    ) == (crop, low, high)


@pytest.mark.parametrize(
    "size,scale", [(3, 1.5), (64, 0.5), (True, 2), (64, float("nan")), (64, True)]
)
def test_invalid_scale_never_silently_rounds(size, scale):
    with pytest.raises(ValueError):
        scaled_size(size, scale)


def test_legacy_resolution_remains_readable():
    assert get_sizes({"crop_size": 16, "input_size": 8, "allow_part": True}) == (
        16,
        8,
        8,
    )
    with pytest.raises(ValueError, match="not both"):
        get_sizes(
            {"crop_size": 16, "input_size": 8, "lo_res_size": 8, "scale_factor": 2}
        )


@pytest.mark.parametrize("scale", [1.5, 2, 4])
def test_sr_scale_changes_hr_without_changing_lr_data_or_preparation(tmp_path, scale):
    labels = torch.randint(3, (16, 16), generator=torch.Generator().manual_seed(71))
    Image.fromarray(labels.numpy().astype(np.uint8)).save(tmp_path / "sample.png")
    cfg = load_train_config("config/train/low_res.yaml")
    cfg["data"].update(
        crop_size=16, lo_res_size=8, num_phases=3, domains={0: {"xy": [str(tmp_path)]}}
    )
    assert "scale_factor" not in cfg["data"]
    assert get_sizes(cfg["data"]) == (16, 8, 8)
    dataset = build_datasets(cfg)[0][0]
    expected_lr = resize_labels(labels, 8, 3)
    assert torch.equal(dataset[tmp_path / "sample.png"], expected_lr)

    sr_cfg = load_train_config("config/train/sr.yaml", "sr")
    sr_cfg["data"] = copy.deepcopy(cfg["data"])
    sr_cfg["model"]["generator"]["scale_factor"] = scale
    high_size = int(8 * scale)
    assert get_sr_sizes(sr_cfg) == (16, 8, high_size)
    high_ds = build_datasets(sr_cfg, high=True)[0][0]
    assert torch.equal(
        high_ds[tmp_path / "sample.png"], resize_labels(labels, high_size, 3)
    )
    assert torch.equal(dataset[tmp_path / "sample.png"], expected_lr)
    assert sr_cfg["data"] == cfg["data"]

    api = object.__new__(InferenceAPI)
    api.data = cfg["data"]
    api._crop_size = 16
    api.generator = SimpleNamespace(num_phases=3, patch_size=8)
    assert torch.equal(api.prepare_image(labels), expected_lr)


def test_legacy_lr_snapshot_retains_the_distinct_two_resize_result(tmp_path):
    labels = torch.randint(3, (16, 16), generator=torch.Generator().manual_seed(71))
    Image.fromarray(labels.numpy().astype(np.uint8)).save(tmp_path / "sample.png")
    cfg = load_train_config("config/train/low_res.yaml")
    cfg["data"].update(
        crop_size=16,
        lo_res_size=8,
        scale_factor=1.5,
        num_phases=3,
        domains={0: {"xy": [str(tmp_path)]}},
    )
    save_yaml(tmp_path / "train.yaml", cfg)
    saved = load_train_config(tmp_path / "train.yaml")
    expected = resize_labels(resize_labels(labels, 12, 3), 8, 3)
    assert not torch.equal(expected, resize_labels(labels, 8, 3))
    assert torch.equal(build_datasets(saved)[0][0][tmp_path / "sample.png"], expected)
    api = object.__new__(InferenceAPI)
    api.data = saved["data"]
    api._crop_size = 16
    api.generator = SimpleNamespace(num_phases=3, patch_size=8)
    assert torch.equal(api.prepare_image(labels), expected)


@pytest.mark.parametrize("scale", [None, False, "2", 0.5, float("inf"), 1.51])
def test_invalid_sr_model_scale_is_rejected_before_model_construction(scale):
    cfg = load_train_config("config/train/sr.yaml", "sr")
    cfg["model"]["generator"]["scale_factor"] = scale
    with pytest.raises(ValueError):
        build_sr_model(cfg)


def test_sr_requires_its_own_scale_and_migrates_old_exports(tmp_path):
    cfg = load_train_config("config/train/sr.yaml", "sr")
    del cfg["model"]["generator"]["scale_factor"]
    with pytest.raises(ValueError, match="model.generator.scale_factor"):
        build_sr_model(cfg)
    old = sr_config(tmp_path, scale=1.5)
    migrated = normalize_train_config(old, "sr")
    assert "scale_factor" not in migrated["data"]
    assert migrated["model"]["generator"]["scale_factor"] == 1.5
    assert get_sr_sizes(migrated) == (16, 8, 12)


def test_crop_hr_lr_pipeline_and_phase_ids(tmp_path):
    cfg = sr_config(tmp_path)
    low_ds = build_datasets(cfg)[0][0]
    high_ds = build_datasets(cfg, high=True)[0][0]
    path = low_ds.path_groups[0][0]
    np.random.seed(12)
    low = low_ds[path]
    np.random.seed(12)
    high = high_ds[path]
    assert low.shape == (8, 8) and high.shape == (12, 12)
    assert torch.equal(low, resize_labels(high, 8, 3))
    assert set(low.unique().tolist()).issubset({0, 1, 2})
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
    expected = resize_labels(resize_labels(crop, 12, 3), 8, 3)
    with TestClient(create_app(inference=api)) as client:
        response = client.post("/prepare", json={"image": crop.tolist()})
        assert response.status_code == 200
        assert response.json()["image"] == expected.tolist()
        assert client.post("/prepare", json={"image": [[0]]}).status_code == 422


def test_consistency_has_gradient_and_a_dead_zone():
    logits = torch.randn(1, 3, 12, 12, 12, requires_grad=True)
    low = phase_channels(torch.randint(3, (1, 8, 8, 8)), 3)
    loss, error = consistency_loss(logits.softmax(1), low, 0.05, 0.005)
    loss.backward()
    assert error > 0 and logits.grad.abs().sum() > 0
    constant = torch.zeros(1, 2, 8, 8, 8)
    constant[:, 0] = 1
    loss, _ = consistency_loss(
        torch.nn.functional.interpolate(constant, scale_factor=2), constant, 0.05, 0.005
    )
    assert loss == 0
    assert torch.allclose(
        downsample(constant, (4, 4, 4), 0.05).sum(1), torch.ones(1, 4, 4, 4)
    )


def test_axis_slicing_keeps_channel_and_plane_semantics():
    volume = torch.arange(2 * 3 * 4 * 5 * 6).reshape(2, 3, 4, 5, 6).float()
    for axis, shape in enumerate(((5, 6), (4, 6), (4, 5))):
        planes = sample_slices(volume, axis, 7)
        assert planes.shape == (7, 3, *shape)
        assert torch.equal(planes[:, 1] - planes[:, 0], torch.full((7, *shape), 120.0))


@pytest.mark.parametrize("legacy", [None, "v1", "v2"])
def test_sr_training_restores_state_without_rng(tmp_path, legacy):
    torch.set_num_threads(1)
    cfg = sr_config(tmp_path)
    if legacy:
        axes = cfg["data"]["domains"][0]
        cfg["data"]["domains"][0] = {2: axes[2], 0: axes[0]}
    bank = {0: torch.randint(3, (2, 8, 8, 8), dtype=torch.uint8)}
    trainer = build_sr_trainer(cfg, bank, torch.device("cpu"))
    before = copy.deepcopy(trainer.model.state_dict())
    critic_before = copy.deepcopy(trainer.critics.state_dict())
    first = trainer.train_step()
    assert np.isfinite(list(first.values())).all()
    assert any(
        not torch.equal(before[k], v) for k, v in trainer.model.state_dict().items()
    )
    assert any(
        not torch.equal(critic_before[k], v)
        for k, v in trainer.critics.state_dict().items()
    )
    checkpoint = tmp_path / "last.pt"
    trainer.save(checkpoint)
    expected = copy.deepcopy(trainer.model.state_dict())
    expected_critics = copy.deepcopy(trainer.critics.state_dict())
    restored = build_sr_trainer(cfg, bank, torch.device("cpu"))
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["format"] == "diffusion-gan3d.sr.train.v4"
    assert not {"torch_rng", "cuda_rng", "numpy_rng"} & payload.keys()
    assert set(trainer.critics) == {"0_xy", "0_yz"}
    if legacy:
        payload["format"] = f"diffusion-gan3d.sr.train.{legacy}"
        payload["torch_rng"] = None  # Old RNG fields must be ignored.
        optimizers = payload.pop("critic_optims")
        params = []
        state = {}
        group = None
        for optimizer in optimizers.values():
            group = optimizer["param_groups"][0]
            for key in group["params"]:
                index = len(params)
                params.append(index)
                if key in optimizer["state"]:
                    state[index] = optimizer["state"][key]
        payload["critic_optim"] = {
            "state": state,
            "param_groups": [{**group, "params": params}],
        }
    if legacy == "v1":
        critics = {}
        for key, value in payload["critics"].items():
            name, _, param = key.partition(".")
            domain, plane = name.split("_")
            critics[f"{domain}_{PLANE_AXES[plane]}.{param}"] = value
        payload["critics"] = critics
    rng = torch.get_rng_state().clone()
    restored.resume(payload)
    assert torch.equal(rng, torch.get_rng_state())
    assert restored.step == 1
    torch.testing.assert_close(
        restored.generator_optim.state_dict(), trainer.generator_optim.state_dict()
    )
    for group in trainer.critic_optims:
        torch.testing.assert_close(
            restored.critic_optims[group].state_dict(),
            trainer.critic_optims[group].state_dict(),
        )
    for key, value in expected.items():
        torch.testing.assert_close(
            restored.model.state_dict()[key], value, rtol=0, atol=0
        )
    for key, value in expected_critics.items():
        torch.testing.assert_close(
            restored.critics.state_dict()[key], value, rtol=0, atol=0
        )
    actual_metrics = restored.train_step()
    assert np.isfinite(list(actual_metrics.values())).all()
    assert restored.step == 2
    path = tmp_path / "model.pt"
    restored.export(path)
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
    tiled = api.predict_probs(low, seed=11, tile_size=8, overlap=2)
    assert full.shape == (3, *expected) and tiled.shape == full.shape
    assert torch.isfinite(tiled).all()
    torch.testing.assert_close(tiled.sum(0), torch.ones(expected))
    torch.testing.assert_close(
        tiled, api.predict_probs(low, seed=11, tile_size=8, overlap=2), rtol=0, atol=0
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert not torch.equal(full, api.predict_probs(low, seed=12))


def test_multidomain_and_invalid_inference_contract(tmp_path):
    cfg = sr_config(tmp_path)
    cfg["data"]["domains"][1] = {1: cfg["data"]["domains"][0][0]}
    path = tmp_path / "model.pt"
    export_model(path, cfg)
    api = SuperResolutionAPI(path)
    low = torch.zeros(8, 8, 8, dtype=torch.uint8)
    with pytest.raises(ValueError, match="domain is required"):
        api.super_resolve(low)
    assert api.super_resolve(low, domain=1).shape == (12, 12, 12)
    with pytest.raises(ValueError, match="integer dtype"):
        api.super_resolve(low.float(), domain=0)
    with pytest.raises(ValueError, match="multiples of 2"):
        api.super_resolve(low, domain=0, tile_size=5, overlap=1)
