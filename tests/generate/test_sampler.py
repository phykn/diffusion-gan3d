import importlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from src.anchor import PlaneAnchor
from src.build import build_models, build_sampler, load_sampler
from src.generate.sample import find_weights
from src.generate.scale import _blend, _make_anchors, generate_scaled
from src.train.config import (
    AnchorConfig,
    DataConfig,
    DiffusionConfig,
    LoopConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
)
from src.train.ema import build_ema
from src.train.weights import save_weights
from src.utils import save_yaml


def test_ema_weights_generate_categorical_volume(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    run_dir = tmp_path / "run" / "sample"
    run_dir.mkdir(parents=True)
    save_yaml(run_dir / "train.yaml", cfg.as_dict())
    denoiser, _ = build_models(cfg)
    ema = build_ema(denoiser)
    weights = save_weights(run_dir, ema)

    sampler = load_sampler(weights, device=torch.device("cpu"))
    probabilities = sampler.sample()
    labels = sampler.generate()

    assert find_weights(tmp_path / "run") == weights
    assert probabilities.shape == (3, 8, 8, 8)
    assert torch.allclose(
        probabilities.sum(dim=0),
        torch.ones(8, 8, 8),
    )
    assert labels.shape == (8, 8, 8)
    assert labels.dtype == torch.uint8
    assert int(labels.max()) < cfg.data.num_phases


def test_anchor_aware_weights_accept_soft_plane_condition(
    tmp_path: Path,
) -> None:
    cfg = replace(
        _config(tmp_path),
        anchor=AnchorConfig(
            probability=0.5,
            loss_weight=1.0,
            max_planes=3,
        ),
    )
    run_dir = tmp_path / "run" / "anchored"
    run_dir.mkdir(parents=True)
    save_yaml(run_dir / "train.yaml", cfg.as_dict())
    denoiser, _ = build_models(cfg)
    weights = save_weights(run_dir, build_ema(denoiser))
    sampler = load_sampler(weights, device=torch.device("cpu"))
    anchor = PlaneAnchor(
        labels=torch.randint(
            0,
            cfg.data.num_phases,
            (8, 8),
        ),
        axis=1,
        index=4,
    )

    labels = sampler.generate(
        anchors=(anchor,),
    )

    assert labels.shape == (8, 8, 8)
    assert torch.equal(labels.select(anchor.axis, anchor.index).long(), anchor.labels)


def test_scaled_generation_blends_overlapping_anchor_blocks(
    tmp_path: Path,
) -> None:
    cfg = replace(
        _config(tmp_path),
        anchor=AnchorConfig(
            probability=0.5,
            loss_weight=1.0,
            max_planes=3,
        ),
    )
    denoiser, _ = build_models(cfg)
    sampler = build_sampler(
        cfg,
        denoiser,
        device=torch.device("cpu"),
        mixed_precision=False,
    )

    labels, stats = generate_scaled(
        sampler,
        blocks=(2, 2, 2),
        overlap=4,
        progress=False,
    )

    assert labels.shape == (12, 12, 12)
    assert labels.dtype == torch.uint8
    assert int(labels.max()) < cfg.data.num_phases
    assert stats.block_grid == (2, 2, 2)
    assert stats.block_count == 8
    assert stats.anchor_planes == 12
    assert stats.seams == ((6,), (6,), (6,))


def test_scaled_generation_supports_an_anisotropic_block_grid(
    tmp_path: Path,
) -> None:
    cfg = replace(
        _config(tmp_path),
        anchor=AnchorConfig(probability=0.5, loss_weight=1.0),
    )
    denoiser, _ = build_models(cfg)
    sampler = build_sampler(
        cfg,
        denoiser,
        device=torch.device("cpu"),
        mixed_precision=False,
    )

    with pytest.warns(UserWarning, match="more simultaneous anchor axes"):
        labels, stats = generate_scaled(
            sampler,
            blocks=(2, 2, 1),
            overlap=4,
            progress=False,
        )

    assert labels.shape == (12, 12, 8)
    assert stats.block_grid == (2, 2, 1)
    assert stats.block_count == 4
    assert stats.seams == ((6,), (6,), ())


def test_scaled_generation_rejects_triple_overlap() -> None:
    cfg = replace(
        _config(Path(".")),
        anchor=AnchorConfig(probability=0.5, loss_weight=1.0),
    )
    denoiser, _ = build_models(cfg)
    sampler = build_sampler(
        cfg,
        denoiser,
        device=torch.device("cpu"),
        mixed_precision=False,
    )

    with pytest.raises(ValueError, match="half the block size"):
        generate_scaled(
            sampler,
            blocks=(2, 1, 1),
            overlap=5,
            progress=False,
        )


def test_scale_up_face_anchors_keep_global_axis_orientation() -> None:
    starts = ((0, 4), (0, 4), (0, 4))
    coordinates = torch.meshgrid(
        *(torch.arange(12) for _ in range(3)),
        indexing="ij",
    )
    labels = (coordinates[0] + coordinates[1] + coordinates[2]) % 3
    scores = torch.nn.functional.one_hot(labels, num_classes=3).movedim(-1, 0).float()

    anchors = _make_anchors(
        (1, 1, 1),
        starts=starts,
        scores=scores,
        block_size=8,
    )

    assert tuple(anchor.axis for anchor in anchors) == (0, 1, 2)
    for anchor in anchors:
        origin = (4, 4, 4)
        seam = origin[anchor.axis] + anchor.index
        expected = labels.select(anchor.axis, seam)[4:12, 4:12]
        assert seam == 6
        assert torch.equal(anchor.labels, expected)


def test_scale_up_feathers_phase_probabilities_across_overlap() -> None:
    starts = ((0, 4), (0,), (0,))
    scores = torch.zeros(2, 12, 8, 8)
    left = torch.empty(2, 8, 8, 8)
    left[0].fill_(0.9)
    left[1].fill_(0.1)
    right = 1.0 - left

    _blend(
        scores,
        left,
        index=(0, 0, 0),
        starts=starts,
        grid=(2, 1, 1),
        overlap=4,
    )
    _blend(
        scores,
        right,
        index=(1, 0, 0),
        starts=starts,
        grid=(2, 1, 1),
        overlap=4,
    )

    assert torch.allclose(scores.sum(dim=0), torch.ones(12, 8, 8))
    assert torch.all(scores[0, 4:7] > scores[0, 5:8])
    assert torch.all(scores.argmax(dim=0)[:6] == 0)
    assert torch.all(scores.argmax(dim=0)[6:] == 1)


def test_scale_up_corner_weights_form_a_partition() -> None:
    starts = ((0, 4), (0, 4), (0, 4))
    scores = torch.zeros(2, 12, 12, 12)
    probabilities = torch.empty(2, 8, 8, 8)
    probabilities[0].fill_(0.25)
    probabilities[1].fill_(0.75)
    for index in (
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
    ):
        _blend(
            scores,
            probabilities,
            index=index,
            starts=starts,
            grid=(2, 2, 2),
            overlap=4,
        )

    assert torch.allclose(scores[0], torch.full((12, 12, 12), 0.25))
    assert torch.allclose(scores[1], torch.full((12, 12, 12), 0.75))


def test_scale_quality_skips_axes_without_seams() -> None:
    module = importlib.import_module("scripts.05_check_scale_up")
    labels = torch.arange(12 * 8 * 8, dtype=torch.long).reshape(12, 8, 8) % 3

    quality = module._measure_seams(
        labels.to(torch.uint8),
        ((6,), (), ()),
        4,
        3,
    )

    assert quality.change_ratio[0] is not None
    assert quality.change_ratio[1:] == (None, None)
    assert quality.transition_tv[1:] == (None, None)
    assert quality.continuation_delta[1:] == (None, None)


def _config(root: Path) -> TrainConfig:
    return TrainConfig(
        data=DataConfig(
            folder={axis: root / str(axis) for axis in (0, 1, 2)},
            crop_size=8,
            patch_size=8,
            num_phases=3,
            batch_size=2,
        ),
        model=ModelConfig(
            base_channels=4,
            channel_multipliers=(1, 2),
            embedding_channels=8,
            latent_channels=4,
            critic_channels=(4, 8),
            gradient_checkpointing=False,
        ),
        diffusion=DiffusionConfig(
            timesteps=2,
            beta_min=0.1,
            beta_max=2.0,
        ),
        anchor=AnchorConfig(
            probability=0.0,
            loss_weight=0.0,
        ),
        optim=OptimConfig(
            generator_lr=1e-3,
            critic_lr=1e-3,
            beta1=0.0,
            beta2=0.9,
            r1_gamma=0.0,
            r1_interval=2,
            critic_local_weight=0.5,
        ),
        train=LoopConfig(
            steps=10,
            volume_batch_size=1,
            slices_per_axis=2,
            mixed_precision=False,
            ema_decay=0.9,
            save_every_steps=1,
        ),
    )
