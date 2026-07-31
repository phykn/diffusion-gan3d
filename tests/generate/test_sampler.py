import importlib
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import pytest
import torch

from src.anchor import PlaneAnchor
from src.build import build_models, load_sampler
from src.diffusion import Diffusion
from src.generate.sample import Sampler, find_weights
from src.generate.scale import _make_weight, generate_scaled
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


def test_joint_scale_shares_global_state_latent_and_posterior() -> None:
    model = _TraceModel()
    diffusion = _TraceDiffusion(timesteps=3)
    sampler = _sampler(model, diffusion)

    labels, stats = generate_scaled(
        sampler,
        blocks=(2, 1, 1),
        overlap=2,
        progress=False,
    )

    assert labels.shape == (6, 4, 4)
    assert labels.dtype == torch.uint8
    assert sampler.anchor_enabled is False
    assert stats.block_count == 2
    assert [call.transition for call in diffusion.calls] == [2, 1, 0]
    assert all(call.current_shape == (1, 3, 6, 4, 4) for call in diffusion.calls)
    assert all(call.noise_shape == (1, 3, 6, 4, 4) for call in diffusion.calls[:-1])
    assert diffusion.calls[-1].noise_shape is None

    assert len(model.calls) == 6
    for offset, transition in enumerate((2, 1, 0)):
        left, right = model.calls[offset * 2 : offset * 2 + 2]
        assert left.transition == right.transition == transition
        assert left.latent is right.latent
        assert torch.equal(
            left.current[:, :, -2:, :, :],
            right.current[:, :, :2, :, :],
        )


def test_joint_scale_blends_every_tile_with_partition_weights() -> None:
    grid = (2, 2, 2)
    block_size = 4
    overlap = 2
    stride = block_size - overlap
    shape = tuple(block_size + (count - 1) * stride for count in grid)
    model = _TileValueModel()
    diffusion = _TraceDiffusion(timesteps=1)

    generate_scaled(
        _sampler(model, diffusion),
        blocks=grid,
        overlap=overlap,
        progress=False,
    )

    weight_sum = torch.zeros(shape)
    expected = torch.zeros(shape)
    for value, index in zip(model.values, product(*(range(2) for _ in range(3)))):
        weight = _make_weight(
            index,
            grid=grid,
            block_size=block_size,
            overlap=overlap,
            dtype=torch.float32,
        )
        region = tuple(
            slice(index[axis] * stride, index[axis] * stride + block_size)
            for axis in range(3)
        )
        weight_sum[region].add_(weight)
        expected[region].add_(weight * value)

    assert len(model.values) == 8
    assert torch.allclose(weight_sum, torch.ones(shape))
    clean = diffusion.calls[0].clean
    assert clean.shape == (1, 3, *shape)
    assert torch.allclose(clean[0, 0], expected)
    assert torch.allclose(clean[0, 1], expected)
    assert torch.allclose(clean[0, 2], expected)


def test_joint_scale_supports_anisotropic_grids_and_categorical_output() -> None:
    labels, stats = generate_scaled(
        _sampler(_TraceModel(), _TraceDiffusion(timesteps=1)),
        blocks=(2, 1, 3),
        overlap=2,
        progress=False,
    )

    assert labels.shape == (6, 4, 8)
    assert labels.dtype == torch.uint8
    assert int(labels.max()) < 3
    assert stats.shape == (6, 4, 8)
    assert stats.block_grid == (2, 1, 3)
    assert stats.block_size == 4
    assert stats.overlap == 2
    assert stats.block_count == 6
    assert stats.seams == ((3,), (), (3, 5))


def test_joint_scale_single_block_matches_regular_sampling() -> None:
    sampler = _sampler(_ControlledModel(), Diffusion(3))

    with torch.random.fork_rng():
        torch.manual_seed(2718)
        expected = sampler.generate()
        torch.manual_seed(2718)
        actual, stats = generate_scaled(
            sampler,
            blocks=(1, 1, 1),
            overlap=2,
            progress=False,
        )

    assert torch.equal(actual, expected)
    assert stats.block_count == 1
    assert stats.seams == ((), (), ())


def test_joint_scale_validates_grid_overlap_progress_and_model_output() -> None:
    sampler = _sampler(_TraceModel(), _TraceDiffusion(timesteps=1))

    with pytest.raises(TypeError, match="blocks must be"):
        generate_scaled(sampler, blocks="2", progress=False)
    with pytest.raises(ValueError, match="three positive integers"):
        generate_scaled(sampler, blocks=(2, 0, 1), progress=False)
    with pytest.raises(ValueError, match="half the block size"):
        generate_scaled(sampler, blocks=(2, 1, 1), overlap=3, progress=False)
    with pytest.raises(TypeError, match="progress must be"):
        generate_scaled(sampler, blocks=1, overlap=2, progress=1)

    sampler.model = _WrongShapeModel()
    with pytest.raises(ValueError, match="model output must match"):
        generate_scaled(sampler, blocks=1, overlap=2, progress=False)


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


@dataclass(frozen=True)
class _ModelCall:
    transition: int
    latent: torch.Tensor
    current: torch.Tensor


@dataclass(frozen=True)
class _PosteriorCall:
    transition: int
    current_shape: tuple[int, ...]
    noise_shape: tuple[int, ...] | None
    clean: torch.Tensor


class _TraceModel(torch.nn.Module):
    downsample_factor = 1

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[_ModelCall] = []

    def forward(
        self,
        current: torch.Tensor,
        timestep: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        self.calls.append(
            _ModelCall(
                transition=int(timestep.item()),
                latent=latent,
                current=current.detach().clone(),
            )
        )
        bias = latent.mean(dim=1).reshape(current.shape[0], 1, 1, 1, 1)
        return torch.tanh(0.25 * current + bias)


class _TileValueModel(torch.nn.Module):
    downsample_factor = 1

    def __init__(self) -> None:
        super().__init__()
        self.values: list[float] = []

    def forward(
        self,
        current: torch.Tensor,
        timestep: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        del timestep, latent
        value = -0.7 + 0.2 * len(self.values)
        self.values.append(value)
        return torch.full_like(current, value)


class _ControlledModel(torch.nn.Module):
    downsample_factor = 1

    def forward(
        self,
        current: torch.Tensor,
        timestep: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        shape = (current.shape[0], 1, 1, 1, 1)
        time = timestep.to(current.dtype).reshape(shape)
        style = latent.mean(dim=1).reshape(shape)
        return torch.tanh(0.2 * current + 0.02 * time + 0.1 * style)


class _WrongShapeModel(torch.nn.Module):
    downsample_factor = 1

    def forward(
        self,
        current: torch.Tensor,
        timestep: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        del timestep, latent
        return current[..., :-1]


class _TraceDiffusion:
    def __init__(self, *, timesteps: int) -> None:
        self.timesteps = timesteps
        self.calls: list[_PosteriorCall] = []

    def sample_posterior(
        self,
        current: torch.Tensor,
        clean_prediction: torch.Tensor,
        transition: int,
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls.append(
            _PosteriorCall(
                transition=transition,
                current_shape=tuple(current.shape),
                noise_shape=None if noise is None else tuple(noise.shape),
                clean=clean_prediction.detach().clone(),
            )
        )
        return clean_prediction


def _sampler(model: torch.nn.Module, diffusion: object) -> Sampler:
    return Sampler(
        model,
        diffusion,
        device=torch.device("cpu"),
        patch_size=4,
        num_phases=3,
        latent_channels=4,
        anchor_enabled=False,
        max_anchor_planes=1,
        use_amp=False,
    )


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
