import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise, product

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .anchor import PlaneAnchor, build_anchors
from .diffusion import Diffusion
from .model.denoiser import Denoiser3D


@dataclass(frozen=True)
class ScaleStats:
    overlap: int
    block_count: int
    seams: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class Tile:
    region: tuple[slice, slice, slice]
    weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class Base:
    clean: torch.Tensor
    noise: torch.Tensor
    tile: Tile
    weight: torch.Tensor


class Generator:
    def __init__(
        self,
        model: Denoiser3D,
        diffusion: Diffusion,
        device: torch.device,
        patch_size: int,
        num_phases: int,
        latent_channels: int,
        anchor_enabled: bool,
        use_amp: bool,
    ) -> None:
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.patch_size = patch_size
        self.num_phases = num_phases
        self.latent_channels = latent_channels
        self.anchor_enabled = anchor_enabled
        self.use_amp = use_amp

    def prepare_vf(
        self,
        vf: Sequence[float] | None,
    ) -> torch.Tensor | None:
        if vf is None:
            return None
        vf = torch.as_tensor(
            vf,
            device=self.device,
            dtype=torch.float32,
        )
        if vf.shape != (self.num_phases,):
            raise ValueError(f"vf must have shape [{self.num_phases}].")
        vf_sum = vf.sum()
        if vf_sum == 0:
            raise ValueError("vf sum must not be zero.")
        return vf.div(vf_sum).unsqueeze(0)

    @torch.no_grad()
    def generate_probs(
        self,
        anchors: Sequence[PlaneAnchor] = (),
        vf: Sequence[float] | None = None,
        size: int | None = None,
    ) -> torch.Tensor:
        size = self.patch_size if size is None else size
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("size must be a positive integer.")
        vf = self.prepare_vf(vf)
        initial_noise = torch.randn(
            1,
            self.num_phases,
            size,
            size,
            size,
            device=self.device,
            dtype=torch.float32,
        )
        anchor = build_anchors(
            anchors,
            batch_size=1,
            num_phases=self.num_phases,
            volume_size=size,
            device=self.device,
            dtype=initial_noise.dtype,
        )
        if anchor is not None and not self.anchor_enabled:
            raise ValueError("selected weights were trained with anchors disabled.")

        conditions = {}
        if anchor is not None:
            conditions.update(
                {
                    "anchor_image": anchor.image,
                    "anchor_mask": anchor.mask,
                }
            )
        if vf is not None:
            conditions["vf"] = vf
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            clean = self.diffusion.sample(
                self.model,
                initial_noise,
                self.latent_channels,
                conditions=conditions or None,
            )
        probs = (clean.float() + 1.0).mul_(0.5).clamp_(0.0, 1.0)
        probs.div_(
            probs.sum(dim=1, keepdim=True).clamp_min_(torch.finfo(probs.dtype).eps)
        )
        return probs.squeeze(0).cpu()

    def generate(
        self,
        anchors: Sequence[PlaneAnchor] = (),
        vf: Sequence[float] | None = None,
        size: int | None = None,
    ) -> torch.Tensor:
        probs = self.generate_probs(
            anchors=anchors,
            vf=vf,
            size=size,
        )
        return probs.argmax(dim=0).to(torch.uint8)


class ScaledGenerator:
    def __init__(self, generator: Generator) -> None:
        self.generator = generator
        self.stats: ScaleStats | None = None

    def prepare_vf(
        self,
        vf: Sequence[float] | None,
    ) -> torch.Tensor | None:
        return self.generator.prepare_vf(vf)

    @torch.no_grad()
    def generate_probs(
        self,
        blocks: int | Sequence[int],
        base: torch.Tensor | None = None,
        vf: Sequence[float] | None = None,
        overlap: int | None = None,
        progress: bool = True,
    ) -> torch.Tensor:
        self.stats = None
        generator = self.generator
        grid = self.parse_grid(blocks)
        block_size = generator.patch_size
        overlap = block_size // 2 if overlap is None else overlap
        if (
            not isinstance(overlap, int)
            or isinstance(overlap, bool)
            or not 1 <= overlap <= block_size // 2
        ):
            raise ValueError("overlap must be between 1 and half the block size.")
        if not isinstance(progress, bool):
            raise TypeError("progress must be a boolean.")
        vf = self.prepare_vf(vf)

        stride = block_size - overlap
        starts = tuple(tuple(idx * stride for idx in range(count)) for count in grid)
        shape = tuple(block_size + (count - 1) * stride for count in grid)
        axis_weights = self.make_axis_weights(
            block_size,
            overlap,
            device=generator.device,
        )
        tiles = self.make_tiles(
            starts,
            grid=grid,
            block_size=block_size,
            axis_weights=axis_weights,
        )
        base = self.prepare_base(base, shape, grid, overlap)

        initial_noise = torch.randn(
            1,
            generator.num_phases,
            *shape,
            device=generator.device,
            dtype=torch.float32,
        )
        bar = tqdm(
            total=generator.diffusion.timesteps,
            desc="Scale up",
            disable=not progress,
        )
        pred_sum = torch.empty(
            1,
            generator.num_phases,
            *shape,
            device=generator.device,
            dtype=torch.float32,
        )
        weighted = torch.empty(
            1,
            generator.num_phases,
            block_size,
            block_size,
            block_size,
            device=generator.device,
            dtype=torch.float32,
        )

        def predict(
            current: torch.Tensor,
            timesteps: torch.Tensor,
            latent: torch.Tensor,
            vf: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return self.predict(
                current,
                timesteps,
                latent,
                vf,
                tiles,
                base,
                pred_sum,
                weighted,
                bar,
            )

        try:
            clean = generator.diffusion.sample(
                predict,
                initial_noise,
                generator.latent_channels,
                conditions=(None if vf is None else {"vf": vf}),
            )
        finally:
            bar.close()

        probs = (clean.float() + 1.0).mul_(0.5).clamp_(0.0, 1.0)
        probs.div_(
            probs.sum(dim=1, keepdim=True).clamp_min_(torch.finfo(probs.dtype).eps)
        )
        self.stats = ScaleStats(
            overlap=overlap,
            block_count=len(tiles),
            seams=tuple(
                tuple(
                    (left + right + block_size) // 2
                    for left, right in pairwise(axis_starts)
                )
                for axis_starts in starts
            ),
        )
        return probs.squeeze(0).cpu()

    def generate(
        self,
        blocks: int | Sequence[int],
        base: torch.Tensor | None = None,
        vf: Sequence[float] | None = None,
        overlap: int | None = None,
        progress: bool = True,
    ) -> torch.Tensor:
        probs = self.generate_probs(
            blocks=blocks,
            base=base,
            vf=vf,
            overlap=overlap,
            progress=progress,
        )
        return probs.argmax(dim=0).to(torch.uint8)

    @staticmethod
    def parse_grid(value: int | Sequence[int]) -> tuple[int, int, int]:
        if isinstance(value, int) and not isinstance(value, bool):
            grid = (value, value, value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            grid = tuple(value)
        else:
            raise TypeError(
                "blocks must be an integer or a sequence of three integers."
            )
        if len(grid) != 3 or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 1
            for count in grid
        ):
            raise ValueError("blocks must contain exactly three positive integers.")
        return grid

    @staticmethod
    def make_axis_weights(
        block_size: int,
        overlap: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pos = torch.arange(overlap, device=device, dtype=torch.float32)
        rise = torch.sin((pos + 0.5) * math.pi / (2.0 * overlap)).square()
        fall = 1.0 - rise
        single = torch.ones(block_size, device=device, dtype=torch.float32)
        start = single.clone()
        start[-overlap:] = fall
        mid = start.clone()
        mid[:overlap] = rise
        end = single.clone()
        end[:overlap] = rise
        return single, start, mid, end

    @classmethod
    def make_tiles(
        cls,
        starts: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        grid: tuple[int, int, int],
        block_size: int,
        axis_weights: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> tuple[Tile, ...]:
        tiles = []
        for idx in product(*(range(count) for count in grid)):
            region = tuple(
                slice(
                    starts[axis][idx[axis]],
                    starts[axis][idx[axis]] + block_size,
                )
                for axis in range(3)
            )
            tiles.append(
                Tile(
                    region=region,
                    weights=tuple(
                        cls.select_weight(axis_weights, idx[axis], grid[axis])
                        for axis in range(3)
                    ),
                )
            )
        return tuple(tiles)

    @staticmethod
    def select_weight(
        weights: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        idx: int,
        count: int,
    ) -> torch.Tensor:
        if count == 1:
            return weights[0]
        if idx == 0:
            return weights[1]
        if idx + 1 == count:
            return weights[3]
        return weights[2]

    def prepare_base(
        self,
        base: torch.Tensor | None,
        shape: tuple[int, int, int],
        grid: tuple[int, int, int],
        overlap: int,
    ) -> Base | None:
        if base is None:
            return None

        generator = self.generator
        base_shape = (generator.patch_size,) * 3
        if not isinstance(base, torch.Tensor):
            raise TypeError("base must be a torch.Tensor.")
        if base.shape != base_shape:
            raise ValueError(f"base must have shape {base_shape}.")
        if base.dtype != torch.uint8:
            raise ValueError("base must use torch.uint8.")
        if int(base.max()) >= generator.num_phases:
            raise ValueError("base contains a phase outside num_phases.")

        start = tuple((size - generator.patch_size) // 2 for size in shape)
        region = tuple(slice(idx, idx + generator.patch_size) for idx in start)
        clean = F.one_hot(
            base.to(device=generator.device, dtype=torch.long),
            num_classes=generator.num_phases,
        )
        clean = clean.movedim(-1, 0).unsqueeze(0).to(torch.float32).mul_(2.0).sub_(1.0)

        axis_weights = self.make_axis_weights(
            generator.patch_size,
            max(1, overlap // 2),
            device=generator.device,
        )
        weights = (
            axis_weights[0 if grid[0] == 1 else 2],
            axis_weights[0 if grid[1] == 1 else 2],
            axis_weights[0 if grid[2] == 1 else 2],
        )
        weight = (
            weights[0].view(1, 1, -1, 1, 1)
            * weights[1].view(1, 1, 1, -1, 1)
            * weights[2].view(1, 1, 1, 1, -1)
        )
        return Base(
            clean=clean,
            noise=torch.randn_like(clean),
            tile=Tile(region=region, weights=weights),
            weight=weight,
        )

    @staticmethod
    def add_prediction(
        pred_sum: torch.Tensor,
        weighted: torch.Tensor,
        pred: torch.Tensor,
        tile: Tile,
    ) -> None:
        key = (slice(None), slice(None), *tile.region)
        weighted.copy_(pred)
        weighted.mul_(tile.weights[0].view(1, 1, -1, 1, 1))
        weighted.mul_(tile.weights[1].view(1, 1, 1, -1, 1))
        weighted.mul_(tile.weights[2].view(1, 1, 1, 1, -1))
        pred_sum[key].add_(weighted)

    def predict(
        self,
        current: torch.Tensor,
        timesteps: torch.Tensor,
        latent: torch.Tensor,
        vf: torch.Tensor | None,
        tiles: tuple[Tile, ...],
        base: Base | None,
        pred_sum: torch.Tensor,
        weighted: torch.Tensor,
        bar: tqdm,
    ) -> torch.Tensor:
        generator = self.generator
        if base is not None:
            key = (slice(None), slice(None), *base.tile.region)
            noisy = generator.diffusion.add_noise(
                base.clean,
                timesteps + 1,
                noise=base.noise,
            )
            current[key].copy_(noisy)

        pred_sum.zero_()
        for tile in tiles:
            key = (slice(None), slice(None), *tile.region)
            with torch.autocast(
                device_type=generator.device.type,
                dtype=torch.float16,
                enabled=generator.use_amp,
            ):
                if vf is None:
                    pred = generator.model(current[key], timesteps, latent)
                else:
                    pred = generator.model(current[key], timesteps, latent, vf=vf)

            self.add_prediction(pred_sum, weighted, pred, tile)
        if base is not None:
            key = (slice(None), slice(None), *base.tile.region)
            pred_sum[key].mul_(1.0 - base.weight)
            self.add_prediction(pred_sum, weighted, base.clean, base.tile)
        bar.update()
        return pred_sum
