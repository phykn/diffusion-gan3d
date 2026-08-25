import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise, product

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .model.generator import Generator


@dataclass(frozen=True)
class ScalePlan:
    shape: tuple[int, int, int]
    tile_size: int
    overlap: int
    stride: int
    grid: tuple[int, int, int]
    tile_count: int
    seams: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    generation_shape: tuple[int, int, int] | None = None
    margin: int = 0

    @property
    def base_shell(self) -> int:
        return min(self.overlap // 2, (self.stride - 1) // 2)


@dataclass(frozen=True)
class Tile:
    source: tuple[slice, slice, slice]
    target: tuple[slice, slice, slice]
    margins: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class Base:
    clean: torch.Tensor
    noise: torch.Tensor
    region: tuple[slice, slice, slice]
    weight: torch.Tensor


@dataclass(frozen=True)
class Fusion:
    axis_windows: dict[tuple[int, int, int], torch.Tensor]
    weight_sum: torch.Tensor
    pred_sum: torch.Tensor


class VolumeState:
    def __init__(
        self,
        num_phases: int,
        shape: tuple[int, int, int],
        device: torch.device,
    ) -> None:
        self.values = torch.empty(
            (1, num_phases, *shape),
            device=device,
            dtype=torch.float16,
        )

    def read(
        self,
        region: tuple[slice, slice, slice],
    ) -> torch.Tensor:
        return self.values[(slice(None), slice(None), *region)]

    def write(
        self,
        region: tuple[slice, slice, slice],
        values: torch.Tensor,
    ) -> None:
        self.values[(slice(None), slice(None), *region)].copy_(
            values.to(device=self.values.device, dtype=torch.float16)
        )


class TileBuffer:
    def __init__(
        self,
        num_phases: int,
        tile_size: int,
        enabled: bool,
    ) -> None:
        self.upload: torch.Tensor | None = None
        self.download: torch.Tensor | None = None
        self.workspace: torch.Tensor | None = None
        self.capacity = num_phases * tile_size**3
        if enabled:
            try:
                self.upload = torch.empty(
                    self.capacity,
                    dtype=torch.float32,
                    pin_memory=True,
                )
                self.download = torch.empty(
                    self.capacity,
                    dtype=torch.float32,
                    pin_memory=True,
                )
            except RuntimeError:
                self.upload = None
                self.download = None

    def read(
        self,
        state: VolumeState,
        region: tuple[slice, slice, slice],
        device: torch.device,
    ) -> torch.Tensor:
        source = state.read(region)
        shape = source.shape
        numel = math.prod(shape)
        if state.values.device != device and self.upload is not None:
            values = self.upload[:numel].view(shape)
        else:
            if (
                self.workspace is None
                or self.workspace.device != state.values.device
                or self.workspace.numel() < numel
            ):
                self.workspace = torch.empty(
                    max(self.capacity, numel),
                    device=state.values.device,
                    dtype=torch.float32,
                )
            values = self.workspace[:numel].view(shape)

        values.copy_(source)
        if values.device == device:
            return values
        return values.to(
            device=device,
            dtype=torch.float32,
            non_blocking=self.upload is not None,
        )

    def stage(self, values: torch.Tensor, device: torch.device) -> torch.Tensor:
        if values.device == device:
            return values
        if (
            device.type == "cpu"
            and self.download is not None
            and values.numel() <= self.download.numel()
        ):
            downloaded = self.download[: values.numel()].view(values.shape)
            downloaded.copy_(values)
            return downloaded
        return values.to(device)


class ScaledGenerator:
    def __init__(self, generator: Generator) -> None:
        self.generator = generator
        self.stats: ScalePlan | None = None

    def plan(
        self,
        shape: int | Sequence[int],
        overlap: int = 8,
    ) -> ScalePlan:
        shape = self.parse_shape(shape)
        factor = self.generator.default_margin
        if overlap < 0:
            raise ValueError("overlap must be a non-negative integer.")
        tile_size = self.generator.patch_size
        if 2 * overlap >= tile_size:
            raise ValueError("twice overlap must be smaller than patch_size.")
        stride = tile_size - 2 * overlap
        if tile_size % factor:
            raise ValueError(
                "patch_size must be divisible by the denoiser "
                f"downsample factor ({factor})."
            )
        if any(size < tile_size for size in shape):
            raise ValueError("shape must not be smaller than patch_size.")
        starts = tuple(self.axis_starts(size, tile_size, stride) for size in shape)
        grid = tuple(len(axis) for axis in starts)
        seams = tuple(
            tuple((left + tile_size + right) // 2 for left, right in pairwise(axis))
            for axis in starts
        )
        return ScalePlan(
            shape=shape,
            tile_size=tile_size,
            overlap=overlap,
            stride=stride,
            grid=grid,
            tile_count=math.prod(grid),
            seams=seams,
        )

    @torch.no_grad()
    def generate_probs(
        self,
        shape: int | Sequence[int],
        overlap: int = 8,
        base: torch.Tensor | None = None,
        vf: Sequence[float] | None = None,
        progress: bool = True,
        guidance: float = 1.0,
        domain: int | None = None,
        margin: int | None = None,
        base_offset: Sequence[int | None] | None = None,
    ) -> torch.Tensor:
        self.stats = None
        margin = self.generator.default_margin if margin is None else margin
        output_shape = self.parse_shape(shape)
        plan = self._generation_plan(
            output_shape,
            overlap,
            margin,
        )
        storage = self.select_storage("auto")
        tiles = self.make_tiles(plan)
        vf = self.generator.prepare_vf(vf)
        domain = self.generator.prepare_domain(domain)
        base = self.prepare_base(base, plan, offset=base_offset)
        current, next_state = self.make_states(plan, storage)
        self.fill_noise(current, tiles)
        current = self.sample(
            current,
            next_state,
            tiles,
            plan,
            base,
            vf,
            domain,
            labels=None,
            progress=progress,
            guidance=guidance,
        )
        probs = ((current.values.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(probs.dtype).eps,
        )
        probs = self.crop_output(probs.squeeze(0).cpu(), output_shape, margin)
        self.stats = self._output_plan(plan, output_shape)
        return probs

    @torch.no_grad()
    def generate(
        self,
        blocks: int | Sequence[int] | None = None,
        overlap: int = 8,
        base: torch.Tensor | None = None,
        vf: Sequence[float] | None = None,
        storage: str = "auto",
        progress: bool = True,
        shape: int | Sequence[int] | None = None,
        guidance: float = 1.0,
        domain: int | None = None,
        margin: int | None = None,
        base_offset: Sequence[int | None] | None = None,
    ) -> torch.Tensor:
        self.stats = None
        margin = self.generator.default_margin if margin is None else margin
        if blocks is None:
            if shape is None:
                raise TypeError("blocks must be provided.")
            output_shape = self.parse_shape(shape)
        else:
            if shape is not None:
                raise ValueError("blocks and shape cannot be provided together.")
            output_shape = self.shape_from_blocks(blocks, overlap)
        plan = self._generation_plan(output_shape, overlap, margin)
        selected = self.select_storage(storage)
        tiles = self.make_tiles(plan)
        vf = self.generator.prepare_vf(vf)
        domain = self.generator.prepare_domain(domain)
        base = self.prepare_base(base, plan, offset=base_offset)
        current, next_state = self.make_states(plan, selected)
        labels = torch.empty(plan.shape, dtype=torch.uint8)
        self.fill_noise(current, tiles)
        self.sample(
            current,
            next_state,
            tiles,
            plan,
            base,
            vf,
            domain,
            labels=labels,
            progress=progress,
            guidance=guidance,
        )
        labels = self.crop_output(labels, output_shape, margin)
        self.stats = self._output_plan(plan, output_shape)
        return labels

    def shape_from_blocks(
        self,
        blocks: int | Sequence[int],
        overlap: int = 8,
    ) -> tuple[int, int, int]:
        counts = self.parse_shape(blocks)
        patch_size = self.generator.patch_size
        if overlap < 0:
            raise ValueError("overlap must be a non-negative integer.")
        if 2 * overlap >= patch_size:
            raise ValueError("twice overlap must be smaller than patch_size.")
        stride = patch_size - 2 * overlap
        return tuple(patch_size + (count - 1) * stride for count in counts)

    def _generation_plan(
        self,
        output_shape: tuple[int, int, int],
        overlap: int,
        margin: int,
    ) -> ScalePlan:
        if margin < 0:
            raise ValueError("margin must be a non-negative integer.")
        if any(size < self.generator.patch_size for size in output_shape):
            raise ValueError("shape must not be smaller than patch_size.")
        generation_shape = tuple(size + 2 * margin for size in output_shape)
        plan = self.plan(generation_shape, overlap)
        return replace(
            plan,
            generation_shape=generation_shape,
            margin=margin,
        )

    @staticmethod
    def _output_plan(
        plan: ScalePlan,
        output_shape: tuple[int, int, int],
    ) -> ScalePlan:
        margin = plan.margin
        seams = tuple(
            tuple(
                seam - margin
                for seam in axis
                if margin < seam < plan.shape[index] - margin
            )
            for index, axis in enumerate(plan.seams)
        )
        return replace(
            plan,
            shape=output_shape,
            seams=seams,
        )

    @staticmethod
    def crop_output(
        volume: torch.Tensor,
        output_shape: tuple[int, int, int],
        margin: int,
    ) -> torch.Tensor:
        if margin == 0:
            return volume
        region = tuple(slice(margin, margin + size) for size in output_shape)
        leading = (slice(None),) * (volume.ndim - 3)
        return volume[leading + region].clone()

    @staticmethod
    def axis_starts(size: int, tile_size: int, stride: int) -> tuple[int, ...]:
        if size == tile_size:
            return (0,)
        count = math.ceil((size - tile_size) / stride) + 1
        starts = [index * stride for index in range(count - 1)]
        starts.append(size - tile_size)
        return tuple(starts)

    @staticmethod
    def parse_shape(value: int | Sequence[int]) -> tuple[int, int, int]:
        if isinstance(value, int) and not isinstance(value, bool):
            shape = (value, value, value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            shape = tuple(value)
        else:
            raise TypeError("shape must be an integer or a sequence of three integers.")
        if len(shape) != 3 or any(
            not isinstance(size, int) or isinstance(size, bool) or size < 1
            for size in shape
        ):
            raise ValueError("shape must contain exactly three positive integers.")
        return shape

    @staticmethod
    def make_tiles(
        plan: ScalePlan,
    ) -> tuple[Tile, ...]:
        starts = tuple(
            ScaledGenerator.axis_starts(size, plan.tile_size, plan.stride)
            for size in plan.shape
        )
        tiles = []
        for idx in product(
            range(plan.grid[0]),
            range(plan.grid[1]),
            range(plan.grid[2]),
        ):
            source = []
            target = []
            margins = []
            for axis, tile_idx in enumerate(idx):
                source_start = starts[axis][tile_idx]
                source_stop = source_start + plan.tile_size
                target_start = 0 if tile_idx == 0 else plan.seams[axis][tile_idx - 1]
                target_stop = (
                    plan.shape[axis]
                    if tile_idx + 1 == plan.grid[axis]
                    else plan.seams[axis][tile_idx]
                )
                source.append(slice(source_start, source_stop))
                target.append(slice(target_start, target_stop))
                margins.append(
                    (
                        plan.overlap if tile_idx > 0 else 0,
                        plan.overlap if tile_idx + 1 < plan.grid[axis] else 0,
                    )
                )
            tiles.append(
                Tile(
                    source=tuple(source),
                    target=tuple(target),
                    margins=tuple(margins),
                )
            )
        return tuple(tiles)

    def prepare_base(
        self,
        base: torch.Tensor | None,
        plan: ScalePlan,
        offset: Sequence[int | None] | None = None,
    ) -> Base | None:
        if base is None:
            if offset is not None:
                raise ValueError("base_offset requires base.")
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
        output_shape = tuple(size - 2 * plan.margin for size in plan.shape)
        output_offset, explicit = self._resolve_base_offset(offset, output_shape)
        start = tuple(plan.margin + value for value in output_offset)
        region = tuple(slice(idx, idx + generator.patch_size) for idx in start)
        clean = F.one_hot(
            base.to(device=generator.device, dtype=torch.long),
            num_classes=generator.num_phases,
        )
        clean = clean.movedim(-1, 0).unsqueeze(0).to(torch.float32)
        clean = clean * 2.0 - 1.0

        weight_axes = []
        for axis in range(3):
            weight_axis = torch.ones(
                generator.patch_size,
                device=generator.device,
                dtype=torch.float32,
            )
            if plan.shape[axis] > generator.patch_size and plan.base_shell:
                positions = torch.arange(
                    1,
                    plan.base_shell + 1,
                    device=generator.device,
                    dtype=torch.float32,
                )
                ramp = (
                    positions.div(plan.base_shell + 1).mul(math.pi / 2).sin().square()
                )
                if not explicit[axis] or output_offset[axis] > 0:
                    weight_axis[: plan.base_shell] = ramp
                if (
                    not explicit[axis]
                    or output_offset[axis] + generator.patch_size < output_shape[axis]
                ):
                    weight_axis[-plan.base_shell :] = ramp.flip(0)
            weight_axes.append(weight_axis)
        weight = (
            weight_axes[0].view(1, 1, -1, 1, 1)
            * weight_axes[1].view(1, 1, 1, -1, 1)
            * weight_axes[2].view(1, 1, 1, 1, -1)
        )
        return Base(
            clean=clean,
            noise=torch.randn_like(clean),
            region=region,
            weight=weight,
        )

    def _resolve_base_offset(
        self,
        offset: Sequence[int | None] | None,
        output_shape: tuple[int, int, int],
    ) -> tuple[tuple[int, int, int], tuple[bool, bool, bool]]:
        patch_size = self.generator.patch_size
        maximum = tuple(size - patch_size for size in output_shape)
        if offset is None:
            values: tuple[int | None, ...] = (None, None, None)
        elif isinstance(offset, Sequence) and not isinstance(offset, (str, bytes)):
            values = tuple(offset)
        else:
            raise TypeError("base_offset must be a sequence of three values.")
        if len(values) != 3:
            raise ValueError("base_offset must contain exactly three values.")

        resolved = []
        explicit = []
        for axis, (value, limit) in enumerate(zip(values, maximum, strict=True)):
            if value is None:
                resolved.append(limit // 2)
                explicit.append(False)
                continue
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError("base_offset values must be integers or None.")
            if not 0 <= value <= limit:
                raise ValueError(
                    f"base_offset axis {axis} must be between 0 and {limit}."
                )
            resolved.append(value)
            explicit.append(True)
        return tuple(resolved), tuple(explicit)

    def select_storage(
        self,
        storage: str,
    ) -> str:
        if storage not in {"auto", "cuda", "cpu"}:
            raise ValueError("storage must be 'auto', 'cuda', or 'cpu'.")
        device = self.generator.device
        if storage == "cuda" and device.type != "cuda":
            raise ValueError("storage='cuda' requires a CUDA generator.")
        if storage == "auto":
            return "cuda" if device.type == "cuda" else "cpu"
        return storage

    def make_states(
        self,
        plan: ScalePlan,
        storage: str,
    ) -> tuple[VolumeState, VolumeState]:
        device = self.generator.device if storage == "cuda" else torch.device("cpu")
        return (
            VolumeState(self.generator.num_phases, plan.shape, device),
            VolumeState(self.generator.num_phases, plan.shape, device),
        )

    def fill_noise(
        self,
        state: VolumeState,
        tiles: tuple[Tile, ...],
    ) -> None:
        generator = self.generator
        for tile in tiles:
            shape = tuple(region.stop - region.start for region in tile.target)
            noise = torch.randn(
                1,
                generator.num_phases,
                *shape,
                device=state.values.device,
                dtype=torch.float32,
            )
            state.write(tile.target, noise)

    def make_fusion(
        self,
        plan: ScalePlan,
        tiles: tuple[Tile, ...],
        device: torch.device,
    ) -> Fusion:
        axis_windows: dict[tuple[int, int, int], torch.Tensor] = {}
        weight_sum = torch.zeros(
            (1, 1, *plan.shape),
            device=device,
            dtype=torch.float32,
        )
        for tile in tiles:
            global_region = (slice(None), slice(None), *tile.source)
            axes = self.get_axis_windows(
                tile,
                plan.overlap,
                self.generator.device,
                axis_windows,
            )
            window = (
                axes[0].view(1, 1, -1, 1, 1)
                * axes[1].view(1, 1, 1, -1, 1)
                * axes[2].view(1, 1, 1, 1, -1)
            )
            weight_sum[global_region].add_(window.to(device))
        if not bool((weight_sum > 0).all().item()):
            raise RuntimeError("blend weights must cover the complete output volume.")
        pred_sum = torch.zeros(
            (1, self.generator.num_phases, *plan.shape),
            device=device,
            dtype=torch.float32,
        )
        return Fusion(
            axis_windows=axis_windows,
            weight_sum=weight_sum,
            pred_sum=pred_sum,
        )

    @classmethod
    def get_axis_windows(
        cls,
        tile: Tile,
        overlap: int,
        device: torch.device,
        cache: dict[tuple[int, int, int], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        windows = []
        for region, (left_margin, right_margin) in zip(
            tile.source,
            tile.margins,
            strict=True,
        ):
            length = region.stop - region.start
            key = (length, left_margin, right_margin)
            if key not in cache:
                cache[key] = cls.make_axis_window(
                    length,
                    overlap,
                    left_margin,
                    right_margin,
                    device,
                )
            windows.append(cache[key])
        return tuple(windows)

    @staticmethod
    def make_axis_window(
        length: int,
        overlap: int,
        left_margin: int,
        right_margin: int,
        device: torch.device,
    ) -> torch.Tensor:
        axis = torch.ones(length, device=device, dtype=torch.float32)
        if not overlap:
            return axis
        positions = torch.arange(overlap, device=device, dtype=torch.float32)
        ramp = torch.sin(positions * (math.pi / (2 * overlap))).square()
        if left_margin:
            axis[:left_margin] = ramp[-left_margin:]
        if right_margin:
            axis[-right_margin:] = ramp.flip(0)[:right_margin]
        return axis

    def sample(
        self,
        current: VolumeState,
        next_state: VolumeState,
        tiles: tuple[Tile, ...],
        plan: ScalePlan,
        base: Base | None,
        vf: torch.Tensor | None,
        domain: torch.Tensor,
        labels: torch.Tensor | None,
        progress: bool,
        guidance: float = 1.0,
    ) -> VolumeState:
        generator = self.generator
        tile_buffer = TileBuffer(
            generator.num_phases,
            plan.tile_size,
            current.values.device.type == "cpu" and generator.device.type == "cuda",
        )
        fusion = self.make_fusion(plan, tiles, current.values.device)
        with tqdm(
            total=generator.diffusion.timesteps,
            desc="Scale up",
            disable=not progress,
        ) as bar:
            for transition in reversed(range(generator.diffusion.timesteps)):
                time = torch.full(
                    (1,),
                    transition,
                    device=generator.device,
                    dtype=torch.long,
                )
                latent = torch.randn(
                    1,
                    generator.latent_channels,
                    device=generator.device,
                    dtype=torch.float32,
                )
                if base is not None:
                    noisy = generator.diffusion.add_noise(
                        base.clean,
                        time + 1,
                        noise=base.noise,
                    )
                    self.condition_base(current, base, noisy)
                final_labels = labels if transition == 0 else None
                self.step(
                    current,
                    next_state,
                    tiles,
                    time,
                    latent,
                    vf,
                    domain,
                    transition,
                    plan,
                    final_labels,
                    fusion,
                    tile_buffer,
                    guidance=guidance,
                )
                if final_labels is None:
                    current, next_state = next_state, current
                bar.update()
        return current

    def step(
        self,
        current: VolumeState,
        next_state: VolumeState,
        tiles: tuple[Tile, ...],
        time: torch.Tensor,
        latent: torch.Tensor,
        vf: torch.Tensor | None,
        domain: torch.Tensor,
        transition: int,
        plan: ScalePlan,
        labels: torch.Tensor | None,
        fusion: Fusion,
        tile_buffer: TileBuffer | None = None,
        guidance: float = 1.0,
    ) -> None:
        generator = self.generator
        if tile_buffer is None:
            tile_buffer = TileBuffer(
                generator.num_phases,
                plan.tile_size,
                current.values.device.type == "cpu" and generator.device.type == "cuda",
            )
        fusion.pred_sum.zero_()
        for tile in tiles:
            values = tile_buffer.read(
                current,
                tile.source,
                generator.device,
            )
            with torch.autocast(
                device_type=generator.device.type,
                dtype=torch.float16,
                enabled=generator.use_amp,
            ):
                pred = generator.predict(
                    values,
                    time,
                    latent,
                    guidance=guidance,
                    domain=domain,
                    vf=vf,
                )
            expected = (1, generator.num_phases, *values.shape[-3:])
            if pred.shape != expected:
                raise ValueError(f"model prediction must have shape {expected}.")
            self.add_prediction(fusion, tile, pred, tile_buffer, plan.overlap)

        if labels is None:
            self.update_state(current, next_state, tiles, fusion, transition)
        else:
            self.write_labels(labels, tiles, fusion)

    @staticmethod
    def add_prediction(
        fusion: Fusion,
        tile: Tile,
        pred: torch.Tensor,
        tile_buffer: TileBuffer,
        overlap: int,
    ) -> None:
        global_region = (slice(None), slice(None), *tile.source)
        weighted = pred.float().clone()
        axes = ScaledGenerator.get_axis_windows(
            tile,
            overlap,
            pred.device,
            fusion.axis_windows,
        )
        for spatial_axis, axis in enumerate(axes, 2):
            shape = [1, 1, 1, 1, 1]
            shape[spatial_axis] = axis.numel()
            weighted.mul_(axis.view(shape))
        staged = tile_buffer.stage(weighted, fusion.pred_sum.device)
        fusion.pred_sum[global_region].add_(staged)

    def update_state(
        self,
        current: VolumeState,
        next_state: VolumeState,
        tiles: tuple[Tile, ...],
        fusion: Fusion,
        transition: int,
    ) -> None:
        generator = self.generator
        for tile in tiles:
            region = (slice(None), slice(None), *tile.target)
            clean = fusion.pred_sum[region] / fusion.weight_sum[region]
            previous = current.read(tile.target).float()
            updated = generator.diffusion.sample_posterior(
                previous,
                clean,
                transition,
            )
            next_state.write(tile.target, updated)

    def write_labels(
        self,
        labels: torch.Tensor,
        tiles: tuple[Tile, ...],
        fusion: Fusion,
    ) -> None:
        for tile in tiles:
            region = (slice(None), slice(None), *tile.target)
            clean = fusion.pred_sum[region] / fusion.weight_sum[region]
            self.write_output(labels, tile.target, clean)

    @staticmethod
    def condition_base(
        state: VolumeState,
        base: Base,
        values: torch.Tensor,
    ) -> None:
        current = state.read(base.region).to(
            device=base.clean.device,
            dtype=torch.float32,
        )
        current.lerp_(values, base.weight)
        state.write(base.region, current)

    @staticmethod
    def write_output(
        labels: torch.Tensor,
        target: tuple[slice, slice, slice],
        clean: torch.Tensor,
    ) -> None:
        values = clean.argmax(dim=1).squeeze(0).to(device="cpu", dtype=torch.uint8)
        labels[target].copy_(values)
