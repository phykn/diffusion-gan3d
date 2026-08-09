import ctypes
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import product

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .generate import Generator
from .model.denoiser import Denoiser3D, validate_guidance_scale

DEFAULT_SCALE_OVERLAP = 8


@dataclass(frozen=True)
class ScalePlan:
    shape: tuple[int, int, int]
    tile_size: int
    overlap: int
    core_size: int
    grid: tuple[int, int, int]
    tile_count: int
    states_bytes: int
    fusion_bytes: int
    tile_bytes: int
    workspace_bytes: int
    cuda_bytes: int
    output_bytes: int
    cpu_bytes: int
    seams: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    guidance_bytes: int = 0

    @property
    def base_shell(self) -> int:
        return min(self.overlap // 2, (self.core_size - 1) // 2)


@dataclass(frozen=True)
class Tile:
    source: tuple[slice, slice, slice]
    target: tuple[slice, slice, slice]
    valid: tuple[slice, slice, slice]


@dataclass(frozen=True)
class Base:
    clean: torch.Tensor
    noise: torch.Tensor
    region: tuple[slice, slice, slice]
    weight: torch.Tensor


@dataclass(frozen=True)
class Fusion:
    window: torch.Tensor
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
        key = (slice(None), slice(None), *region)
        return self.values[key]

    def write(
        self,
        region: tuple[slice, slice, slice],
        values: torch.Tensor,
    ) -> None:
        key = (slice(None), slice(None), *region)
        self.values[key].copy_(
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
        if enabled:
            try:
                self.upload = torch.empty(
                    num_phases * tile_size**3,
                    dtype=torch.float32,
                    pin_memory=True,
                )
                self.download = torch.empty(
                    num_phases * tile_size**3,
                    dtype=torch.float32,
                    pin_memory=True,
                )
            except RuntimeError:
                self.upload = None
                self.download = None

    def read_periodic(
        self,
        state: VolumeState,
        start: tuple[int, int, int],
        tile_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        shape = (1, state.values.shape[1], tile_size, tile_size, tile_size)
        numel = math.prod(shape)
        if state.values.device != device and self.upload is not None:
            values = self.upload[:numel].view(shape)
        else:
            if (
                self.workspace is None
                or self.workspace.shape != shape
                or self.workspace.device != state.values.device
            ):
                self.workspace = torch.empty(
                    shape,
                    device=state.values.device,
                    dtype=torch.float32,
                )
            values = self.workspace

        segments = tuple(
            self.periodic_segments(offset, tile_size, size)
            for offset, size in zip(start, state.values.shape[-3:], strict=True)
        )
        for blocks in product(*segments):
            target = tuple(block[0] for block in blocks)
            source = tuple(block[1] for block in blocks)
            values[(slice(None), slice(None), *target)].copy_(state.read(source))

        if values.device == device:
            return values
        return values.to(
            device=device,
            dtype=torch.float32,
            non_blocking=self.upload is not None,
        )

    @staticmethod
    def periodic_segments(
        start: int,
        length: int,
        size: int,
    ) -> tuple[tuple[slice, slice], ...]:
        segments = []
        target_start = 0
        source_start = start % size
        while target_start < length:
            count = min(size - source_start, length - target_start)
            segments.append(
                (
                    slice(target_start, target_start + count),
                    slice(source_start, source_start + count),
                )
            )
            target_start += count
            source_start = 0
        return tuple(segments)

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

    def prepare_vf(
        self,
        vf: Sequence[float] | None,
    ) -> torch.Tensor | None:
        return self.generator.prepare_vf(vf)

    def plan(
        self,
        shape: int | Sequence[int],
        overlap: int = DEFAULT_SCALE_OVERLAP,
    ) -> ScalePlan:
        shape = self.parse_shape(shape)
        factor = self.get_downsample_factor()
        if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
            raise ValueError("overlap must be a non-negative integer.")
        core_size = self.generator.patch_size
        tile_size = core_size + 2 * overlap
        if core_size % factor or tile_size % factor:
            raise ValueError(
                "patch_size and the resulting tile size must be divisible by the denoiser "
                f"downsample factor ({factor})."
            )
        grid = tuple(math.ceil(size / core_size) for size in shape)
        voxels = math.prod(shape)
        tile_voxels = tile_size**3
        tile_bytes = self.generator.num_phases * tile_voxels * 4
        input_layer = getattr(self.generator.model, "input", None)
        width = getattr(input_layer, "out_channels", self.generator.num_phases)
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            width = self.generator.num_phases
        workspace_bytes = 12 * width * tile_voxels * 4
        states_bytes = 2 * self.generator.num_phases * voxels * 2
        fusion_bytes = (self.generator.num_phases + 1) * voxels * 4
        output_bytes = voxels
        cpu_workspace = workspace_bytes if self.generator.device.type == "cpu" else 0
        seams = tuple(tuple(range(core_size, size, core_size)) for size in shape)
        return ScalePlan(
            shape=shape,
            tile_size=tile_size,
            overlap=overlap,
            core_size=core_size,
            grid=grid,
            tile_count=math.prod(grid),
            states_bytes=states_bytes,
            fusion_bytes=fusion_bytes,
            tile_bytes=tile_bytes,
            workspace_bytes=workspace_bytes,
            cuda_bytes=states_bytes + fusion_bytes + workspace_bytes,
            output_bytes=output_bytes,
            cpu_bytes=(
                states_bytes
                + fusion_bytes
                + output_bytes
                + 2 * tile_bytes
                + cpu_workspace
            ),
            seams=seams,
        )

    def _account_guidance_memory(
        self,
        plan: ScalePlan,
        guidance_scale: float,
        conditioned: bool,
    ) -> ScalePlan:
        if not conditioned or guidance_scale in {0.0, 1.0}:
            return plan
        bytes_per_logit = 2 if self.generator.use_amp else 4
        # Keep both logits and their two float32 guidance work buffers.
        guidance_bytes = (
            self.generator.num_phases
            * plan.tile_size**3
            * (2 * bytes_per_logit + 2 * 4)
        )
        cpu_bytes = plan.cpu_bytes
        if self.generator.device.type == "cpu":
            cpu_bytes += guidance_bytes
        return replace(
            plan,
            guidance_bytes=guidance_bytes,
            workspace_bytes=plan.workspace_bytes + guidance_bytes,
            cuda_bytes=plan.cuda_bytes + guidance_bytes,
            cpu_bytes=cpu_bytes,
        )

    @torch.no_grad()
    def generate_probs(
        self,
        shape: int | Sequence[int],
        overlap: int = DEFAULT_SCALE_OVERLAP,
        base: torch.Tensor | None = None,
        vf: Sequence[float] | None = None,
        progress: bool = True,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        self.stats = None
        if not isinstance(progress, bool):
            raise TypeError("progress must be a boolean.")
        guidance_scale = validate_guidance_scale(guidance_scale)
        plan = self._account_guidance_memory(
            self.plan(shape, overlap),
            guidance_scale,
            vf is not None,
        )
        if plan.states_bytes > 1024**3:
            raise ValueError(
                "generate_probs only supports small in-memory volumes; "
                "use generate for large output."
            )
        output_bytes = 4 * self.generator.num_phases * math.prod(plan.shape)
        storage = self.select_storage(plan, "auto", output_bytes)
        tiles = self.make_tiles(plan)
        vf = self.prepare_vf(vf)
        base = self.prepare_base(base, plan)
        current, next_state = self.make_states(plan, storage)
        self.fill_noise(current, tiles)
        current = self.run(
            current,
            next_state,
            tiles,
            plan,
            base,
            vf,
            labels=None,
            progress=progress,
            guidance_scale=guidance_scale,
        )
        probs = current.values.float()
        probs.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
        probs.div_(
            probs.sum(dim=1, keepdim=True).clamp_min_(torch.finfo(probs.dtype).eps)
        )
        probs = probs.squeeze(0).cpu()
        self.stats = plan
        return probs

    @torch.no_grad()
    def generate(
        self,
        blocks: int | Sequence[int] | None = None,
        overlap: int = DEFAULT_SCALE_OVERLAP,
        base: torch.Tensor | None = None,
        vf: Sequence[float] | None = None,
        storage: str = "auto",
        progress: bool = True,
        *,
        shape: int | Sequence[int] | None = None,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        self.stats = None
        if not isinstance(progress, bool):
            raise TypeError("progress must be a boolean.")
        guidance_scale = validate_guidance_scale(guidance_scale)
        if blocks is None:
            if shape is None:
                raise TypeError("blocks must be provided.")
            output_shape = self.parse_shape(shape)
        else:
            if shape is not None:
                raise ValueError("blocks and shape cannot be provided together.")
            output_shape = self.shape_from_blocks(blocks)
        plan = self._account_guidance_memory(
            self.plan(output_shape, overlap),
            guidance_scale,
            vf is not None,
        )
        selected = self.select_storage(plan, storage)
        tiles = self.make_tiles(plan)
        vf = self.prepare_vf(vf)
        base = self.prepare_base(base, plan)
        current, next_state = self.make_states(plan, selected)
        labels = torch.empty(plan.shape, dtype=torch.uint8)
        self.fill_noise(current, tiles)
        self.run(
            current,
            next_state,
            tiles,
            plan,
            base,
            vf,
            labels=labels,
            progress=progress,
            guidance_scale=guidance_scale,
        )
        self.stats = plan
        return labels

    def shape_from_blocks(
        self,
        blocks: int | Sequence[int],
    ) -> tuple[int, int, int]:
        counts = self.parse_shape(blocks)
        return tuple(self.generator.patch_size * count for count in counts)

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

    def get_downsample_factor(self) -> int:
        model = self.generator.model
        factor = getattr(model, "downsample_factor", None)
        if factor is None:
            if isinstance(model, Denoiser3D):
                raise AttributeError("Denoiser3D must expose downsample_factor.")
            factor = 1
        if not isinstance(factor, int) or isinstance(factor, bool) or factor < 1:
            raise ValueError("model.downsample_factor must be a positive integer.")
        return factor

    @staticmethod
    def make_tiles(
        plan: ScalePlan,
    ) -> tuple[Tile, ...]:
        tiles = []
        for idx in product(*(range(count) for count in plan.grid)):
            source = []
            target = []
            pads = []
            for axis, tile_idx in enumerate(idx):
                target_start = tile_idx * plan.core_size
                target_stop = min(target_start + plan.core_size, plan.shape[axis])
                input_start = target_start - plan.overlap
                input_stop = target_start + plan.core_size + plan.overlap
                source_start = max(input_start, 0)
                source_stop = min(input_stop, plan.shape[axis])
                source.append(slice(source_start, source_stop))
                target.append(slice(target_start, target_stop))
                pads.append((source_start - input_start, input_stop - source_stop))
            tiles.append(
                Tile(
                    source=tuple(source),
                    target=tuple(target),
                    valid=tuple(
                        slice(left, plan.tile_size - right) for left, right in pads
                    ),
                )
            )
        return tuple(tiles)

    def prepare_base(
        self,
        base: torch.Tensor | None,
        plan: ScalePlan,
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
        if any(size < generator.patch_size for size in plan.shape):
            raise ValueError("shape must not be smaller than the base.")

        start = tuple((size - generator.patch_size) // 2 for size in plan.shape)
        region = tuple(slice(idx, idx + generator.patch_size) for idx in start)
        clean = F.one_hot(
            base.to(device=generator.device, dtype=torch.long),
            num_classes=generator.num_phases,
        )
        clean = clean.movedim(-1, 0).unsqueeze(0).to(torch.float32).mul_(2.0).sub_(1.0)

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
                weight_axis[: plan.base_shell] = ramp
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

    def select_storage(
        self,
        plan: ScalePlan,
        storage: str,
        output_bytes: int | None = None,
    ) -> str:
        if storage not in {"auto", "cuda", "cpu"}:
            raise ValueError("storage must be 'auto', 'cuda', or 'cpu'.")

        cuda_output_bytes = 0 if output_bytes is None else output_bytes
        output_bytes = plan.output_bytes if output_bytes is None else output_bytes
        cuda_required = plan.cuda_bytes + cuda_output_bytes
        device = self.generator.device
        free_cuda = 0
        if device.type == "cuda":
            free_cuda = self.get_cuda_available_memory(device)
            if plan.workspace_bytes > free_cuda:
                raise MemoryError("the planned tile workspace does not fit on CUDA.")
        elif storage == "cuda":
            raise ValueError("storage='cuda' requires a CUDA generator.")

        if storage == "cuda":
            if cuda_required > free_cuda:
                raise MemoryError(
                    "the planned CUDA state, workspace, and output do not fit."
                )
            selected = "cuda"
        elif storage == "cpu" or device.type != "cuda":
            selected = "cpu"
        elif cuda_required <= int(free_cuda * 0.8):
            selected = "cuda"
        else:
            selected = "cpu"

        cpu_bytes = output_bytes
        if selected == "cpu":
            cpu_bytes += plan.states_bytes + plan.fusion_bytes + 2 * plan.tile_bytes
            if device.type == "cpu":
                cpu_bytes += plan.workspace_bytes
        self.check_cpu_memory(cpu_bytes)
        return selected

    @staticmethod
    def get_cuda_available_memory(device: torch.device) -> int:
        free, total = torch.cuda.mem_get_info(device)
        reclaimable = max(
            torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device),
            0,
        )
        return min(total, free + reclaimable)

    @staticmethod
    def check_cpu_memory(required: int) -> None:
        available = ScaledGenerator.get_available_memory()
        if available is None:
            raise RuntimeError("available CPU memory could not be determined.")
        safe = int(available * 0.8)
        if required > safe:
            raise MemoryError(
                f"planned CPU allocation requires {required} bytes, "
                f"which exceeds 80% of the {available} available bytes."
            )

    @staticmethod
    def get_available_memory() -> int | None:
        if os.name == "nt":

            class MemoryStatus(ctypes.Structure):
                _fields_ = (
                    ("length", ctypes.c_uint32),
                    ("load", ctypes.c_uint32),
                    ("total_physical", ctypes.c_uint64),
                    ("available_physical", ctypes.c_uint64),
                    ("total_page_file", ctypes.c_uint64),
                    ("available_page_file", ctypes.c_uint64),
                    ("total_virtual", ctypes.c_uint64),
                    ("available_virtual", ctypes.c_uint64),
                    ("available_extended_virtual", ctypes.c_uint64),
                )

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            try:
                success = ctypes.windll.kernel32.GlobalMemoryStatusEx(  # type: ignore[attr-defined]
                    ctypes.byref(status)
                )
            except (AttributeError, OSError):
                return None
            return int(status.available_physical) if success else None

        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
        except (AttributeError, OSError, ValueError):
            return None
        if not isinstance(pages, int) or not isinstance(page_size, int):
            return None
        return pages * page_size

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
        window = self.make_window(
            plan.tile_size,
            plan.overlap,
            self.generator.device,
        )
        weight_sum = torch.zeros(
            (1, 1, *plan.shape),
            device=device,
            dtype=torch.float32,
        )
        for tile in tiles:
            global_region = (slice(None), slice(None), *tile.source)
            tile_region = (slice(None), slice(None), *tile.valid)
            weight_sum[global_region].add_(window[tile_region].to(device))
        if not bool((weight_sum > 0).all().item()):
            raise RuntimeError("blend weights must cover the complete output volume.")
        pred_sum = torch.zeros(
            (1, self.generator.num_phases, *plan.shape),
            device=device,
            dtype=torch.float32,
        )
        return Fusion(
            window=window,
            weight_sum=weight_sum,
            pred_sum=pred_sum,
        )

    @staticmethod
    def make_window(
        tile_size: int,
        overlap: int,
        device: torch.device,
    ) -> torch.Tensor:
        axis = torch.ones(tile_size, device=device, dtype=torch.float32)
        if overlap:
            positions = torch.arange(
                overlap,
                device=device,
                dtype=torch.float32,
            )
            ramp = torch.sin(positions.mul(math.pi / (2 * overlap))).square()
            axis[:overlap] = ramp
            axis[-overlap:] = ramp.flip(0)
        return (
            axis.view(1, 1, -1, 1, 1)
            * axis.view(1, 1, 1, -1, 1)
            * axis.view(1, 1, 1, 1, -1)
        )

    def run(
        self,
        current: VolumeState,
        next_state: VolumeState,
        tiles: tuple[Tile, ...],
        plan: ScalePlan,
        base: Base | None,
        vf: torch.Tensor | None,
        labels: torch.Tensor | None,
        progress: bool,
        guidance_scale: float = 1.0,
    ) -> VolumeState:
        generator = self.generator
        tile_buffer = TileBuffer(
            generator.num_phases,
            plan.tile_size,
            current.values.device.type == "cpu" and generator.device.type == "cuda",
        )
        fusion = self.make_fusion(plan, tiles, current.values.device)
        bar = tqdm(
            total=generator.diffusion.timesteps,
            desc="Scale up",
            disable=not progress,
        )
        try:
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
                    transition,
                    plan,
                    final_labels,
                    fusion,
                    tile_buffer,
                    guidance_scale=guidance_scale,
                )
                if final_labels is None:
                    current, next_state = next_state, current
                bar.update()
        finally:
            bar.close()
        return current

    def step(
        self,
        current: VolumeState,
        next_state: VolumeState,
        tiles: tuple[Tile, ...],
        time: torch.Tensor,
        latent: torch.Tensor,
        vf: torch.Tensor | None,
        transition: int,
        plan: ScalePlan,
        labels: torch.Tensor | None,
        fusion: Fusion,
        tile_buffer: TileBuffer | None = None,
        guidance_scale: float = 1.0,
    ) -> None:
        generator = self.generator
        if tile_buffer is None:
            tile_buffer = TileBuffer(
                generator.num_phases,
                plan.tile_size,
                current.values.device.type == "cpu" and generator.device.type == "cuda",
            )
        fusion.pred_sum.zero_()
        expected = (
            1,
            generator.num_phases,
            plan.tile_size,
            plan.tile_size,
            plan.tile_size,
        )
        for tile in tiles:
            start = tuple(region.start - plan.overlap for region in tile.target)
            values = tile_buffer.read_periodic(
                current,
                start,
                plan.tile_size,
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
                    guidance_scale=guidance_scale,
                    vf=vf,
                )
            if pred.shape != expected:
                raise ValueError(f"model prediction must have shape {expected}.")
            self.add_prediction(fusion, tile, pred, tile_buffer)

        self.update_state(
            current,
            next_state,
            tiles,
            fusion,
            transition,
            labels,
        )

    @staticmethod
    def add_prediction(
        fusion: Fusion,
        tile: Tile,
        pred: torch.Tensor,
        tile_buffer: TileBuffer,
    ) -> None:
        global_region = (slice(None), slice(None), *tile.source)
        tile_region = (slice(None), slice(None), *tile.valid)
        weighted = pred[tile_region].float() * fusion.window[tile_region]
        staged = tile_buffer.stage(weighted, fusion.pred_sum.device)
        fusion.pred_sum[global_region].add_(staged)

    def update_state(
        self,
        current: VolumeState,
        next_state: VolumeState,
        tiles: tuple[Tile, ...],
        fusion: Fusion,
        transition: int,
        labels: torch.Tensor | None,
    ) -> None:
        generator = self.generator
        for tile in tiles:
            region = (slice(None), slice(None), *tile.target)
            clean = fusion.pred_sum[region] / fusion.weight_sum[region]
            if labels is not None:
                self.write_output(labels, tile.target, clean)
                continue
            previous = current.read(tile.target).float()
            updated = generator.diffusion.sample_posterior(
                previous,
                clean,
                transition,
            )
            next_state.write(tile.target, updated)

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
