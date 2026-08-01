import ctypes
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .anchor import PlaneAnchor, build_anchors
from .diffusion import Diffusion
from .model.denoiser import Denoiser3D


@dataclass(frozen=True)
class ScalePlan:
    shape: tuple[int, int, int]
    tile_size: int
    overlap: int
    core_size: int
    grid: tuple[int, int, int]
    tile_count: int
    states_bytes: int
    tile_bytes: int
    workspace_bytes: int
    cuda_bytes: int
    output_bytes: int
    cpu_bytes: int
    seams: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class Tile:
    source: tuple[slice, slice, slice]
    target: tuple[slice, slice, slice]
    core: tuple[slice, slice, slice]
    padding: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class Base:
    clean: torch.Tensor
    noise: torch.Tensor
    region: tuple[slice, slice, slice]
    mask: torch.Tensor


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
        if enabled:
            try:
                self.upload = torch.empty(
                    num_phases * tile_size**3,
                    dtype=torch.float32,
                    pin_memory=True,
                )
                self.download = torch.empty(
                    num_phases * tile_size**3,
                    dtype=torch.float16,
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
        if source.device == device:
            return source.float()
        if self.upload is None:
            return source.to(device=device, dtype=torch.float32)
        values = self.upload[: source.numel()].view(source.shape)
        values.copy_(source)
        return values.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

    def write(
        self,
        state: VolumeState,
        region: tuple[slice, slice, slice],
        values: torch.Tensor,
    ) -> None:
        if (
            self.download is None
            or values.device == state.values.device
            or values.numel() > self.download.numel()
        ):
            state.write(region, values)
            return
        downloaded = self.download[: values.numel()].view(values.shape)
        downloaded.copy_(values)
        state.write(region, downloaded)


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
        self.stats: ScalePlan | None = None

    def prepare_vf(
        self,
        vf: Sequence[float] | None,
    ) -> torch.Tensor | None:
        return self.generator.prepare_vf(vf)

    def plan(
        self,
        shape: int | Sequence[int],
        overlap: int,
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
        workspace_bytes = max(16 * tile_bytes, 4 * 1024**3)
        states_bytes = 2 * self.generator.num_phases * voxels * 2
        output_bytes = voxels
        seams = tuple(tuple(range(core_size, size, core_size)) for size in shape)
        return ScalePlan(
            shape=shape,
            tile_size=tile_size,
            overlap=overlap,
            core_size=core_size,
            grid=grid,
            tile_count=math.prod(grid),
            states_bytes=states_bytes,
            tile_bytes=tile_bytes,
            workspace_bytes=workspace_bytes,
            cuda_bytes=states_bytes + workspace_bytes,
            output_bytes=output_bytes,
            cpu_bytes=states_bytes + output_bytes + 3 * tile_bytes // 2,
            seams=seams,
        )

    @torch.no_grad()
    def generate_probs(
        self,
        shape: int | Sequence[int],
        overlap: int,
        base: torch.Tensor | None = None,
        vf: Sequence[float] | None = None,
        progress: bool = True,
    ) -> torch.Tensor:
        self.stats = None
        if not isinstance(progress, bool):
            raise TypeError("progress must be a boolean.")
        plan = self.plan(shape, overlap)
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
        shape: int | Sequence[int],
        overlap: int,
        base: torch.Tensor | None = None,
        vf: Sequence[float] | None = None,
        storage: str = "auto",
        progress: bool = True,
    ) -> torch.Tensor:
        self.stats = None
        if not isinstance(progress, bool):
            raise TypeError("progress must be a boolean.")
        plan = self.plan(shape, overlap)
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
        )
        self.stats = plan
        return labels

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
            core = []
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
                core.append(
                    slice(
                        plan.overlap,
                        plan.overlap + target_stop - target_start,
                    )
                )
                pads.append((source_start - input_start, input_stop - source_stop))
            tiles.append(
                Tile(
                    source=tuple(source),
                    target=tuple(target),
                    core=tuple(core),
                    padding=tuple(value for pair in reversed(pads) for value in pair),
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

        shell = min(max(plan.overlap // 2, 1), (generator.patch_size - 1) // 2)
        axes = []
        for axis in range(3):
            active = torch.ones(
                generator.patch_size,
                device=generator.device,
                dtype=torch.bool,
            )
            if plan.shape[axis] > generator.patch_size and shell:
                active[:shell] = False
                active[-shell:] = False
            axes.append(active)
        mask = (
            axes[0].view(1, 1, -1, 1, 1)
            & axes[1].view(1, 1, 1, -1, 1)
            & axes[2].view(1, 1, 1, 1, -1)
        )
        return Base(
            clean=clean,
            noise=torch.randn_like(clean),
            region=region,
            mask=mask,
        )

    def select_storage(
        self,
        plan: ScalePlan,
        storage: str,
        output_bytes: int | None = None,
    ) -> str:
        if storage not in {"auto", "cuda", "cpu"}:
            raise ValueError("storage must be 'auto', 'cuda', or 'cpu'.")

        output_bytes = plan.output_bytes if output_bytes is None else output_bytes
        device = self.generator.device
        free_cuda = 0
        if device.type == "cuda":
            free_cuda, _ = torch.cuda.mem_get_info(device)
            if plan.workspace_bytes > free_cuda:
                raise MemoryError("the planned tile workspace does not fit on CUDA.")
        elif storage == "cuda":
            raise ValueError("storage='cuda' requires a CUDA generator.")

        if storage == "cuda":
            if plan.cuda_bytes > free_cuda:
                raise MemoryError("the planned CUDA state and workspace do not fit.")
            selected = "cuda"
        elif storage == "cpu" or device.type != "cuda":
            selected = "cpu"
        elif plan.cuda_bytes <= int(free_cuda * 0.8):
            selected = "cuda"
        else:
            selected = "cpu"

        cpu_bytes = output_bytes
        if selected == "cpu":
            cpu_bytes += plan.states_bytes + 3 * plan.tile_bytes // 2
        self.check_cpu_memory(cpu_bytes)
        return selected

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
    ) -> VolumeState:
        generator = self.generator
        tile_buffer = TileBuffer(
            generator.num_phases,
            plan.tile_size,
            current.values.device.type == "cpu" and generator.device.type == "cuda",
        )
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
                    plan.tile_size,
                    final_labels,
                    tile_buffer,
                )
                if final_labels is None:
                    current, next_state = next_state, current
                bar.update()
            if base is not None:
                if labels is None:
                    self.condition_base(current, base, base.clean)
                else:
                    self.write_base(labels, base)
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
        tile_size: int,
        labels: torch.Tensor | None,
        tile_buffer: TileBuffer | None = None,
    ) -> None:
        generator = self.generator
        if tile_buffer is None:
            tile_buffer = TileBuffer(
                generator.num_phases,
                tile_size,
                current.values.device.type == "cpu" and generator.device.type == "cuda",
            )
        for tile in tiles:
            values = tile_buffer.read(current, tile.source, generator.device)
            if any(tile.padding):
                values = F.pad(values, tile.padding)
            with torch.autocast(
                device_type=generator.device.type,
                dtype=torch.float16,
                enabled=generator.use_amp,
            ):
                if vf is None:
                    pred = generator.model(values, time, latent)
                else:
                    pred = generator.model(values, time, latent, vf=vf)
            expected = (1, generator.num_phases, tile_size, tile_size, tile_size)
            if pred.shape != expected:
                raise ValueError(f"model prediction must have shape {expected}.")

            key = (slice(None), slice(None), *tile.core)
            clean = pred[key].float()
            if labels is not None:
                self.write_output(labels, tile.target, clean)
                continue
            previous = values[key]
            updated = generator.diffusion.sample_posterior(
                previous,
                clean,
                transition,
            )
            tile_buffer.write(next_state, tile.target, updated)

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
        current.copy_(torch.where(base.mask, values, current))
        state.write(base.region, current)

    @staticmethod
    def write_base(labels: torch.Tensor, base: Base) -> None:
        values = base.clean.argmax(dim=1).squeeze(0).to(device="cpu", dtype=torch.uint8)
        mask = base.mask[0, 0].to(device="cpu")
        region = labels[base.region]
        region[mask] = values[mask]

    @staticmethod
    def write_output(
        labels: torch.Tensor,
        target: tuple[slice, slice, slice],
        clean: torch.Tensor,
    ) -> None:
        values = clean.argmax(dim=1).squeeze(0).to(device="cpu", dtype=torch.uint8)
        labels[target].copy_(values)
