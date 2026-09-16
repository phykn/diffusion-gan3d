import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.anchor import PlaneAnchor, encode_anchors
from src.predict.generator import Generator
from src.predict.tile import (
    Fusion,
    Tile,
    TileBuffer,
    TilePlan,
    VolumeState,
    add_prediction,
    axis_starts,
    crop_output,
    make_fusion,
    make_tiles,
    output_plan,
    parse_shape,
    write_labels,
)


@dataclass(frozen=True)
class Base:
    clean: torch.Tensor
    noise: torch.Tensor
    region: tuple[slice, slice, slice]
    weight: torch.Tensor


class TiledGenerator:
    def __init__(self, generator: Generator) -> None:
        self.generator = generator
        self.stats: TilePlan | None = None

    def plan(
        self,
        shape: int | Sequence[int],
        overlap: int = 8,
    ) -> TilePlan:
        shape = parse_shape(shape)
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
        starts = tuple(axis_starts(size, tile_size, stride) for size in shape)
        grid = tuple(len(axis) for axis in starts)
        seams = tuple(
            tuple((left + tile_size + right) // 2 for left, right in pairwise(axis))
            for axis in starts
        )
        return TilePlan(
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
        anchors: Sequence[PlaneAnchor] = (),
        anchor_strength: float = 1.0,
        height_origin: float = 0.0,
    ) -> torch.Tensor:
        self.stats = None
        margin = self.generator.default_margin if margin is None else margin
        output_shape = parse_shape(shape)
        plan = self._generation_plan(
            output_shape,
            overlap,
            margin,
        )
        self.generator.validate_height(output_shape, domain, height_origin)
        storage = self.select_storage("auto")
        tiles = make_tiles(plan)
        tile_anchors = self.prepare_anchors(anchors, tiles, plan, anchor_strength)
        vf = self.generator.prepare_vf(vf)
        height_domain = domain
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
            tile_anchors=tile_anchors,
            anchor_strength=anchor_strength,
            height_origin=height_origin,
            height_domain=height_domain,
        )
        probs = ((current.values.float() + 1.0) * 0.5).clamp(0.0, 1.0)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(probs.dtype).eps,
        )
        probs = crop_output(probs.squeeze(0).cpu(), output_shape, margin)
        self.stats = output_plan(plan, output_shape)
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
        anchors: Sequence[PlaneAnchor] = (),
        anchor_strength: float = 1.0,
        height_origin: float = 0.0,
    ) -> torch.Tensor:
        self.stats = None
        margin = self.generator.default_margin if margin is None else margin
        if blocks is None:
            if shape is None:
                raise TypeError("blocks must be provided.")
            output_shape = parse_shape(shape)
        else:
            if shape is not None:
                raise ValueError("blocks and shape cannot be provided together.")
            output_shape = self.shape_from_blocks(blocks, overlap)
        plan = self._generation_plan(output_shape, overlap, margin)
        self.generator.validate_height(output_shape, domain, height_origin)
        selected = self.select_storage(storage)
        tiles = make_tiles(plan)
        tile_anchors = self.prepare_anchors(anchors, tiles, plan, anchor_strength)
        vf = self.generator.prepare_vf(vf)
        height_domain = domain
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
            tile_anchors=tile_anchors,
            anchor_strength=anchor_strength,
            height_origin=height_origin,
            height_domain=height_domain,
        )
        labels = crop_output(labels, output_shape, margin)
        self.stats = output_plan(plan, output_shape)
        return labels

    def prepare_anchors(self, anchors, tiles, plan, strength):
        self.generator.validate_anchor_strength(strength)
        shape = tuple(size - 2 * plan.margin for size in plan.shape)
        placed = []
        for anchor in anchors:
            if (
                anchor.axis not in (0, 1, 2)
                or not 0 <= anchor.index < shape[anchor.axis]
            ):
                raise ValueError("anchor.index is outside the generated volume.")
            height, width = anchor.image.shape[-2:]
            axes = tuple(axis for axis in range(3) if axis != anchor.axis)
            row, col = anchor.position or tuple(
                (shape[axis] - length) // 2
                for axis, length in zip(axes, (height, width))
            )
            if (
                row < 0
                or col < 0
                or row + height > shape[axes[0]]
                or col + width > shape[axes[1]]
            ):
                raise ValueError("anchor.position places the image outside the plane.")
            placed.append((anchor, axes, row + plan.margin, col + plan.margin))
        result = []
        for tile in tiles:
            local = []
            for anchor, axes, row, col in placed:
                index = anchor.index + plan.margin
                normal = tile.source[anchor.axis]
                if not normal.start <= index < normal.stop:
                    continue
                rows, cols = (tile.source[axis] for axis in axes)
                top, left = max(row, rows.start), max(col, cols.start)
                bottom = min(row + anchor.image.shape[-2], rows.stop)
                right = min(col + anchor.image.shape[-1], cols.stop)
                if top >= bottom or left >= right:
                    continue
                local.append(
                    PlaneAnchor(
                        anchor.image[
                            ..., top - row : bottom - row, left - col : right - col
                        ],
                        anchor.axis,
                        index - normal.start,
                        (top - rows.start, left - cols.start),
                    )
                )
            result.append(tuple(local))
            if local:
                encode_anchors(
                    tuple(
                        replace(anchor, image=anchor.image.cpu()) for anchor in local
                    ),
                    1,
                    self.generator.num_phases,
                    plan.tile_size,
                    torch.device("cpu"),
                    torch.float32,
                )
        return tuple(result)

    def shape_from_blocks(
        self,
        blocks: int | Sequence[int],
        overlap: int = 8,
    ) -> tuple[int, int, int]:
        counts = parse_shape(blocks)
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
    ) -> TilePlan:
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

    def prepare_base(
        self,
        base: torch.Tensor | None,
        plan: TilePlan,
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
        plan: TilePlan,
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

    def sample(
        self,
        current: VolumeState,
        next_state: VolumeState,
        tiles: tuple[Tile, ...],
        plan: TilePlan,
        base: Base | None,
        vf: torch.Tensor | None,
        domain: torch.Tensor,
        labels: torch.Tensor | None,
        progress: bool,
        guidance: float = 1.0,
        tile_anchors: tuple = (),
        anchor_strength: float = 1.0,
        height_origin: float = 0.0,
        height_domain: int | None = None,
    ) -> VolumeState:
        generator = self.generator
        tile_buffer = TileBuffer(
            generator.num_phases,
            plan.tile_size,
            current.values.device.type == "cpu" and generator.device.type == "cuda",
        )
        fusion = make_fusion(
            plan, tiles, generator.num_phases, current.values.device, generator.device
        )
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
                    tile_anchors=tile_anchors,
                    anchor_strength=anchor_strength,
                    height_origin=height_origin,
                    height_domain=height_domain,
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
        plan: TilePlan,
        labels: torch.Tensor | None,
        fusion: Fusion,
        tile_buffer: TileBuffer | None = None,
        guidance: float = 1.0,
        tile_anchors: tuple = (),
        anchor_strength: float = 1.0,
        height_origin: float = 0.0,
        height_domain: int | None = None,
    ) -> None:
        generator = self.generator
        if tile_buffer is None:
            tile_buffer = TileBuffer(
                generator.num_phases,
                plan.tile_size,
                current.values.device.type == "cpu" and generator.device.type == "cuda",
            )
        fusion.pred_sum.zero_()
        for index, tile in enumerate(tiles):
            values = tile_buffer.read(
                current,
                tile.source,
                generator.device,
            )
            conditions = {}
            if tile_anchors and tile_anchors[index] and anchor_strength > 0:
                anchor = encode_anchors(
                    tile_anchors[index],
                    1,
                    generator.num_phases,
                    tuple(values.shape[-3:]),
                    generator.device,
                    values.dtype,
                    validate=False,
                )
                conditions = {
                    "anchor_image": anchor.image,
                    "anchor_mask": anchor.mask,
                    "anchor_strength": anchor_strength,
                }
            if generator.height_data is not None:
                axis = {"z": 0, "y": 1, "x": 2}[generator.height_data["thickness_axis"]]
                conditions["height"] = generator.height_condition(
                    values.shape[-3:],
                    height_domain,
                    height_origin,
                    tile.source[axis].start - plan.margin,
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
                    **conditions,
                )
            expected = (1, generator.num_phases, *values.shape[-3:])
            if pred.shape != expected:
                raise ValueError(f"model prediction must have shape {expected}.")
            add_prediction(fusion, tile, pred, tile_buffer, plan.overlap)

        if labels is None:
            self.update_state(current, next_state, tiles, fusion, transition)
        else:
            write_labels(labels, tiles, fusion)

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
