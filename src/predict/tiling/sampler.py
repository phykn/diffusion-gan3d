import math
from collections.abc import Sequence
from dataclasses import replace
from itertools import pairwise

import torch
from tqdm import tqdm

from src.anchor import PlaneAnchor, encode_anchors
from src.predict.base import Base, prepare_base, validate_base_anchors, volume_shape
from src.predict.generator import Generator
from src.predict.memory import MemoryEstimate, estimate_memory, select_storage
from src.predict.tiling.fusion import Fusion, add_prediction, make_fusion
from src.predict.tiling.layout import (
    Tile,
    TilePlan,
    axis_starts,
    crop_output,
    make_tiles,
    output_plan,
    parse_shape,
)
from src.predict.tiling.state import (
    TileBuffer,
    VolumeState,
    collect_probabilities,
    write_output,
)


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
        if type(overlap) is not int or overlap < 0:
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
        height_extent: float | None = None,
        vf_profile: dict | None = None,
        storage: str = "auto",
        preserve_base: bool = False,
    ) -> torch.Tensor:
        return self._generate(
            shape=shape,
            overlap=overlap,
            base=base,
            vf=vf,
            progress=progress,
            guidance=guidance,
            domain=domain,
            margin=margin,
            base_offset=base_offset,
            anchors=anchors,
            anchor_strength=anchor_strength,
            height_origin=height_origin,
            height_extent=height_extent,
            vf_profile=vf_profile,
            storage=storage,
            preserve_base=preserve_base,
            probabilities=True,
        )

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
        height_extent: float | None = None,
        vf_profile: dict | None = None,
        preserve_base: bool = False,
    ) -> torch.Tensor:
        if blocks is not None:
            if shape is not None:
                raise ValueError("blocks and shape cannot be provided together.")
            shape = self.shape_from_blocks(blocks, overlap)
        elif shape is None:
            raise TypeError("blocks or shape must be provided.")
        return self._generate(
            overlap=overlap,
            base=base,
            vf=vf,
            storage=storage,
            progress=progress,
            shape=shape,
            guidance=guidance,
            domain=domain,
            margin=margin,
            base_offset=base_offset,
            anchors=anchors,
            anchor_strength=anchor_strength,
            height_origin=height_origin,
            height_extent=height_extent,
            vf_profile=vf_profile,
            preserve_base=preserve_base,
            probabilities=False,
        )

    @torch.no_grad()
    def _generate(
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
        height_extent: float | None = None,
        vf_profile: dict | None = None,
        storage: str = "auto",
        preserve_base: bool = False,
        probabilities: bool = False,
    ) -> torch.Tensor:
        self.stats = None
        guidance = self.generator.validate_guidance(guidance)
        margin = self.generator.default_margin if margin is None else margin
        output_shape = parse_shape(shape)
        plan = self._generation_plan(
            output_shape,
            overlap,
            margin,
        )
        height_extent = self.generator.validate_height(
            output_shape, domain, height_origin, height_extent
        )
        vf_profile = self.generator.validate_profile_request(
            vf_profile, output_shape[0], domain, height_origin, height_extent, vf
        )
        storage = self.select_storage(
            storage,
            estimate_memory(
                output_shape,
                self.generator.num_phases,
                tile_size=plan.tile_size,
                margin=margin,
                overlap=overlap,
                probabilities=probabilities,
                base_shape=volume_shape(base),
            ),
        )
        tiles = make_tiles(plan)
        tile_anchors = self.prepare_anchors(anchors, tiles, plan, anchor_strength)
        vf = self.generator.prepare_vf(vf)
        height_domain = domain
        domain = self.generator.prepare_domain(domain)
        base = self.prepare_base(base, plan, offset=base_offset, preserve=preserve_base)
        if anchor_strength > 0:
            base = validate_base_anchors(base, anchors, output_shape, plan.margin)
        vf_profile = self.validate_profile_constraints(
            vf_profile,
            output_shape,
            plan,
            tiles,
            tile_anchors if anchor_strength == 1 else ((),) * len(tiles),
            base,
            height_domain,
            height_origin,
            height_extent,
        )
        current, next_state = self.make_states(plan, storage)
        labels = None if probabilities else torch.empty(plan.shape, dtype=torch.uint8)
        self.fill_noise(current, tiles)
        current = self.sample(
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
            height_extent=height_extent,
            vf_profile=vf_profile,
            height_domain=height_domain,
        )
        self.stats = output_plan(plan, output_shape)
        if labels is not None:
            return crop_output(labels, output_shape, margin)
        return collect_probabilities(
            current, tiles, output_shape, margin, self.generator.num_phases
        )

    def validate_profile_constraints(
        self, spec, shape, plan, tiles, anchors, base, domain, origin, extent
    ):
        if spec is None or (not any(anchors) and (base is None or not base.preserve)):
            return spec
        generator = self.generator
        target = generator.profile_condition(
            spec, shape[0], domain, origin, extent=extent
        ).cpu() * (shape[1] * shape[2])
        known = torch.zeros_like(target)
        count = torch.zeros(1, 1, shape[0])
        for tile, planes in zip(tiles, anchors, strict=True):
            region = tuple(
                slice(max(p.start, plan.margin), min(p.stop, plan.margin + n))
                for p, n in zip(tile.target, shape)
            )
            if any(p.start >= p.stop for p in region):
                continue
            tile_shape = tuple(p.stop - p.start for p in tile.source)
            values = torch.zeros(1, generator.num_phases, *tile_shape)
            mask = torch.zeros(1, 1, *tile_shape, dtype=torch.bool)
            if planes:
                condition = encode_anchors(
                    tuple(replace(p, image=p.image.cpu()) for p in planes),
                    1,
                    generator.num_phases,
                    tile_shape,
                    torch.device("cpu"),
                    torch.float32,
                )
                values = (condition.image + 1) * 0.5
                mask = condition.mask.bool()
            if (
                base is not None
                and base.preserve
                and (parts := base.intersection(tile.source)) is not None
            ):
                source, local = parts
                values[(..., *local)] = (base.clean[(..., *source)].cpu() + 1) * 0.5
                mask[(..., *local)] = True
            local = tuple(
                slice(p.start - t.start, p.stop - t.start)
                for p, t in zip(region, tile.source)
            )
            mask = mask[(..., *local)]
            depth = slice(region[0].start - plan.margin, region[0].stop - plan.margin)
            known[..., depth] += (values[(..., *local)] * mask).sum((-1, -2))
            count[..., depth] += mask.sum((-1, -2))
        free = shape[1] * shape[2] - count
        if bool(((target < known - 1e-4) | (target > known + free + 1e-4)).any()):
            raise ValueError("profile conflicts with fixed anchors or preserved base.")
        return spec

    def prepare_anchors(self, anchors, tiles, plan, strength):
        strength = self.generator.validate_anchor_strength(strength)
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
        if type(overlap) is not int or overlap < 0:
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
        if type(margin) is not int or margin < 0:
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

    def prepare_base(self, base, plan, offset=None, preserve=False) -> Base | None:
        shape = tuple(n - 2 * plan.margin for n in plan.shape)
        return prepare_base(
            base,
            self.generator.num_phases,
            shape,
            plan.margin,
            offset,
            plan.base_shell,
            preserve,
        )

    def select_storage(
        self,
        storage: str,
        estimate: MemoryEstimate,
    ) -> str:
        return select_storage(storage, estimate, self.generator)

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
        height_extent: float | None = None,
        vf_profile: dict | None = None,
        height_domain: int | None = None,
        tile_conditions=None,
    ) -> VolumeState:
        generator = self.generator
        tile_buffer = TileBuffer(
            generator.num_phases,
            tuple(min(plan.tile_size, size) for size in plan.shape),
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
                    base.condition(
                        current, generator.diffusion, transition + 1, plan.tile_size
                    )
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
                    height_extent=height_extent,
                    vf_profile=vf_profile,
                    height_domain=height_domain,
                    tile_conditions=tile_conditions,
                    base=base,
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
        height_extent: float | None = None,
        vf_profile: dict | None = None,
        height_domain: int | None = None,
        tile_conditions=None,
        base: Base | None = None,
    ) -> None:
        generator = self.generator
        if tile_buffer is None:
            tile_buffer = TileBuffer(
                generator.num_phases,
                tuple(min(plan.tile_size, size) for size in plan.shape),
                current.values.device.type == "cpu" and generator.device.type == "cuda",
            )
        fusion.pred_sum.zero_()
        fusion.weight_sum.zero_()
        layer_start = 0
        for index, tile in enumerate(tiles):
            values = tile_buffer.read(
                current,
                tile.source,
                generator.device,
            )
            conditions = {}
            if tile_conditions is not None:
                conditions.update(tile_conditions(tile))
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
                conditions.update(
                    {
                        "anchor_image": anchor.image,
                        "anchor_mask": anchor.mask,
                        "anchor_strength": anchor_strength,
                    }
                )
            if generator.height_data is not None:
                conditions["height"] = generator.height_condition(
                    values.shape[-3:],
                    height_domain,
                    height_origin,
                    tile.source[0].start - plan.margin,
                    height_extent,
                )
            tile_vf = vf
            if vf_profile is not None:
                profile = generator.profile_condition(
                    vf_profile,
                    values.shape[-3],
                    height_domain,
                    height_origin,
                    tile.source[0].start - plan.margin,
                    height_extent,
                )
                conditions["profile"] = profile
                tile_vf = profile.mean(-1)
            pred = (
                None if base is None else base.prediction(tile.source, generator.device)
            )
            if pred is None:
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
                        vf=tile_vf,
                        **conditions,
                    )
            expected = (1, generator.num_phases, *values.shape[-3:])
            if pred.shape != expected:
                raise ValueError(f"model prediction must have shape {expected}.")
            add_prediction(fusion, tile, pred, tile_buffer, plan.overlap)
            next_z = (
                tiles[index + 1].source[0].start
                if index + 1 < len(tiles)
                else plan.shape[0]
            )
            if next_z != tile.source[0].start:
                self.flush_slab(
                    current,
                    next_state,
                    tiles[layer_start : index + 1],
                    fusion,
                    transition,
                    labels,
                    next_z,
                    base=base,
                )
                layer_start = index + 1

    def flush_slab(
        self,
        current: VolumeState,
        next_state: VolumeState,
        tiles: tuple[Tile, ...],
        fusion: Fusion,
        transition: int,
        labels: torch.Tensor | None,
        stop: int,
        base: Base | None = None,
    ) -> None:
        generator = self.generator
        for tile in tiles:
            target = (slice(tile.source[0].start, stop), *tile.target[1:])
            for global_region, local_region in fusion.regions(target):
                region = (slice(None), slice(None), *local_region)
                weights = fusion.weight_sum[region]
                if not bool((weights > 0).all()):
                    raise RuntimeError("blend weights must cover the output slab.")
                clean = fusion.pred_sum[region] / weights
                if base is not None:
                    base.constrain(clean, global_region)
                if labels is None:
                    previous = current.read(global_region).float()
                    updated = generator.diffusion.sample_posterior(
                        previous,
                        clean,
                        transition,
                    )
                    if base is not None:
                        base.constrain(
                            updated, global_region, generator.diffusion, transition
                        )
                    next_state.write(global_region, updated)
                else:
                    write_output(labels, global_region, clean)
                fusion.pred_sum[region].zero_()
                weights.zero_()

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
