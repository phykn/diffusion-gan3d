from collections.abc import Sequence
from dataclasses import replace

import torch

from src.predict.anchor import (
    AnchorGuidance,
    PlaneAnchor,
    anchor_origin,
    encode_anchors,
)
from src.predict.generator import Generator, profile_feasible
from src.predict.memory import estimate_memory, select_storage
from src.predict.tiling.layout import (
    DEFAULT_OVERLAP,
    TilePlan,
    check_overlap,
    make_plan,
    make_tiles,
    output_plan,
    parse_shape,
)
from src.predict.tiling.sampler import TileSampler
from src.predict.volume_condition import (
    VolumeCondition,
    prepare_base,
    validate_base_anchors,
    volume_shape,
)


class TiledGenerator:
    """Prepare LR geometry and conditions for the shared tile sampler."""

    def __init__(self, generator: Generator) -> None:
        self.generator = generator
        self.stats: TilePlan | None = None

    def plan(
        self,
        shape: int | Sequence[int],
        overlap: int = DEFAULT_OVERLAP,
    ) -> TilePlan:
        shape = parse_shape(shape)
        factor = self.generator.model.downsample_factor
        tile_size = self.generator.patch_size
        check_overlap(tile_size, overlap)
        if tile_size % factor:
            raise ValueError(
                "patch_size must be divisible by the denoiser "
                f"downsample factor ({factor})."
            )
        if any(size < tile_size for size in shape):
            raise ValueError("shape must not be smaller than patch_size.")
        return make_plan(shape, tile_size, overlap)

    def generate(self, *, blocks=None, shape=None, **options) -> torch.Tensor:
        """Tiled uint8 labels for blocks or an LR shape (see _generate)."""
        return self._generate(
            self._resolve_shape(blocks, shape, options), probabilities=False, **options
        )

    def generate_probs(self, *, blocks=None, shape=None, **options) -> torch.Tensor:
        """Tiled phase fractions for blocks or an LR shape (see _generate)."""
        return self._generate(
            self._resolve_shape(blocks, shape, options), probabilities=True, **options
        )

    def _resolve_shape(self, blocks, shape, options):
        if (blocks is None) == (shape is None):
            raise ValueError("provide exactly one of blocks or shape.")
        if blocks is None:
            return shape
        return self.shape_from_blocks(blocks, options.get("overlap", DEFAULT_OVERLAP))

    @torch.no_grad()
    def _generate(
        self,
        shape: int | Sequence[int],
        overlap: int = DEFAULT_OVERLAP,
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
        domain = self.generator.resolve_domain(domain)
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
        # A preserved base may not fit the training profile; only a requested
        # profile is checked against it.
        if vf_profile is None and vf is None and not preserve_base:
            vf_profile = self.generator.default_profile(domain)
        self.generator.validate_profile_request(
            vf_profile, output_shape[0], domain, height_origin, height_extent, vf
        )
        storage = select_storage(
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
            self.generator,
        )
        tiles = make_tiles(plan)
        anchor_strength = self.generator.validate_anchor_strength(anchor_strength)
        tile_anchors = (
            self.prepare_anchors(anchors, tiles, plan)
            if anchor_strength > 0
            else ((),) * len(tiles)
        )
        vf = self.generator.prepare_vf(vf)
        base = self.prepare_base(base, plan, offset=base_offset, preserve=preserve_base)
        if anchor_strength > 0:
            validate_base_anchors(base, anchors, output_shape, plan.margin)
        self.validate_profile_constraints(
            vf_profile,
            output_shape,
            plan,
            tiles,
            base,
            domain,
            height_origin,
            height_extent,
        )

        def guidance_for(tile, index):
            if not tile_anchors[index]:
                return None
            anchor = encode_anchors(
                tile_anchors[index],
                1,
                self.generator.num_phases,
                tuple(part.stop - part.start for part in tile.source),
                self.generator.device,
                torch.float32,
                validate=False,
            )
            return AnchorGuidance(anchor.image, anchor.mask, anchor_strength)

        def conditions(tile, index):
            generator = self.generator
            shape = tuple(part.stop - part.start for part in tile.source)
            result = {"vf": vf}
            offset = tile.source[0].start - plan.margin
            result["height"] = generator.height_condition(
                shape, domain, height_origin, offset, height_extent
            )
            if vf_profile is not None:
                profile = generator.profile_condition(
                    vf_profile,
                    shape[0],
                    domain,
                    height_origin,
                    offset,
                    height_extent,
                )
                result.update(profile=profile, vf=profile.mean(-1))
            return result

        output = TileSampler(self.generator).sample(
            tiles,
            plan,
            base,
            self.generator.prepare_domain(domain),
            probabilities=probabilities,
            storage=storage,
            progress=progress,
            guidance=guidance,
            tile_conditions=conditions,
            tile_guidance=guidance_for if any(tile_anchors) else None,
        )
        self.stats = output_plan(plan, output_shape)
        return output

    def validate_profile_constraints(
        self, spec, shape, plan, tiles, base, domain, origin, extent
    ):
        if spec is None or base is None or not base.preserve:
            return
        generator = self.generator
        target = generator.profile_condition(
            spec, shape[0], domain, origin, extent=extent
        ).cpu() * (shape[1] * shape[2])
        known = torch.zeros_like(target)
        count = torch.zeros(1, 1, shape[0])
        for tile in tiles:
            region = tuple(
                slice(max(p.start, plan.margin), min(p.stop, plan.margin + n))
                for p, n in zip(tile.target, shape)
            )
            if any(p.start >= p.stop for p in region):
                continue
            tile_shape = tuple(p.stop - p.start for p in tile.source)
            values = torch.zeros(1, generator.num_phases, *tile_shape)
            mask = torch.zeros(1, 1, *tile_shape, dtype=torch.bool)
            if (parts := base.intersection(tile.source)) is not None:
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
        if not profile_feasible(target, known, free):
            raise ValueError("profile conflicts with the preserved base.")

    def prepare_anchors(self, anchors, tiles, plan):
        shape = tuple(size - 2 * plan.margin for size in plan.shape)
        placed = []
        for anchor in anchors:
            # A bool would pass the range checks and shift into a valid index.
            if type(anchor.axis) is not int or anchor.axis not in (0, 1, 2):
                raise ValueError("anchor.axis must be one of 0, 1, or 2.")
            if (
                type(anchor.index) is not int
                or not 0 <= anchor.index < shape[anchor.axis]
            ):
                raise ValueError("anchor.index is outside the generated volume.")
            axes = tuple(axis for axis in range(3) if axis != anchor.axis)
            row, col = anchor_origin(
                anchor, anchor.image.shape[-2:], tuple(shape[axis] for axis in axes)
            )
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
        overlap: int = DEFAULT_OVERLAP,
    ) -> tuple[int, int, int]:
        counts = parse_shape(blocks)
        patch_size = self.generator.patch_size
        check_overlap(patch_size, overlap)
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

    def prepare_base(
        self, base, plan, offset=None, preserve=False
    ) -> VolumeCondition | None:
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
