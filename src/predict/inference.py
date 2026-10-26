from collections.abc import Sequence
from pathlib import Path

import torch

from src.anchor import PlaneAnchor
from src.build.predict import build_generator, resolve_weights
from src.config.files import find_saved_config
from src.config.generation import load_generation_settings
from src.config.train import load_train_config
from src.device import check_device, default_device
from src.predict.memory import estimate_memory, select_storage
from src.predict.tiling.generator import TiledGenerator
from src.predict.tiling.layout import parse_shape
from src.prepare.resize import resize_anchor
from src.random import seeded_rng


class LowResolutionAPI:
    """Generate LR label volumes (z, y, x), directly or with overlapping tiles.

    blocks sets the output size as patch + (n - 1) * (patch - 2 * overlap)
    voxels per axis; the tile grid also covers the context margin, so it can
    hold more tiles than blocks. shape and overlap use LR voxels.
    Height coordinates use source pixels.
    Refinement to HR is a separate SuperResolutionAPI operation.
    """

    def __init__(
        self,
        weights: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = check_device(default_device() if device is None else device)
        self.weights = resolve_weights(weights)
        self.settings = load_generation_settings()
        cfg = load_train_config(find_saved_config(self.weights))
        self.generator = build_generator(self.weights, cfg, device=self.device)
        self.data = cfg["data"]
        self.scaled = TiledGenerator(self.generator)

    @property
    def input_size(self) -> int:
        """Model grid edge in LR voxels; crop_size is measured in source pixels."""
        return self.generator.patch_size

    @property
    def crop_size(self) -> int:
        return self.data["crop_size"]

    @property
    def num_phases(self) -> int:
        return self.generator.num_phases

    def prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape != (self.crop_size, self.crop_size):
            raise ValueError(
                f"image must be a {self.crop_size} x {self.crop_size} original crop."
            )
        return resize_anchor(image, self.input_size, self.num_phases)

    def estimate_memory(
        self,
        *,
        blocks=None,
        shape=None,
        size=None,
        overlap=None,
        probabilities=False,
        base_shape=None,
    ):
        tiled = _tiling_mode(blocks, shape, size)
        overlap = self.settings.overlap if overlap is None else overlap
        if blocks is not None:
            shape = self.scaled.shape_from_blocks(blocks, overlap)
        elif shape is None:
            shape = self.input_size if size is None else size
        shape = parse_shape(shape)
        if tiled:
            self.scaled.plan(shape, overlap)
        return estimate_memory(
            shape,
            self.num_phases,
            tile_size=self.input_size if tiled else None,
            margin=self.generator.default_margin,
            overlap=overlap if tiled else 0,
            probabilities=probabilities,
            base_shape=base_shape,
        )

    def check_memory(self, estimate, *, storage="auto"):
        # A direct reverse chain holds its entire state on the model device.
        if not estimate.tiled:
            storage = "cuda" if self.device.type == "cuda" else "cpu"
        return select_storage(storage, estimate, self.generator)

    def generate(
        self,
        anchors: Sequence[PlaneAnchor] = (),
        *,
        blocks: int | Sequence[int] | None = None,
        shape: int | Sequence[int] | None = None,
        size: int | None = None,
        base: torch.Tensor | None = None,
        vf: Sequence[float] | None = None,
        domain: int | None = None,
        seed: int | None = None,
        guidance: float | None = None,
        anchor_strength: float | None = None,
        overlap: int | None = None,
        storage: str = "auto",
        progress: bool = False,
        height_origin: float = 0.0,
        base_offset=None,
        preserve_base: bool = False,
        height_extent: float | None = None,
        vf_profile: dict | None = None,
    ) -> torch.Tensor:
        """Generate uint8 labels; see _run for the options."""
        return self._run(
            anchors,
            probabilities=False,
            blocks=blocks,
            shape=shape,
            size=size,
            base=base,
            vf=vf,
            domain=domain,
            seed=seed,
            guidance=guidance,
            anchor_strength=anchor_strength,
            overlap=overlap,
            storage=storage,
            progress=progress,
            height_origin=height_origin,
            base_offset=base_offset,
            preserve_base=preserve_base,
            height_extent=height_extent,
            vf_profile=vf_profile,
        )

    def generate_probs(
        self,
        anchors: Sequence[PlaneAnchor] = (),
        *,
        blocks: int | Sequence[int] | None = None,
        shape: int | Sequence[int] | None = None,
        size: int | None = None,
        base: torch.Tensor | None = None,
        vf: Sequence[float] | None = None,
        domain: int | None = None,
        seed: int | None = None,
        guidance: float | None = None,
        anchor_strength: float | None = None,
        overlap: int | None = None,
        storage: str = "auto",
        progress: bool = False,
        height_origin: float = 0.0,
        base_offset=None,
        preserve_base: bool = False,
        height_extent: float | None = None,
        vf_profile: dict | None = None,
    ) -> torch.Tensor:
        """Generate phase fractions (C,D,H,W); see _run for the options."""
        return self._run(
            anchors,
            probabilities=True,
            blocks=blocks,
            shape=shape,
            size=size,
            base=base,
            vf=vf,
            domain=domain,
            seed=seed,
            guidance=guidance,
            anchor_strength=anchor_strength,
            overlap=overlap,
            storage=storage,
            progress=progress,
            height_origin=height_origin,
            base_offset=base_offset,
            preserve_base=preserve_base,
            height_extent=height_extent,
            vf_profile=vf_profile,
        )

    def _run(
        self,
        anchors: Sequence[PlaneAnchor],
        *,
        probabilities,
        blocks,
        shape,
        size,
        base,
        vf,
        domain,
        seed,
        guidance,
        anchor_strength,
        overlap,
        storage,
        progress,
        height_origin,
        base_offset,
        preserve_base,
        height_extent,
        vf_profile,
    ) -> torch.Tensor:
        """blocks or shape tile the output; size, or neither, samples one block.

        shape, size and overlap are LR voxels; height values are source pixels.
        Unset guidance, anchor_strength and overlap come from config/gen.yaml.
        """
        anchors = _validate_anchors(anchors)
        tiled = _validate_generation_options(
            blocks, shape, size, base, base_offset, preserve_base, storage, overlap
        )
        common = dict(
            anchors=anchors,
            vf=vf,
            domain=domain,
            guidance=self.settings.guidance if guidance is None else guidance,
            anchor_strength=self.settings.anchor_strength
            if anchor_strength is None
            else anchor_strength,
            height_origin=height_origin,
            height_extent=height_extent,
            vf_profile=vf_profile,
        )
        sampler = self.scaled if tiled else self.generator
        if tiled:
            common.update(
                blocks=blocks,
                shape=shape,
                overlap=self.settings.overlap if overlap is None else overlap,
                base=base,
                base_offset=base_offset,
                preserve_base=preserve_base,
                storage=storage,
                progress=progress,
            )
        else:
            common["size"] = size
        with seeded_rng(seed, self.device):
            if probabilities:
                return sampler.generate_probs(**common)
            return sampler.generate(**common)


def _tiling_mode(blocks, shape, size) -> bool:
    if blocks is not None and shape is not None:
        raise ValueError("blocks and shape cannot be provided together.")
    tiled = blocks is not None or shape is not None
    if tiled and size is not None:
        raise ValueError("size cannot be combined with blocks or shape.")
    return tiled


def _validate_generation_options(
    blocks, shape, size, base, base_offset, preserve_base, storage, overlap
) -> bool:
    tiled = _tiling_mode(blocks, shape, size)
    if not tiled and base is not None:
        raise ValueError("base requires blocks or shape.")
    if base is None and (base_offset is not None or preserve_base):
        raise ValueError("base_offset and preserve_base require base.")
    if not tiled and (storage != "auto" or overlap is not None):
        raise ValueError("storage and overlap apply only to scale-up.")
    return tiled


def _validate_anchors(anchors: Sequence[PlaneAnchor]) -> tuple[PlaneAnchor, ...]:
    values = tuple(anchors)
    if any(not isinstance(anchor, PlaneAnchor) for anchor in values):
        raise TypeError("anchors must contain only PlaneAnchor values.")
    return values
