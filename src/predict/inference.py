from collections.abc import Sequence
from pathlib import Path

import torch

from src.anchor import PlaneAnchor
from src.build.predict import build_generator
from src.config.data import get_sizes
from src.config.files import find_train_config
from src.config.generation import load_generation_settings
from src.config.train import load_train_config
from src.predict.memory import estimate_memory, select_storage
from src.predict.random import seeded_rng
from src.predict.tiling.layout import parse_shape
from src.predict.tiling.sampler import TiledGenerator
from src.prepare.resize import resize_crop


class InferenceAPI:
    def __init__(
        self,
        weights: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.weights = _resolve_weights(weights)
        self.settings = load_generation_settings()
        cfg = load_train_config(find_train_config(self.weights))
        self.generator = build_generator(self.weights, cfg, device=self.device)
        data = cfg["data"]
        self.data = data
        self._crop_size = data["crop_size"]
        self.scaled = TiledGenerator(self.generator)

    @property
    def input_size(self) -> int:
        return self.generator.patch_size

    @property
    def crop_size(self) -> int:
        return self._crop_size

    @property
    def num_phases(self) -> int:
        return self.generator.num_phases

    def prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape != (self.crop_size, self.crop_size):
            raise ValueError(
                f"image must be a {self.crop_size} x {self.crop_size} original crop."
            )
        return resize_crop(image, get_sizes(self.data)[1], self.num_phases)

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
        if sum(value is not None for value in (blocks, shape, size)) > 1:
            raise ValueError("blocks and shape and size cannot be provided together.")
        tiled = blocks is not None or shape is not None
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

    def check_memory(self, estimate, *, storage="auto", tiled=True):
        # A direct reverse chain holds its entire state on the model device.
        if not tiled:
            storage = self.device.type
        return select_storage(storage, estimate, self.generator)

    def generate(
        self,
        anchors: Sequence[PlaneAnchor] = (),
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
        anchors = _validate_anchors(anchors)
        tiled = _validate_generation_options(
            blocks, shape, size, base, base_offset, preserve_base, storage, overlap
        )

        guidance = self.settings.guidance if guidance is None else guidance
        anchor_strength = (
            self.settings.anchor_strength
            if anchor_strength is None
            else anchor_strength
        )
        overlap = self.settings.overlap if overlap is None else overlap

        with seeded_rng(seed, self.device):
            if not tiled:
                return self.generator.generate(
                    anchors=anchors,
                    vf=vf,
                    size=size,
                    anchor_strength=anchor_strength,
                    guidance=guidance,
                    domain=domain,
                    height_origin=height_origin,
                    height_extent=height_extent,
                    vf_profile=vf_profile,
                )

            return self.scaled.generate(
                blocks=blocks,
                shape=shape,
                overlap=overlap,
                base=base,
                base_offset=base_offset,
                preserve_base=preserve_base,
                anchors=anchors,
                anchor_strength=anchor_strength,
                vf=vf,
                storage=storage,
                progress=progress,
                guidance=guidance,
                domain=domain,
                height_origin=height_origin,
                height_extent=height_extent,
                vf_profile=vf_profile,
            )

    def generate_probs(
        self,
        anchors=(),
        vf=None,
        domain=None,
        seed=None,
        guidance=None,
        anchor_strength=None,
        height_origin=0.0,
        height_extent=None,
        vf_profile=None,
        *,
        size=None,
        shape=None,
        blocks=None,
        overlap=None,
        storage="auto",
        progress=False,
        base=None,
        base_offset=None,
        preserve_base=False,
    ):
        tiled = _validate_generation_options(
            blocks, shape, size, base, base_offset, preserve_base, storage, overlap
        )

        options = {} if size is None else {"size": size}
        sampler = self.generator
        if tiled:
            overlap = self.settings.overlap if overlap is None else overlap
            if blocks is not None:
                shape = self.scaled.shape_from_blocks(blocks, overlap)
            sampler = self.scaled
            options = dict(
                shape=shape,
                overlap=overlap,
                storage=storage,
                progress=progress,
                base=base,
                base_offset=base_offset,
                preserve_base=preserve_base,
            )
        with seeded_rng(seed, self.device):
            return sampler.generate_probs(
                anchors=_validate_anchors(anchors),
                vf=vf,
                domain=domain,
                guidance=self.settings.guidance if guidance is None else guidance,
                anchor_strength=self.settings.anchor_strength
                if anchor_strength is None
                else anchor_strength,
                height_origin=height_origin,
                height_extent=height_extent,
                vf_profile=vf_profile,
                **options,
            )


def _validate_generation_options(
    blocks, shape, size, base, base_offset, preserve_base, storage, overlap
) -> bool:
    tiled = blocks is not None or shape is not None
    if blocks is not None and shape is not None:
        raise ValueError("blocks and shape cannot be provided together.")
    if tiled and size is not None:
        raise ValueError("size cannot be combined with blocks or shape.")
    if not tiled and base is not None:
        raise ValueError("base requires blocks or shape.")
    if base is None and (base_offset is not None or preserve_base):
        raise ValueError("base_offset and preserve_base require base.")
    if not tiled and (storage != "auto" or overlap is not None):
        raise ValueError("storage and overlap apply only to scale-up.")
    return tiled


def _resolve_device(device: str | torch.device | None) -> torch.device:
    resolved = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("device must be CPU or CUDA.")
    return resolved


def _resolve_weights(weights: str | Path) -> Path:
    path = Path(weights).expanduser().resolve()
    if path.is_dir():
        path = path / "generator.pt"
    if not path.is_file():
        raise FileNotFoundError(f"generator weights do not exist: {path}")
    return path


def _validate_anchors(anchors: Sequence[PlaneAnchor]) -> tuple[PlaneAnchor, ...]:
    values = tuple(anchors)
    if any(not isinstance(anchor, PlaneAnchor) for anchor in values):
        raise TypeError("anchors must contain only PlaneAnchor values.")
    return values
