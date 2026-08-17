from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import torch

from ..anchor import PlaneAnchor
from ..build import load_generator
from ..config import find_train_config, load_generation_settings
from ..scale import ScaledGenerator
from ..train.weights import GENERATOR_FILE
from ..utils import load_yaml


class InferenceAPI:
    """Inference-only facade for direct, anchored, and scaled generation."""

    def __init__(
        self,
        weights: str | Path,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.weights = _resolve_weights(weights)
        self.settings = load_generation_settings()
        self.generator = load_generator(self.weights, device=self.device)
        data = load_yaml(find_train_config(self.weights))["data"]
        self._crop_size = _positive_int(data["crop_size"], "data.crop_size")
        configured_input = _positive_int(data["input_size"], "data.input_size")
        if configured_input != self.generator.patch_size:
            raise ValueError(
                "data.input_size does not match the loaded generator patch size."
            )
        self.scaled = ScaledGenerator(self.generator)

    @property
    def input_size(self) -> int:
        return self.generator.patch_size

    @property
    def crop_size(self) -> int:
        return self._crop_size

    @property
    def num_phases(self) -> int:
        return self.generator.num_phases

    def generate(
        self,
        *,
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
    ) -> torch.Tensor:
        """Generate one categorical volume and return a CPU uint8 tensor.

        No anchors and no scale geometry performs unconditional direct generation.
        Anchors enable direct anchor-conditioned generation. ``blocks`` or ``shape``
        selects scale-up; when anchors are also supplied, their direct result becomes
        the scale-up base.
        """
        anchors = _validate_anchors(anchors)
        scaled = blocks is not None or shape is not None
        if blocks is not None and shape is not None:
            raise ValueError("blocks and shape cannot be provided together.")
        if scaled and size is not None:
            raise ValueError("size cannot be combined with blocks or shape.")
        if not scaled and base is not None:
            raise ValueError("base requires blocks or shape.")
        if base is not None and anchors:
            raise ValueError("base and anchors cannot be provided together.")
        if not scaled and (storage != "auto" or overlap is not None):
            raise ValueError("storage and overlap apply only to scale-up.")
        if not isinstance(progress, bool):
            raise TypeError("progress must be a boolean.")

        guidance = self.settings.guidance if guidance is None else guidance
        anchor_strength = (
            self.settings.anchor_strength
            if anchor_strength is None
            else anchor_strength
        )
        overlap = self.settings.overlap if overlap is None else overlap

        with _seeded_rng(seed, self.device):
            if not scaled:
                return self.generator.generate(
                    anchors=anchors,
                    vf=vf,
                    size=size,
                    anchor_strength=anchor_strength,
                    guidance=guidance,
                    domain=domain,
                )

            if anchors:
                base = self.generator.generate(
                    anchors=anchors,
                    vf=vf,
                    anchor_strength=anchor_strength,
                    guidance=guidance,
                    domain=domain,
                )
            base_offset = _anchor_base_offset(anchors) if anchors else None
            return self.scaled.generate(
                blocks=blocks,
                shape=shape,
                overlap=overlap,
                base=base,
                base_offset=base_offset,
                vf=vf,
                storage=storage,
                progress=progress,
                guidance=guidance,
                domain=domain,
            )


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError("device must be CPU or CUDA.")
    return resolved


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _resolve_weights(weights: str | Path) -> Path:
    path = Path(weights).expanduser().resolve()
    if path.is_dir():
        path = path / GENERATOR_FILE
    if not path.is_file():
        raise FileNotFoundError(f"generator weights do not exist: {path}")
    return path


def _validate_anchors(anchors: Sequence[PlaneAnchor]) -> tuple[PlaneAnchor, ...]:
    if isinstance(anchors, (str, bytes)) or not isinstance(anchors, Sequence):
        raise TypeError("anchors must be a sequence of PlaneAnchor values.")
    values = tuple(anchors)
    if any(not isinstance(anchor, PlaneAnchor) for anchor in values):
        raise TypeError("anchors must contain only PlaneAnchor values.")
    return values


def _anchor_base_offset(
    anchors: Sequence[PlaneAnchor],
) -> tuple[int | None, int | None, int | None]:
    """Keep anchor coordinates global while centering unconstrained axes."""
    fixed_axes = {anchor.axis for anchor in anchors}
    for anchor in anchors:
        if anchor.position is not None:
            fixed_axes.update(axis for axis in range(3) if axis != anchor.axis)
    return tuple(0 if axis in fixed_axes else None for axis in range(3))


@contextmanager
def _seeded_rng(seed: int | None, device: torch.device):
    if seed is None:
        yield
        return
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    devices = []
    if device.type == "cuda":
        index = device.index
        devices = [torch.cuda.current_device() if index is None else index]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        yield
