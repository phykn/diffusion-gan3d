from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.split import training_origin
from src.plane import get_axis
from src.prepare.resize import resize_crop

# Decoding dominates crop cost for large sources. Entries are keyed by file
# state, so a rewritten image is decoded again. Each worker has its own cache.
DECODE_CACHE_BYTES = 256 * 1024**2
_decoded: OrderedDict[tuple[str, int, int], tuple[np.ndarray, int]] = OrderedDict()


def read_labels(path: Path) -> tuple[np.ndarray, int]:
    """Return a read-only 2D uint8 label image and its largest label."""
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key in _decoded:
        _decoded.move_to_end(key)
        return _decoded[key]
    with Image.open(path) as img:
        if getattr(img, "n_frames", 1) != 1:
            raise ValueError(f"training images must contain one 2D frame: {path}")
        data = np.asarray(img)
    if data.ndim != 2:
        raise ValueError(f"image must be two-dimensional: {path}")
    if data.dtype != np.uint8:
        raise ValueError(f"image must use uint8: {path}")
    data.flags.writeable = False
    entry = (data, int(data.max()))
    if data.nbytes <= DECODE_CACHE_BYTES:
        _decoded[key] = entry
        while sum(image.nbytes for image, _ in _decoded.values()) > DECODE_CACHE_BYTES:
            _decoded.popitem(last=False)
    return entry


class RealDataset(Dataset[dict]):
    """Crop crop_size source pixels and resample to patch_size grid cells per edge.

    Records retain source geometry; z is the row direction of xz/yz images.
    """

    def __init__(
        self,
        path_groups: Sequence[Sequence[str | Path]],
        crop_size: int,
        patch_size: int,
        num_phases: int,
        plane: str | int = "xz",
        validation_regions: dict | None = None,
    ) -> None:
        self.path_groups = tuple(
            tuple(Path(path) for path in group) for group in path_groups
        )
        if not self.path_groups or any(not group for group in self.path_groups):
            raise ValueError("path groups must be non-empty.")
        self.crop_size = crop_size
        self.patch_size = patch_size
        self.num_phases = num_phases
        self.axis = get_axis(plane)
        self.height_direction = 0 if self.axis != 0 else None
        self.validation_regions = validation_regions or {}

    def __len__(self) -> int:
        return sum(len(group) for group in self.path_groups)

    def __getitem__(self, path: str | Path) -> dict:
        return self.sample(path)

    def sample(self, path: str | Path, rng=np.random) -> dict:
        """One crop of path; a seeded RandomState picks the same crop every time."""
        path = Path(path).resolve()
        source = self.decode(path)
        excluded = self.validation_regions.get(str(path))
        crop, origin = self.crop_with_origin(source, excluded, rng)
        labels = torch.from_numpy(crop.copy()).long()
        image = resize_crop(labels, self.patch_size, self.num_phases)
        return {
            "image": image,
            "image_id": str(path),
            "source_shape": torch.tensor(source.shape),
            "crop_origin": torch.tensor(origin),
            "height_extent": float(source.shape[self.height_direction])
            if self.height_direction is not None
            else -1.0,
            "height_origin": float(origin[self.height_direction])
            if self.height_direction is not None
            else -1.0,
        }

    def decode(self, path: Path) -> np.ndarray:
        image, largest = read_labels(path)
        if largest >= self.num_phases:
            raise ValueError(
                f"phase labels must be in [0, {self.num_phases - 1}], "
                f"found {largest}: {path}"
            )
        return image

    def crop_with_origin(
        self, img: np.ndarray, excluded: Sequence[int] | None = None, rng=np.random
    ) -> tuple[np.ndarray, tuple[int, int]]:
        height, width = img.shape
        size = self.crop_size
        if size > min(height, width):
            raise ValueError("crop size must fit inside the image.")
        if excluded is None:
            top = int(rng.randint(0, height - size + 1))
            left = int(rng.randint(0, width - size + 1))
        else:
            top, left = training_origin(img.shape, size, excluded, rng)
        return img[top : top + size, left : left + size], (top, left)
