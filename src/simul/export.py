from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from .. import AXES
from .geometry import pack

PALETTE = [0, 0, 0, 140, 140, 140, 255, 255, 255] + [0, 0, 0] * 253


@dataclass(frozen=True)
class Export:
    volumes: tuple[Path, ...]
    slices: dict[int, tuple[Path, ...]]


def generate(
    *,
    data_dir: str | Path,
    count: int,
    size: int,
    big_radius: int,
    small_radius: int,
    big_vf: float,
    small_vf: float,
    big_elongation: float,
) -> Export:
    root = Path(data_dir)
    vol_dir, slice_dirs = _make_dirs(root)
    volumes: list[Path] = []
    slices: dict[int, list[Path]] = {axis: [] for axis in AXES}

    for idx in range(count):
        vol = pack(
            size=size,
            big_radius=big_radius,
            small_radius=small_radius,
            big_vf=big_vf,
            small_vf=small_vf,
            big_elongation=big_elongation,
        )
        stem = f"volume_{idx:03d}"
        path = vol_dir / f"{stem}.tiff"
        tifffile.imwrite(path, vol)
        volumes.append(path)

        for axis in AXES:
            paths = _save_slices(
                vol,
                slice_dirs[axis],
                axis=axis,
                stem=stem,
            )
            slices[axis].extend(paths)

    return Export(
        volumes=tuple(volumes),
        slices={axis: tuple(paths) for axis, paths in slices.items()},
    )


def _make_dirs(root: Path) -> tuple[Path, dict[int, Path]]:
    vol_dir = root / "volumes"
    slice_dirs = {axis: root / "slices" / str(axis) for axis in AXES}
    paths = (vol_dir, *slice_dirs.values())
    for path in paths:
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(f"output directory is not empty: {path}")
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return vol_dir, slice_dirs


def _save_slices(
    vol: np.ndarray,
    dst: Path,
    *,
    axis: int,
    stem: str,
) -> list[Path]:
    stack = np.moveaxis(vol, axis, 0)
    paths = []
    for idx in range(stack.shape[0]):
        path = dst / f"{stem}_{axis}_{idx:03d}.png"
        _write_image(path, stack[idx])
        paths.append(path)
    return paths


def _write_image(path: Path, data: np.ndarray) -> None:
    img = Image.fromarray(data)
    img.putpalette(PALETTE)
    img.save(path)
