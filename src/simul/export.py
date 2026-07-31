from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from ..data import AXES
from .config import SimulationConfig
from .geometry import pack

PALETTE = [0, 0, 0, 140, 140, 140, 255, 255, 255] + [0, 0, 0] * 253


@dataclass(frozen=True)
class Export:
    volumes: tuple[Path, ...]
    slices: dict[int, tuple[Path, ...]]


def generate(cfg: SimulationConfig) -> Export:
    root = Path(cfg.output.data_dir)
    volume_dir, slice_dirs = _make_dirs(root)
    volumes: list[Path] = []
    slices: dict[int, list[Path]] = {axis: [] for axis in AXES}

    for index in range(cfg.output.count):
        geo = pack(cfg.geometry)
        vol = geo.labels
        _check_volume(vol, cfg.geometry.size)
        stem = f"volume_{index:03d}"
        meta = {
            "generator": "packing",
            "index": index,
            **geo.report.as_dict(),
        }
        path = volume_dir / f"{stem}.tif"
        _write_volume(
            path,
            vol,
            meta=meta,
            labels=_name_phases(cfg.geometry.big_elongation),
        )
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


def _check_volume(vol: np.ndarray, size: int) -> None:
    if vol.shape != (size, size, size):
        raise ValueError("generated volume must have shape [size, size, size].")
    if vol.dtype != np.uint8:
        raise ValueError("generated volume must have dtype uint8.")
    if not np.isin(vol, (0, 1, 2)).all():
        raise ValueError("generated volume may contain only labels 0, 1, and 2.")


def _make_dirs(root: Path) -> tuple[Path, dict[int, Path]]:
    volume_dir = root / "volumes"
    slice_dirs = {axis: root / "slices" / str(axis) for axis in AXES}
    paths = (volume_dir, *slice_dirs.values())
    for path in paths:
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(f"output directory is not empty: {path}")
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return volume_dir, slice_dirs


def _save_slices(
    vol: np.ndarray,
    dst: Path,
    *,
    axis: int,
    stem: str,
) -> list[Path]:
    stack = np.moveaxis(vol, axis, 0)
    paths = []
    for index in range(stack.shape[0]):
        path = dst / f"{stem}_{axis}_{index:03d}.png"
        _write_image(path, stack[index])
        paths.append(path)
    return paths


def _name_phases(big_elongation: float) -> dict[str, str]:
    return {
        "0": "background",
        "1": "small_sphere",
        "2": "big_sphere" if big_elongation == 1.0 else "big_ellipsoid",
    }


def _write_volume(
    path: Path,
    vol: np.ndarray,
    *,
    meta: Mapping[str, object],
    labels: Mapping[str, str],
) -> None:
    tifffile.imwrite(
        path,
        vol,
        photometric="minisblack",
        metadata={
            "axes": "ZYX",
            "phase_labels": dict(labels),
            "simulation": dict(meta),
        },
    )


def _write_image(path: Path, data: np.ndarray) -> None:
    image = Image.fromarray(data)
    image.putpalette(PALETTE)
    image.save(path)
