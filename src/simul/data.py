from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from ..data import AXES
from ..misc import require_int
from .geometry import make_geometry

PALETTE = [0, 0, 0, 140, 140, 140, 255, 255, 255] + [0, 0, 0] * 253


@dataclass(frozen=True)
class SimulationExport:
    volumes: tuple[Path, ...]
    slices: dict[int, tuple[Path, ...]]


def save_simulation(
    data_dir: str | Path,
    *,
    count: int,
    geometry: Mapping[str, object],
    axes: Sequence[int] = AXES,
) -> SimulationExport:
    require_int("count", count)
    if count <= 0:
        raise ValueError("count must be positive.")
    if not isinstance(geometry, Mapping):
        raise TypeError("geometry must be a mapping.")
    axes = _validate_axes(axes)

    root = Path(data_dir)
    dirs = _make_simulation_dirs(root, axes)
    volumes: list[Path] = []
    slices: dict[int, list[Path]] = {axis: [] for axis in axes}

    cfg = dict(geometry)
    for index in range(count):
        geo = make_geometry(**cfg)
        vol = geo.labels
        _validate_volume(vol, int(cfg["size"]))
        stem = f"volume_{index:03d}"
        meta = {
            "generator": "packing",
            "index": index,
            **geo.report.as_dict(),
        }
        path = dirs["volumes"] / f"{stem}.tif"
        _write_volume(
            path,
            vol,
            meta=meta,
            labels=_phase_labels(float(cfg.get("big_elongation", 1.0))),
        )
        volumes.append(path)

        for axis in axes:
            paths = _save_axis_slices(
                vol,
                dirs["slices"][axis],
                axis=axis,
                stem=stem,
            )
            slices[axis].extend(paths)

    return SimulationExport(
        volumes=tuple(volumes),
        slices={axis: tuple(paths) for axis, paths in slices.items()},
    )


def _validate_axes(value: object) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("axes must be a list of axis indices.")
    axes = tuple(value)
    if any(not isinstance(axis, int) or isinstance(axis, bool) for axis in axes):
        raise ValueError("axes must contain integer axis indices.")
    if axes != AXES:
        raise ValueError("axes must contain exactly 0, 1, and 2.")
    return axes


def _validate_volume(vol: np.ndarray, size: int) -> None:
    if vol.shape != (size, size, size):
        raise ValueError("generated volume must have shape [size, size, size].")
    if vol.dtype != np.uint8:
        raise ValueError("generated volume must have dtype uint8.")
    if not np.isin(vol, (0, 1, 2)).all():
        raise ValueError("generated volume may contain only labels 0, 1, and 2.")


def _make_simulation_dirs(
    root: Path,
    axes: tuple[int, ...],
) -> dict[str, object]:
    volumes = root / "volumes"
    slices = {axis: root / "slices" / str(axis) for axis in axes}
    for path in (volumes, *slices.values()):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"output directory is not empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
    return {"volumes": volumes, "slices": slices}


def _save_axis_slices(
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
        _write_label_image(path, stack[index])
        paths.append(path)
    return paths


def _phase_labels(big_elongation: float) -> dict[str, str]:
    return {
        "0": "background",
        "1": "small_sphere",
        "2": "big_sphere" if big_elongation == 1.0 else "big_ellipse",
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


def _write_label_image(path: Path, data: np.ndarray) -> None:
    image = Image.fromarray(data)
    image.putpalette(PALETTE)
    image.save(path)
