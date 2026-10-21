import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import yaml
from PIL import Image

from simul.src.anode import GRAPHITE, SILICON, check_anode, pack_anode
from simul.src.geometry import check_geometry, pack
from src.config.files import PROJECT_ROOT
from src.plane import AXES

PALETTE = [0, 0, 0, 140, 140, 140, 255, 255, 255] + [0, 0, 0] * 253

# geometry.kind -> (defaults, check, pack, measured fraction name -> label)
STRUCTURES = {
    "particles": (
        {"big_elongation": 1.0, "radius_gradient": [1.0, 1.0]},
        check_geometry,
        pack,
        {"big_vf": 2, "small_vf": 1},
    ),
    "anode": (
        {"size_spread": 0.0, "min_section": 0},
        check_anode,
        pack_anode,
        {"graphite_vf": GRAPHITE, "silicon_vf": SILICON},
    ),
}


@dataclass(frozen=True)
class Export:
    volumes: tuple[Path, ...]
    slices: dict[int, tuple[Path, ...]]
    measured: tuple[dict[str, float], ...] = ()


def generate(cfg: dict) -> Export:
    output = cfg["output"]
    geometry = dict(cfg["geometry"])
    kind = geometry.pop("kind", "particles")
    if kind not in STRUCTURES:
        raise ValueError(f"geometry kind must be one of {sorted(STRUCTURES)}.")
    defaults, check, pack_volume, fractions = STRUCTURES[kind]
    geometry = {**defaults, **geometry}
    check(**geometry)
    count = output["count"]
    if type(count) is not int or count < 1:
        raise ValueError("output count must be a positive integer.")
    seed = output.get("seed")
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("output seed must be a non-negative integer or null.")
    rng = np.random.default_rng(seed)
    root = PROJECT_ROOT / output["data_dir"]
    vol_dir, slice_dirs = make_dirs(root)
    volumes: list[Path] = []
    slices: dict[int, list[Path]] = {axis: [] for axis in AXES}
    measured = []

    jammed = False
    for idx in range(count):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            vol = pack_volume(**geometry, rng=rng)
        jammed |= bool(caught)
        stem = f"volume_{idx:03d}"
        path = vol_dir / f"{stem}.tiff"
        tifffile.imwrite(path, vol)
        volumes.append(path)
        measured.append(
            {key: float((vol == label).mean()) for key, label in fractions.items()}
        )

        for axis in AXES:
            paths = save_slices(
                vol,
                slice_dirs[axis],
                axis=axis,
                stem=stem,
            )
            slices[axis].extend(paths)

    if jammed:
        means = {key: np.mean([item[key] for item in measured]) for key in measured[0]}
        warnings.warn(
            "random packing jammed below the configured fractions: "
            + ", ".join(
                f"{key} {means[key]:.3f} (target {geometry[key]})" for key in means
            ),
            stacklevel=2,
        )
    # Geometry holds targets; packing can jam below them, so record the result.
    (root / "simulation.yaml").write_text(
        yaml.safe_dump(
            {
                "geometry": {"kind": kind, **geometry},
                "height_axis": "z",
                "count": count,
                "seed": seed,
                "measured": measured,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return Export(
        volumes=tuple(volumes),
        slices={axis: tuple(paths) for axis, paths in slices.items()},
        measured=tuple(measured),
    )


def make_dirs(root: Path) -> tuple[Path, dict[int, Path]]:
    vol_dir = root / "volumes"
    slice_dirs = {axis: root / "slices" / str(axis) for axis in AXES}
    paths = (vol_dir,) + tuple(slice_dirs.values())
    for path in paths:
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(f"output directory is not empty: {path}")
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return vol_dir, slice_dirs


def save_slices(
    vol: np.ndarray,
    dst: Path,
    axis: int,
    stem: str,
) -> list[Path]:
    stack = np.moveaxis(vol, axis, 0)
    paths: list[Path] = []
    for idx in range(stack.shape[0]):
        path = dst / f"{stem}_{axis}_{idx:03d}.png"
        image = Image.fromarray(stack[idx])
        image.putpalette(PALETTE)
        image.save(path)
        paths.append(path)
    return paths
