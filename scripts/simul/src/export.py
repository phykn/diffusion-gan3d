import warnings
from dataclasses import dataclass

import numpy as np
import tifffile
import yaml
from PIL import Image

from scripts.simul.src.anode import (
    GRAPHITE,
    SILICON,
    check_anode,
    pack_anode,
    vf_profile,
)
from scripts.simul.src.geometry import check_geometry, pack
from src.axes import AXES
from src.config.io import project_path

PALETTE = [0, 0, 0, 140, 140, 140, 255, 255, 255] + [0, 0, 0] * 253


@dataclass(frozen=True)
class Export:
    volumes: tuple
    slices: dict
    measured: tuple
    targets: dict


def generate(cfg):
    output = cfg["output"]
    kind, geometry, pack_volume, phases = prepare_geometry(cfg["geometry"])
    count = output["count"]
    seed = output.get("seed")
    if type(count) is not int or count < 1:
        raise ValueError("output count must be a positive integer.")
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("output seed must be a non-negative integer or null.")

    root = project_path(output["data_dir"])
    volume_dir = root / "volumes"
    slice_dirs = {axis: root / "slices" / str(axis) for axis in AXES}
    directories = (volume_dir, *slice_dirs.values())
    for folder in directories:
        if folder.exists() and (not folder.is_dir() or any(folder.iterdir())):
            raise FileExistsError(f"output directory is not empty: {folder}")
    for folder in directories:
        folder.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    targets = {
        name: float(vf_profile(geometry["size"], 0, geometry[name]).mean())
        for name in phases
    }
    volumes, measured = [], []
    slices = {axis: [] for axis in AXES}
    jammed = set()
    for index in range(count):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            volume = pack_volume(**geometry, rng=rng)
        stem = f"volume_{index:03d}"
        path = volume_dir / f"{stem}.tiff"
        tifffile.imwrite(path, volume)
        volumes.append(path)
        fractions = {
            name: float((volume == label).mean()) for name, label in phases.items()
        }
        measured.append(fractions)
        if caught:
            jammed.update(name for name in phases if fractions[name] < targets[name])
        for axis, folder in slice_dirs.items():
            slices[axis].extend(write_slices(volume, axis, folder, stem))

    if jammed:
        detail = ", ".join(
            f"{name} {np.mean([row[name] for row in measured]):.3f} (target {targets[name]:.3f})"
            for name in sorted(jammed)
        )
        warnings.warn(
            f"random packing jammed below the configured fractions: {detail}",
            stacklevel=2,
        )
    manifest = {
        "geometry": {"kind": kind, **geometry},
        "height_axis": "z",
        "count": count,
        "seed": seed,
        "measured": measured,
    }
    (root / "simulation.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return Export(
        tuple(volumes),
        {axis: tuple(paths) for axis, paths in slices.items()},
        tuple(measured),
        targets,
    )


def prepare_geometry(values):
    values = dict(values)
    kind = values.pop("kind", "particles")
    if kind == "particles":
        geometry = {"big_elongation": 1.0, "radius_gradient": [1.0, 1.0], **values}
        check_geometry(**geometry)
        return kind, geometry, pack, {"big_vf": 2, "small_vf": 1}
    if kind == "anode":
        geometry = {"size_spread": 0.0, "min_section": 0, **values}
        check_anode(**geometry)
        return (
            kind,
            geometry,
            pack_anode,
            {"graphite_vf": GRAPHITE, "silicon_vf": SILICON},
        )
    raise ValueError("geometry kind must be one of ['anode', 'particles'].")


def write_slices(volume, axis, folder, stem):
    for index, labels in enumerate(np.moveaxis(volume, axis, 0)):
        path = folder / f"{stem}_{axis}_{index:03d}.png"
        image = Image.fromarray(labels)
        image.putpalette(PALETTE)
        image.save(path)
        yield path
