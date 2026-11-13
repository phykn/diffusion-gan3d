import argparse
import math
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.axes import AXIS_NAMES, PLANES, PORE_PHASE
from src.config.io import PROJECT_ROOT, project_path
from src.data.crops import phase_channels, resize_phases
from src.data.images import list_images, read_labels
from src.device import default_device
from src.stats.sections import compare_sections, section_statistics
from src.stats.volume import interface_density, spanning_fractions, tortuosity
from src.storage import torch_save
from src.train.selection import pick_sections

# Independent of the selection reference (seed 0), so the best step is not
# scored on the crops that chose it.
REAL_SEED = 1
REAL_CROPS = 128
TILT_KEYS = ("count", "balance", "mean", "sem")


def parse_args(description, run_help, argv=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--run", type=Path, required=True, help=run_help)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default=default_device(),
        help="Compute device; CUDA, then Apple MPS, then CPU when available.",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Recompute the metrics from cached volumes instead of sampling them.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Result folder (default: checks/rebuild/<run folder name>/); "
        "volumes are cached by run path under cache/.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Few volumes, seeds and anchor cases, for tiny test runs; the "
        "external anchor is cropped at the image's top-left corner.",
    )
    args = parser.parse_args(argv)
    args.run = args.run.resolve()
    if args.out is None:
        args.out = PROJECT_ROOT / "checks" / "rebuild" / args.run.name
    args.out.mkdir(parents=True, exist_ok=True)
    return args


def cache_dir(root, run, smoke=None):
    run = run.resolve()
    identity = sha256(str(run).encode("utf-8")).hexdigest()[:16]
    directory = root / "cache" / f"{run.name}-{identity}"
    directory.mkdir(parents=True, exist_ok=True)
    if smoke is None:
        return directory

    marker = directory / "SMOKE"
    if marker.exists() != smoke and any(directory.glob("*.pt")):
        kind = "full" if smoke else "--smoke"
        raise SystemExit(f"{directory} holds {kind} volumes; pass another --out.")
    if smoke:
        marker.touch()
    else:
        marker.unlink(missing_ok=True)
    return directory


def cached(path, reuse, make):
    if reuse and path.is_file():
        return torch.load(path, weights_only=True)
    value = make()
    torch_save(value, path)
    return value


def archived_generators(run):
    """Archived EMA exports of a run by step."""
    paths = sorted((run / "checkpoints").glob("generator_*.pt"))
    return {int(path.stem.split("_")[1]): path for path in paths}


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def normalized(errors, noise):
    """Mean of the errors over the real half-split errors, as selection scores."""
    return float(np.mean([v / max(n, 1e-6) for v, n in zip(errors, noise)]))


def describe_volumes(volumes, reference, phases, noise):
    planes, scores, clusters = {}, [], []
    for axis, plane in enumerate(PLANES):
        generated = section_statistics(pick_sections(volumes, axis, phases))
        errors = compare_sections(reference, generated)
        planes[plane] = errors._asdict()
        scores.append(normalized(errors, noise))
        clusters.append(generated.clusters[0])
    return {
        "samples": len(volumes),
        "planes": planes,
        "structure": float(np.mean(scores)),
        "phase0_fraction": mean_std(
            [(volume == PORE_PHASE).float().mean() for volume in volumes]
        ),
        "interface_density": mean_std(
            [interface_density(volume, (0, 1, 2)) for volume in volumes]
        ),
    }, clusters


def connectivity(volumes):
    result = {}
    for phase in (0, 1):
        spans = np.asarray([spanning_fractions((v == phase).numpy()) for v in volumes])
        result[f"percolation_phase{phase}"] = {
            axis: mean_std(spans[:, index]) for index, axis in enumerate(AXIS_NAMES)
        }
    taus = np.asarray(
        [
            [tortuosity(v, PORE_PHASE, axis, device="cpu") for axis in range(3)]
            for v in volumes
        ]
    )
    result["tortuosity_phase0"] = {
        axis: mean_std(taus[:, index]) for index, axis in enumerate(AXIS_NAMES)
    }
    return result


def first_image(config):
    path = list_images(project_path(config.data.images["xy"][0]))[0]
    return path, read_labels(path)


def real_crops(config, patch_size):
    labels = first_image(config)[1]
    size = config.data.crop_size
    height, width = labels.shape
    # Preserve the published half-split normalization by drawing top then left.
    rng = np.random.RandomState(REAL_SEED)
    crops = []
    for _ in range(REAL_CROPS):
        top = int(rng.randint(0, height - size + 1))
        left = int(rng.randint(0, width - size + 1))
        crops.append(labels[top : top + size, left : left + size])
    probs = phase_channels(torch.from_numpy(np.stack(crops)), config.data.num_phases)
    return resize_phases(probs, (patch_size,) * 2)


def read_events(log_dir):
    events = EventAccumulator(str(log_dir), size_guidance={"scalars": 0}).Reload()
    for tag in events.Tags()["scalars"]:
        for event in events.Scalars(tag):
            yield tag, event


def read_metrics(log_dir):
    rows = {}
    for tag, event in read_events(log_dir):
        row = rows.setdefault(event.step, {"step": event.step})
        row[tag] = None if math.isnan(event.value) else event.value
    return [rows[step] for step in sorted(rows)]


def wall_minutes(log_dir):
    times = [event.wall_time for _, event in read_events(log_dir)]
    return (max(times) - min(times)) / 60
