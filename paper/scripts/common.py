import argparse
import math
from pathlib import Path

import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.config.io import PROJECT_ROOT, project_path
from src.data.crops import phase_channels, resize_phases
from src.data.images import list_images, read_labels
from src.device import default_device
from src.stats.volume import spanning_fractions, tortuosity
from src.storage import torch_save

# Independent of the selection reference (seed 0), so the best step is not
# scored on the crops that chose it.
REAL_SEED = 1
REAL_CROPS = 128
TILT_KEYS = ("count", "balance", "mean", "sem")


def parse_args(description: str, run_help: str, argv=None) -> argparse.Namespace:
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
        "volumes are cached in its cache/<run folder name>/.",
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


def cache_dir(root: Path, run: Path, smoke: bool | None = None) -> Path:
    """Cache folder of one run, so volumes of another run are never reused.

    An evaluation passes smoke: a SMOKE marker then keeps --smoke and full
    evaluations from sharing one cache.
    """
    path = root / "cache" / run.name
    path.mkdir(parents=True, exist_ok=True)
    if smoke is not None:
        marker = path / "SMOKE"
        if marker.exists() != smoke and any(path.glob("*.pt")):
            kind = "full" if smoke else "--smoke"
            raise SystemExit(f"{path} holds {kind} volumes; pass another --out.")
        if smoke:
            marker.touch()
        else:
            marker.unlink(missing_ok=True)
    return path


def cached(path: Path, reuse: bool, make):
    if reuse and path.is_file():
        return torch.load(path, weights_only=True)
    value = make()
    torch_save(value, path)
    return value


def archived_generators(run: Path) -> dict[int, Path]:
    """Archived EMA exports of a run by step."""
    paths = sorted((run / "checkpoints").glob("generator_*.pt"))
    return {int(path.stem.split("_")[1]): path for path in paths}


def mean_std(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def normalized(errors, noise) -> float:
    """Mean of the errors over the real half-split errors, as selection scores."""
    return float(np.mean([v / max(n, 1e-6) for v, n in zip(errors, noise)]))


def connectivity(volumes) -> dict:
    """Spanning fractions of both phases and phase-0 tortuosity along z, y, x."""
    result = {}
    for phase in (0, 1):
        spans = np.asarray([spanning_fractions((v == phase).numpy()) for v in volumes])
        result[f"percolation_phase{phase}"] = {
            axis: mean_std(spans[:, index]) for index, axis in enumerate("zyx")
        }
    taus = np.asarray(
        [[tortuosity(v, 0, axis, device="cpu") for axis in range(3)] for v in volumes]
    )
    result["tortuosity_phase0"] = {
        axis: mean_std(taus[:, index]) for index, axis in enumerate("zyx")
    }
    return result


def first_image(config) -> tuple[Path, np.ndarray]:
    """The first xy training image, whose crops stand for the real data."""
    path = list_images(project_path(config.data.images["xy"][0]))[0]
    return path, read_labels(path)


def real_crops(config, patch_size: int) -> torch.Tensor:
    """Phase fractions of fixed random crops of the first xy training image.

    The crop corners are those behind paper/assets: numpy RandomState(REAL_SEED)
    draws the top, then the left of each crop. Structure scores divide by the
    half-split errors of these crops, which move up to threefold with the crop
    seed, so other crops would make the scores incomparable with paper/assets.
    """
    labels = first_image(config)[1]
    size = config.data.crop_size
    height, width = labels.shape
    rng = np.random.RandomState(REAL_SEED)
    crops = []
    for _ in range(REAL_CROPS):
        top = int(rng.randint(0, height - size + 1))
        left = int(rng.randint(0, width - size + 1))
        crops.append(
            torch.from_numpy(np.array(labels[top : top + size, left : left + size]))
        )
    probs = phase_channels(torch.stack(crops).long(), config.data.num_phases)
    return resize_phases(probs, (patch_size,) * 2)


def read_metrics(log_dir: Path) -> list[dict]:
    """All TensorBoard scalars by step, without reservoir sampling."""
    events = EventAccumulator(str(log_dir), size_guidance={"scalars": 0}).Reload()
    rows = {}
    for tag in events.Tags()["scalars"]:
        for event in events.Scalars(tag):
            row = rows.setdefault(event.step, {"step": event.step})
            row[tag] = None if math.isnan(event.value) else event.value
    return [rows[step] for step in sorted(rows)]


def wall_minutes(log_dir: Path) -> float:
    """Minutes from the first to the last TensorBoard scalar of an unresumed run."""
    events = EventAccumulator(str(log_dir), size_guidance={"scalars": 0}).Reload()
    times = [
        event.wall_time
        for tag in events.Tags()["scalars"]
        for event in events.Scalars(tag)
    ]
    return (max(times) - min(times)) / 60
