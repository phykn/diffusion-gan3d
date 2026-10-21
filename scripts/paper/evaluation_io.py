import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter

import numpy as np
import tifffile
import torch


def load_binary_volume(path: Path) -> np.ndarray:
    volume = np.asarray(tifffile.imread(path))
    if volume.ndim != 3 or volume.dtype != np.uint8 or volume.size == 0:
        raise ValueError(f"volume must be a non-empty 3D uint8 array: {path}")
    if int(volume.max()) > 1:
        raise ValueError(f"volume contains a phase outside 0 and 1: {path}")
    return volume


def write_csv(
    rows: Sequence[Mapping[str, object]],
    output: Path,
) -> None:
    if not rows:
        raise ValueError("CSV rows must not be empty.")
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def start_generation_timer(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return perf_counter()


def stop_generation_timer(device: torch.device, start: float) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return perf_counter() - start


def set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def summarize(
    rows: Sequence[Mapping[str, object]],
    conditions: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, object]]:
    """Per-condition mean and sample standard deviation, skipping missing values."""
    result = []
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        summary: dict[str, object] = {
            "condition": condition,
            "samples": len(selected),
        }
        for metric in metrics:
            values = np.asarray(
                [row[metric] for row in selected if row[metric] is not None],
                dtype=float,
            )
            summary[f"{metric}_mean"] = float(values.mean()) if values.size else None
            summary[f"{metric}_std"] = (
                float(values.std(ddof=1)) if values.size > 1 else None
            )
        result.append(summary)
    return result
