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
