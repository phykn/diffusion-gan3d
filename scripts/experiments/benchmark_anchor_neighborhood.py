"""Compare loop and batched anchor-neighborhood loss, including backward.

Run from the repository root with .venv Python. The fixed loop reference is also
used by regression tests; this measures the loss, not an entire training step.
"""

import argparse
import hashlib
import json
import platform
import runpy
import statistics
import sys
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.anchor import PlaneAnchor, encode_anchors
from src.train.loss.anchor_neighborhood import compute_anchor_neighborhood_loss


def make_case(size, axes, device):
    rng = torch.Generator().manual_seed(731)
    plane = torch.rand(1, 3, 1, size, size, generator=rng).softmax(1)
    volume = plane.expand(-1, -1, size, -1, -1).clone()
    center = size // 2
    volume[:, :, center - 1 : center + 2] = volume[:, :, center - 1 : center + 2].roll(
        1, 1
    )
    volume = volume.to(device).requires_grad_()
    condition = encode_anchors(
        [
            PlaneAnchor(volume.detach().select(axis + 2, center), axis, center)
            for axis in axes
        ],
        1,
        3,
        size,
        device,
        torch.float32,
    )
    return volume, condition


def measure(functions, volume, condition, radius, backward, warmup, repeats):
    def call(fn):
        volume.grad = None
        loss, _ = fn(volume, condition, radius)
        if backward:
            loss.backward()

    def sync():
        if volume.is_cuda:
            torch.cuda.synchronize(volume.device)

    for _ in range(warmup):
        for fn in functions.values():
            call(fn)
    sync()
    samples = {name: [] for name in functions}
    for iteration in range(repeats):
        # Reverse each round to reduce bias from clock and background load drift.
        names = list(functions)
        if iteration % 2:
            names.reverse()
        for name in names:
            sync()
            start = perf_counter()
            call(functions[name])
            sync()
            samples[name].append((perf_counter() - start) * 1000)
    return {
        name: {
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
            "samples_ms": values,
        }
        for name, values in samples.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.threads, args.warmup, args.repeats) < 1:
        parser.error("threads, warmup and repeats must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    path = ROOT / "tests/train/anchor_neighborhood_reference.py"
    reference = runpy.run_path(str(path))["compute_anchor_neighborhood_loss"]
    functions = {"loop": reference, "batched": compute_anchor_neighborhood_loss}
    report = {
        "scope": "isolated loss; forward and forward+backward; no model optimizer step",
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "device": str(device),
        "threads": args.threads,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "reference_commit": "0dce898",
        "reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cases": [],
    }
    for size, radius, axes in ((32, 2, (0,)), (32, 4, (0, 1, 2)), (64, 2, (0, 1, 2))):
        volume, condition = make_case(size, axes, device)
        expected, expected_logs = reference(volume, condition, radius)
        expected_grad = torch.autograd.grad(expected, volume)[0]
        actual, actual_logs = compute_anchor_neighborhood_loss(
            volume, condition, radius
        )
        actual_grad = torch.autograd.grad(actual, volume)[0]
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-8, rtol=2e-5)
        for key, value in expected_logs.items():
            torch.testing.assert_close(actual_logs[key], value, atol=2e-7, rtol=2e-5)
        del expected, actual, expected_grad, actual_grad, expected_logs, actual_logs
        for backward in (False, True):
            timing = measure(
                functions,
                volume,
                condition,
                radius,
                backward,
                args.warmup,
                args.repeats,
            )
            result = {
                "shape": list(volume.shape),
                "radius": radius,
                "axes": axes,
                "backward": backward,
                "timing": timing,
                "speedup": timing["loop"]["median_ms"] / timing["batched"]["median_ms"],
            }
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
