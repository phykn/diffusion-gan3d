import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.cli import (
    add_device_argument,
    add_domain_seed_arguments,
    add_height_arguments,
    generation_options,
    non_negative_int,
    positive_int,
    result_directory,
)
from src.build.predict import load_generator
from src.config.generation import load_generation_settings
from src.console import emit, paint, path_field, section, status
from src.device import check_device
from src.evaluate.tilt import TILT_PAIRS, tilt_balance
from src.phase import PORE_PHASE
from src.random import seeded_rng
from src.storage import load_volume


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the signed particle tilt balance of generated and real "
        "volumes."
    )
    parser.add_argument(
        "--weight", type=Path, help="LR generator.pt or run directory to sample."
    )
    parser.add_argument(
        "--volumes", type=Path, nargs="*", default=[], help="Generated TIFF volumes."
    )
    parser.add_argument(
        "--real", type=Path, nargs="*", default=[], help="Real TIFF volumes."
    )
    parser.add_argument(
        "--real-stride",
        type=positive_int,
        default=1,
        help="Subsample real volumes to the generated voxel scale, e.g. 2 for "
        "128 -> 64 (default: 1).",
    )
    parser.add_argument(
        "--samples",
        type=positive_int,
        default=6,
        help="Volumes to generate with --weight (default: 6).",
    )
    add_domain_seed_arguments(parser)
    parser.add_argument(
        "--guidance", type=float, help="Guidance scale (default: config/gen.yaml)."
    )
    parser.add_argument(
        "--phases",
        type=int,
        nargs="*",
        help="Phases to measure (default: every phase except the pore phase 0).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Smallest |slope| counted as tilted (default: 0.05).",
    )
    parser.add_argument(
        "--min-voxels",
        type=non_negative_int,
        default=8,
        help="Smallest particle measured (default: 8).",
    )
    parser.add_argument(
        "--out", type=Path, help="report.json path (default: under checks/)."
    )
    add_device_argument(parser)
    add_height_arguments(parser)
    args = parser.parse_args()
    section("Particle tilt check")
    if args.weight is None and not args.volumes:
        parser.error("give --weight or --volumes.")

    generated = [load_labels(path) for path in args.volumes]
    if args.weight is not None:
        generated += generate_volumes(args)
    real = [
        load_labels(path)[:: args.real_stride, :: args.real_stride, :: args.real_stride]
        for path in args.real
    ]

    phases = args.phases or sorted(
        {int(p) for volume in generated + real for p in np.unique(volume)}
        - {PORE_PHASE}
    )
    report = {"threshold": args.threshold, "phases": {}}
    for phase in phases:
        entry = {
            "generated": tilt_balance(generated, phase, args.threshold, args.min_voxels)
        }
        if real:
            entry["real"] = tilt_balance(real, phase, args.threshold, args.min_voxels)
        report["phases"][phase] = entry
        print_phase(phase, entry)

    out = args.out or result_directory(__file__) / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    emit()
    path_field("Report", out.resolve())


def load_labels(path: Path) -> np.ndarray:
    return np.asarray(load_volume(path))


def generate_volumes(args: argparse.Namespace) -> list[np.ndarray]:
    device = check_device(args.device)
    if args.guidance is None:
        args.guidance = load_generation_settings().guidance
    generator = load_generator(args.weight, device=device)
    options = generation_options(args, generator.default_margin)
    volumes = []
    status(f"Generating {args.samples} samples")
    for index in range(args.samples):
        with seeded_rng(args.seed + index, device):
            vol = generator.generate(**options)
        volumes.append(vol.numpy())
        emit(paint(f"  sample {index + 1}/{args.samples} done", "dim"))
    return volumes


def print_phase(phase: int, entry: dict) -> None:
    section(f"Phase {phase}")
    emit(paint("  +/- = tilted fraction, balance = + minus -", "dim"))
    for name in TILT_PAIRS:
        for source, stats in entry.items():
            s = stats[name]
            emit(
                f"  d{name[0]}/d{name[1]} {source:9s} n={s['count']:4d}  "
                f"+{s['positive']:5.0%} -{s['negative']:5.0%}  "
                f"balance {s['balance']:+.2f}  mean {s['mean']:+.3f}±{s['sem']:.3f}"
            )


if __name__ == "__main__":
    main()
