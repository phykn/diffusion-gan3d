import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.cli import (
    add_device_argument,
    add_height_arguments,
    check_parser,
    height_options,
    non_negative_int,
    positive_int,
    result_directory,
)
from src.evaluate.tilt import TILT_PAIRS, tilt_balance
from src.phase import PORE_PHASE
from src.storage import load_volume


def main() -> None:
    parser = check_parser(
        __file__,
        "Compare the signed particle tilt balance of generated and real volumes.",
    )
    parser.add_argument("--weight", type=Path, help="LR generator.pt or run directory")
    parser.add_argument(
        "--volumes", type=Path, nargs="*", default=[], help="Generated TIFF volumes"
    )
    parser.add_argument(
        "--real", type=Path, nargs="*", default=[], help="Real TIFF volumes"
    )
    parser.add_argument(
        "--real-stride",
        type=positive_int,
        default=1,
        help="Subsample real volumes to the generated voxel scale (2 for 128 -> 64).",
    )
    parser.add_argument("--samples", type=positive_int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--guidance", type=float, help="Default: config/gen.yaml.")
    parser.add_argument(
        "--phases",
        type=int,
        nargs="*",
        help="Default: every phase except the pore phase 0.",
    )
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--min-voxels", type=non_negative_int, default=8)
    parser.add_argument("--out", type=Path, help="report.json path")
    add_device_argument(parser)
    add_height_arguments(parser)
    args = parser.parse_args()
    if args.weight is None and not args.volumes:
        parser.error("give --weight or --volumes.")

    generated = [load_labels(path) for path in args.volumes]
    if args.weight is not None:
        generated += generate(args)
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

    out = args.out or result_directory(Path(__file__).stem) / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport  : {out.resolve()}")


def load_labels(path: Path) -> np.ndarray:
    return np.asarray(load_volume(path))


def generate(args: argparse.Namespace) -> list[np.ndarray]:
    from src.build.predict import load_generator, resolve_weights
    from src.config.generation import load_generation_settings
    from src.device import check_device
    from src.random import seeded_rng

    device = check_device(args.device)
    guidance = (
        load_generation_settings().guidance if args.guidance is None else args.guidance
    )
    generator = load_generator(resolve_weights(args.weight), device=device)
    volumes = []
    for index in range(args.samples):
        with seeded_rng(args.seed + index, device):
            vol = generator.generate(
                vf=None,
                guidance=guidance,
                domain=args.domain,
                **height_options(args),
                margin=generator.default_margin,
            )
        volumes.append(vol.detach().cpu().numpy())
        print(f"Generated sample {index + 1}/{args.samples}", flush=True)
    return volumes


def print_phase(phase: int, entry: dict) -> None:
    print(f"\nPhase {phase}  (+/- = tilted fraction, balance = + minus -)")
    for name in TILT_PAIRS:
        for source, stats in entry.items():
            s = stats[name]
            print(
                f"  d{name[0]}/d{name[1]} {source:9s} n={s['count']:4d}  "
                f"+{s['positive']:5.0%} -{s['negative']:5.0%}  "
                f"balance {s['balance']:+.2f}  mean {s['mean']:+.3f}±{s['sem']:.3f}"
            )


if __name__ == "__main__":
    main()
