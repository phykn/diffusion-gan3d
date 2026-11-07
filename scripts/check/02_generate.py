import argparse
import dataclasses
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.check.common import (
    add_generation_arguments,
    format_phases,
    generation_options,
    non_negative_int,
    positive_int,
    prepare_check,
    save_results,
    show,
    write_json,
)
from scripts.simul.src.anode import depth_profile
from src.api import LowResolutionAPI
from src.config.io import load_yaml
from src.console import field, metric, path_field, section, status
from src.stats.labels import phase_fractions
from src.stats.seam import measure_seams


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate an LR volume, tiled over blocks, and inspect its seams."
    )
    parser.add_argument(
        "--blocks",
        nargs=3,
        type=positive_int,
        default=(1, 1, 1),
        metavar=("D", "H", "W"),
        help="Output blocks of one LR patch along each axis (default: 1 1 1).",
    )
    parser.add_argument(
        "--overlap",
        type=non_negative_int,
        help="Voxels each block shares with its neighbor "
        "(default: config/gen.yaml, below half a patch).",
    )
    parser.add_argument(
        "--margin",
        type=non_negative_int,
        help="Voxels generated and discarded on every outer face "
        "(default: the model's).",
    )
    parser.add_argument(
        "--vf-profile",
        type=Path,
        help="YAML/JSON file with a z vf_profile (axis, points, interpolation), or "
        "an anode simulation config for its target depth fractions "
        "(default: the training depth profile of a profile-conditioned model).",
    )
    add_generation_arguments(parser)
    args = parser.parse_args(argv)
    section("LR generation check")
    settings = prepare_check(args, __file__)
    api = LowResolutionAPI(args.weight, args.device)
    patch = api.input_size
    # A small patch cannot take the default overlap; keep below half of it.
    if args.overlap is None:
        args.overlap = min(settings.overlap, (patch - 1) // 2)
    blocks = tuple(args.blocks)
    options = generation_options(args) | {
        "blocks": blocks,
        "overlap": args.overlap,
        "margin": args.margin,
        "seed": args.seed,
    }
    if args.vf_profile is not None:
        options["vf_profile"] = load_vf_profile(args.vf_profile)
    plan = api.plan(blocks=blocks, overlap=args.overlap, margin=args.margin)
    field(
        "Plan",
        f"{' × '.join(map(str, plan.grid))} tiles ({plan.tile_count} total), "
        f"overlap {plan.overlap}, margin {plan.margin}, {api.device}",
    )
    field("Output", " × ".join(map(str, plan.output_shape)))
    status("Generating")
    started = perf_counter()
    volume = api.generate(**options)
    elapsed = perf_counter() - started
    save_results(volume, args, api.num_phases)
    section("Quality")
    metric("Phase VF", format_phases(phase_fractions(volume, api.num_phases)))
    if plan.tile_count > 1:
        quality = measure_seams(
            volume, plan.output_seams(), plan.overlap, api.num_phases
        )
        metric("Seam TV", format_axes(quality.transition_tv))
        metric("Seam continuity", format_axes(quality.continuation_delta))
        report = dataclasses.asdict(quality) | {
            "seams": [list(axis) for axis in plan.output_seams()],
            "overlap": plan.overlap,
        }
        path_field("Seams", write_json(args.out / "seams.json", report))
    field("Elapsed", f"{elapsed:.1f} s")
    show(volume, args, api.num_phases, "EMA model")


def load_vf_profile(path: Path) -> dict:
    spec = load_yaml(path)
    if spec.get("geometry", {}).get("kind") == "anode":
        return depth_profile(spec["geometry"])
    return spec


def format_axes(values) -> str:
    return "  ".join(
        f"axis {axis}: {'n/a' if value is None else f'{value:.4f}'}"
        for axis, value in enumerate(values)
    )


if __name__ == "__main__":
    main()
