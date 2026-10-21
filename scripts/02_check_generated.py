import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.cli import (
    add_generation_arguments,
    height_options,
    prepare_check,
    save_results,
)
from scripts.common.diagnostic import show_napari, show_slices
from simul.src.anode import depth_profile
from src.build.predict import load_generator
from src.config.files import load_yaml
from src.device import check_device
from src.random import seeded_rng


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect an LR model with three orthogonal slices."
    )
    add_generation_arguments(parser)
    parser.add_argument(
        "--vf-profile",
        type=Path,
        help="YAML/JSON file with a z vf_profile (axis, points, interpolation), or "
        "an anode simulation config for its target depth fractions "
        "(default: the training depth profile of a profile-conditioned model).",
    )
    args = parser.parse_args()
    prepare_check(args, __file__)

    device = check_device(args.device)
    print(f"\nWeights : {args.weight}", flush=True)

    generator = load_generator(args.weight, device=device)
    shape = (generator.patch_size,) * 3
    print(
        f"Generate: {' × '.join(map(str, shape))}, domain {args.domain}, {device}",
        flush=True,
    )

    with seeded_rng(args.seed, device):
        vol = generator.generate(
            vf=None,
            guidance=args.guidance,
            domain=args.domain,
            **height_options(args),
            **(
                {}
                if args.vf_profile is None
                else {"vf_profile": load_vf_profile(args.vf_profile)}
            ),
            margin=generator.default_margin,
        )
    save_results(vol, args, generator.num_phases)
    if args.no_view:
        return
    if args.napari:
        show_napari(vol)
    else:
        show_slices(vol, generator.num_phases, "EMA model")


def load_vf_profile(path: Path) -> dict:
    spec = load_yaml(path)
    if spec.get("geometry", {}).get("kind") == "anode":
        return depth_profile(spec["geometry"])
    return spec


if __name__ == "__main__":
    main()
