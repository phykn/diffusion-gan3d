import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.cli import (
    add_generation_arguments,
    check_parser,
    height_options,
    prepare_check,
    save_preview,
)
from scripts.common.diagnostic import show_napari
from src.build.predict import load_generator
from src.config.files import load_yaml
from src.config.generation import load_generation_settings
from src.device import check_device
from src.plane import PLANES
from src.random import seeded_rng
from src.storage import save_volume


def main() -> None:
    parser = check_parser(__file__, "Inspect an LR model with three orthogonal slices.")
    add_generation_arguments(parser)
    parser.add_argument(
        "--vf-profile",
        type=Path,
        help="YAML/JSON file with a z vf_profile (axis, points, interpolation); "
        "default: height conditioning only.",
    )
    args = parser.parse_args()
    args = prepare_check(args, __file__)
    if args.guidance is None:
        args.guidance = load_generation_settings().guidance

    device = check_device(args.device)
    weight = args.weight
    print(f"\nWeights : {weight.resolve()}", flush=True)

    generator = load_generator(weight, device=device)
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
                else {"vf_profile": load_yaml(args.vf_profile)}
            ),
            margin=generator.default_margin,
        )
    save_volume(vol, args.out)
    print(f"Saved   : {args.out.resolve()}", flush=True)
    save_preview(vol, args, generator.num_phases)
    if args.no_view:
        return
    if args.napari:
        show_napari(vol)
    else:
        show_slices(vol, generator.num_phases)


def show_slices(
    vol: torch.Tensor,
    num_phases: int,
) -> None:
    mid = tuple(size // 2 for size in vol.shape)
    slices = (
        vol[mid[0], :, :],
        vol[:, mid[1], :],
        vol[:, :, mid[2]],
    )

    fig, panels = plt.subplots(1, 3, figsize=(10, 4))
    for axis, img in enumerate(slices):
        panels[axis].imshow(
            img,
            cmap="gray",
            vmin=-0.5,
            vmax=num_phases - 0.5,
            interpolation="nearest",
        )
        panels[axis].set_title(PLANES[axis])
        panels[axis].axis("off")
    fig.suptitle("EMA model")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
