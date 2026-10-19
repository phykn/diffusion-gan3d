import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from src.build.predict import resolve_weights
from src.config.files import PROJECT_ROOT
from src.device import default_device


def check_parser(script, description):
    return argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=f'Example: python scripts/{Path(script).name} --weight "run/my-run"',
    )


def parse_unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between zero and one")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def add_device_argument(parser):
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default=default_device(),
        help="Compute device; CUDA, then Apple MPS, then CPU when available.",
    )


def add_generation_arguments(parser, *, anchors: bool = False, origin_default=0.0):
    """Options shared by the LR checks 02-05; unset guidance uses config/gen.yaml."""
    parser.add_argument(
        "--weight",
        type=Path,
        required=True,
        help="LR generator.pt or its run directory",
    )
    parser.add_argument("--domain", type=int, default=0, help="numeric domain ID")
    parser.add_argument("--seed", type=int, default=0, help="reproducible seed")
    parser.add_argument(
        "--guidance",
        type=float,
        help="classifier-free guidance scale (default: config/gen.yaml)",
    )
    if anchors:
        parser.add_argument(
            "--anchor-strength",
            type=parse_unit_interval,
            help="normalized anchor prediction strength (default: config/gen.yaml)",
        )
    parser.add_argument(
        "--out", type=Path, help="TIFF path; default: a new directory under checks/"
    )
    parser.add_argument(
        "--napari", action="store_true", help="show the 3D phase volume in Napari"
    )
    parser.add_argument(
        "--no-view", action="store_true", help="save results without a viewer"
    )
    add_device_argument(parser)
    add_height_arguments(parser, origin_default)


def add_height_arguments(parser, origin_default=0.0):
    text = "Output Z origin in source pixels."
    if origin_default is None:
        text += " Real xz/yz crops infer it when omitted."
    parser.add_argument(
        "--height-origin", type=float, default=origin_default, help=text
    )
    parser.add_argument(
        "--height-extent", type=float, help="Full source Z extent in pixels."
    )


def height_options(args):
    origin, extent = args.height_origin, args.height_extent
    if origin in (None, 0) and extent is None:
        return {}
    return {"height_origin": 0.0 if origin is None else origin, "height_extent": extent}


def result_directory(name):
    return (
        PROJECT_ROOT
        / "checks"
        / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + name)
    )


def prepare_check(args, script):
    if args.no_view:
        plt.switch_backend("Agg")
    args.weight = resolve_weights(args.weight)
    if args.out is None:
        args.out = result_directory(Path(script).stem) / "volume.tiff"
    args.out = args.out.expanduser().resolve()
    print(f"Results : {args.out.parent}", flush=True)
    return args


def save_preview(volume, args, num_phases):
    volume = volume.detach().cpu()
    middle = tuple(size // 2 for size in volume.shape)
    images = (volume[middle[0]], volume[:, middle[1]], volume[:, :, middle[2]])
    fig, panels = plt.subplots(1, 3, figsize=(10, 4))
    for panel, plane, image in zip(panels, ("xy", "xz", "yz"), images):
        panel.imshow(
            image.numpy(),
            cmap="gray",
            vmin=0,
            vmax=num_phases - 1,
            interpolation="nearest",
        )
        panel.set_title(plane)
        panel.axis("off")
    fig.tight_layout()
    path = args.out.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    metadata = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    metadata["shape"] = list(volume.shape)
    args.out.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Preview : {path}", flush=True)
