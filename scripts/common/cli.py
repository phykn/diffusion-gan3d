import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from scripts.common.diagnostic import slice_figure
from src.build.predict import resolve_weights
from src.config.files import PROJECT_ROOT
from src.config.generation import load_generation_settings
from src.console import path_field
from src.device import default_device
from src.storage import save_volume


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed < float("inf"):
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
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


def add_domain_seed_arguments(parser):
    parser.add_argument(
        "--domain", type=int, default=0, help="Numeric domain ID (default: 0)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Reproducible seed (default: 0)."
    )


def add_generation_arguments(parser, *, anchors: bool = False, origin_default=0.0):
    """Options shared by the LR checks 02-05; unset guidance uses config/gen.yaml."""
    parser.add_argument(
        "--weight",
        type=Path,
        required=True,
        help="LR generator.pt or its run directory.",
    )
    add_domain_seed_arguments(parser)
    parser.add_argument(
        "--guidance",
        type=float,
        help="Classifier-free guidance scale (default: config/gen.yaml).",
    )
    if anchors:
        parser.add_argument(
            "--anchor-strength",
            type=non_negative_float,
            help="Anchor guidance step scale; 0 ignores anchors (default: config/gen.yaml).",
        )
    parser.add_argument(
        "--out", type=Path, help="TIFF path (default: a new directory under checks/)."
    )
    parser.add_argument(
        "--napari", action="store_true", help="Show the 3D phase volume in Napari."
    )
    parser.add_argument(
        "--no-view", action="store_true", help="Save results without a viewer."
    )
    add_device_argument(parser)
    add_height_arguments(parser, origin_default)


def add_height_arguments(parser, origin_default=0.0):
    text = "Output Z origin in source pixels"
    if origin_default is None:
        text += "; real xz/yz crops infer it when omitted."
    else:
        text += f" (default: {origin_default:g})."
    parser.add_argument(
        "--height-origin", type=float, default=origin_default, help=text
    )
    parser.add_argument(
        "--height-extent", type=float, help="Full source Z extent in pixels."
    )


def generation_options(args, margin):
    """Generator options from the shared check arguments."""
    options = {"guidance": args.guidance, "domain": args.domain, "margin": margin}
    origin, extent = args.height_origin, args.height_extent
    if origin not in (None, 0) or extent is not None:
        options["height_origin"] = 0.0 if origin is None else origin
        options["height_extent"] = extent
    return options


def result_directory(script):
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + Path(script).stem
    return PROJECT_ROOT / "checks" / name


def prepare_check(args, script):
    """Resolve paths and unset generation options; return config/gen.yaml settings."""
    if args.no_view:
        plt.switch_backend("Agg")
    args.weight = resolve_weights(args.weight)
    if args.out is None:
        args.out = result_directory(script) / "volume.tiff"
    args.out = args.out.expanduser().resolve()
    path_field("Results", args.out.parent)
    settings = load_generation_settings()
    if args.guidance is None:
        args.guidance = settings.guidance
    if hasattr(args, "anchor_strength") and args.anchor_strength is None:
        args.anchor_strength = settings.anchor_strength
    return settings


def save_results(volume, args, num_phases):
    """Save the TIFF, a middle-slice PNG and the options as JSON next to it."""
    save_volume(volume, args.out)
    path_field("Volume", args.out)
    fig = slice_figure(volume, num_phases)
    path = args.out.with_suffix(".png")
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
    path_field("Preview", path)
