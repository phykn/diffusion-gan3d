import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.io import PROJECT_ROOT, load_generation_settings  # noqa: E402
from src.console import path_field  # noqa: E402
from src.device import default_device  # noqa: E402
from src.export import resolve_export  # noqa: E402
from src.plane import PLANES  # noqa: E402
from src.storage import save_volume  # noqa: E402

DIFF_CMAP = ListedColormap(("#f7f7f7", "#e63946"))


# Flags -------------------------------------------------------------------


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


def add_device_argument(parser) -> None:
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default=default_device(),
        help="Compute device; CUDA, then Apple MPS, then CPU when available.",
    )


def add_seed_argument(parser) -> None:
    parser.add_argument(
        "--seed", type=int, default=0, help="Reproducible seed (default: 0)."
    )


def add_out_argument(parser) -> None:
    parser.add_argument(
        "--out",
        type=Path,
        help="Results folder (default: a new folder under checks/).",
    )


def add_view_arguments(parser) -> None:
    parser.add_argument(
        "--napari", action="store_true", help="Show the 3D phase volume in Napari."
    )
    parser.add_argument(
        "--no-view", action="store_true", help="Save results without a viewer."
    )


def add_height_arguments(parser, origin_default: float | None = 0.0) -> None:
    text = "Output z origin in source pixels"
    if origin_default is None:
        text += "; real xz/yz crops infer it when omitted."
    else:
        text += f" (default: {origin_default:g})."
    parser.add_argument(
        "--height-origin", type=float, default=origin_default, help=text
    )
    parser.add_argument(
        "--height-extent", type=float, help="Full source z extent in pixels."
    )


def add_generation_arguments(
    parser, *, anchors: bool = False, origin_default: float | None = 0.0
) -> None:
    """Flags shared by the LR checks; unset guidance comes from config/gen.yaml."""
    parser.add_argument(
        "--weight",
        type=Path,
        required=True,
        help="LR generator.pt or its run folder.",
    )
    add_seed_argument(parser)
    parser.add_argument(
        "--guidance",
        type=float,
        help="Classifier-free guidance scale (default: config/gen.yaml).",
    )
    if anchors:
        parser.add_argument(
            "--anchor-strength",
            type=non_negative_float,
            help="Anchor guide step scale; 0 ignores anchors "
            "(default: config/gen.yaml).",
        )
    add_out_argument(parser)
    add_view_arguments(parser)
    add_device_argument(parser)
    add_height_arguments(parser, origin_default)


def generation_options(args) -> dict:
    """generate() options from the shared flags."""
    options = {"guidance": args.guidance}
    origin, extent = args.height_origin, args.height_extent
    if origin not in (None, 0) or extent is not None:
        options["height_origin"] = 0.0 if origin is None else origin
        options["height_extent"] = extent
    return options


# Results -----------------------------------------------------------------


def result_directory(script: str) -> Path:
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + Path(script).stem
    return PROJECT_ROOT / "checks" / name


def prepare_output(args, script: str) -> Path:
    """The results folder: --out, or a new folder under checks/."""
    if getattr(args, "no_view", False):
        plt.switch_backend("Agg")
    out = result_directory(script) if args.out is None else args.out
    args.out = out.expanduser().resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    path_field("Results", args.out)
    return args.out


def prepare_check(args, script: str):
    """Results folder, the export path and the unset generation options;
    returns the config/gen.yaml settings."""
    prepare_output(args, script)
    args.weight = resolve_export(args.weight).resolve()
    path_field("Weights", args.weight)
    settings = load_generation_settings()
    if args.guidance is None:
        args.guidance = settings.guidance
    if hasattr(args, "anchor_strength") and args.anchor_strength is None:
        args.anchor_strength = settings.anchor_strength
    return settings


def write_json(path: Path, data) -> Path:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def save_results(volume: torch.Tensor, args, num_phases: int, name="volume") -> None:
    """name.tiff, a middle-slice name.png and the options as name.json."""
    path = args.out / f"{name}.tiff"
    save_volume(volume, path)
    path_field("Volume", path)
    fig = slice_figure(volume, num_phases)
    fig.savefig(path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    metadata = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    metadata["shape"] = list(volume.shape)
    write_json(path.with_suffix(".json"), metadata)
    path_field("Preview", path.with_suffix(".png"))


def format_phases(values) -> str:
    return "  ".join(
        f"phase {phase}: {float(value):.2%}" for phase, value in enumerate(values)
    )


# Figures -----------------------------------------------------------------


def middle_slices(volume: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Center xy, xz and yz sections."""
    middle = tuple(size // 2 for size in volume.shape)
    return volume[middle[0]], volume[:, middle[1]], volume[:, :, middle[2]]


def show_labels(panel, image: torch.Tensor, num_phases: int, title: str) -> None:
    panel.imshow(
        image.numpy(),
        cmap="gray",
        vmin=-0.5,
        vmax=num_phases - 0.5,
        interpolation="nearest",
    )
    panel.set_title(title)
    panel.axis("off")


def show_difference(panel, mask: torch.Tensor, title: str, **options) -> None:
    panel.imshow(
        mask.numpy(), cmap=DIFF_CMAP, vmin=0, vmax=1, interpolation="nearest", **options
    )
    panel.set_title(title)
    panel.axis("off")


def slice_figure(volume: torch.Tensor, num_phases: int):
    fig, panels = plt.subplots(1, 3, figsize=(10, 4))
    for panel, plane, image in zip(panels, PLANES, middle_slices(volume), strict=True):
        show_labels(panel, image, num_phases, plane)
    fig.tight_layout()
    return fig


def show_slices(volume: torch.Tensor, num_phases: int, title: str) -> None:
    fig = slice_figure(volume, num_phases)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def show_napari(volume: torch.Tensor, name: str = "Generated output") -> None:
    import napari

    viewer = napari.Viewer()
    viewer.add_labels(volume.numpy(), name=name)
    viewer.dims.ndisplay = 3
    napari.run()


def show(volume: torch.Tensor, args, num_phases: int, title: str) -> None:
    """The viewer the flags ask for; nothing with --no-view."""
    if args.no_view:
        return
    if args.napari:
        show_napari(volume)
    else:
        show_slices(volume, num_phases, title)
