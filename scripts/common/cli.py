import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from src.config.files import PROJECT_ROOT


def check_parser(script, description):
    return argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=f'Example: python scripts/{Path(script).name} --weight "run/my-run"',
    )


def resolve_weight(path):
    path = Path(path).expanduser().resolve()
    return path / "generator.pt" if path.is_dir() else path


def add_height_arguments(parser, origin_default=0.0):
    parser.add_argument(
        "--height-origin",
        type=float,
        default=origin_default,
        help="Output Z origin in source pixels; real xz/yz crops infer it when omitted.",
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
        / "run"
        / "checks"
        / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + name)
    )


def prepare_check(args, script):
    if args.no_view:
        plt.switch_backend("Agg")
    args.weight = resolve_weight(args.weight)
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
