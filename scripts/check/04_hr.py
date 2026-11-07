import argparse
import gc
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.check.common import (
    add_device_argument,
    add_height_arguments,
    add_out_argument,
    add_seed_argument,
    format_phases,
    non_negative_int,
    positive_int,
    prepare_output,
    write_json,
)
from src.api import SuperResolutionAPI, lr_fractions
from src.config.io import PROJECT_ROOT, load_generation_settings, load_yaml
from src.console import field, metric, path_field, section, status
from src.data.crops import downsample
from src.device import empty_cache
from src.export import resolve_export
from src.plane import PLANES
from src.stats.labels import phase_fractions
from src.storage import load_probabilities, load_volume, save_probabilities, save_volume


def latest_sr() -> Path:
    candidates = [
        directory / "generator.pt"
        for directory in (PROJECT_ROOT / "run").glob("*-sr/")
        if (directory / "generator.pt").is_file()
        and (directory / "train.yaml").is_file()
        and load_yaml(directory / "train.yaml").get("stage") == "sr"
    ]
    if not candidates:
        raise FileNotFoundError("No trained SR export found in run/. Supply --weight.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def comparison(low, high, phases, path, show):
    fig, panels = plt.subplots(3, 3, figsize=(10, 10))
    for row, (title, volume) in enumerate(
        (("LR", low), ("Nearest baseline", high), ("Generated HR", high))
    ):
        for axis, plane in enumerate(PLANES):
            index = volume.shape[axis] // 2
            if row == 1:
                # The three displayed planes match 3D nearest interpolation
                # without allocating a second complete HR volume.
                source_index = index * low.shape[axis] // high.shape[axis]
                image = low.select(axis, source_index)
                size = tuple(n for a, n in enumerate(high.shape) if a != axis)
                image = F.interpolate(
                    image[None, None].float(), size=size, mode="nearest"
                )[0, 0]
            else:
                image = volume.select(axis, index)
            panels[row, axis].imshow(
                image.numpy(),
                cmap="gray",
                vmin=0,
                vmax=max(1, phases - 1),
                interpolation="nearest",
            )
            panels[row, axis].set_title(f"{title} / {plane} / index {index}")
            panels[row, axis].axis("off")
    fig.suptitle("Same field of view — generated HR has no measured 3D reference")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Refine an LR volume with SR; the SR export carries its LR model."
    )
    parser.add_argument(
        "--weight",
        type=Path,
        help="SR generator.pt or run folder (default: the newest SR run in run/).",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--lr-weight", type=Path, help="LR export to use instead of the stored one."
    )
    source.add_argument(
        "--input", type=Path, help="Existing LR label TIFF or fractional .pt."
    )
    add_out_argument(parser)
    add_device_argument(parser)
    add_seed_argument(parser)
    add_height_arguments(parser)
    parser.add_argument(
        "--lr-guidance",
        type=float,
        help="LR guidance scale (default: config/gen.yaml).",
    )
    parser.add_argument(
        "--tile-size",
        type=positive_int,
        help="HR tile edge (default: SR training size).",
    )
    parser.add_argument(
        "--overlap",
        type=non_negative_int,
        help="HR overlap (default: scale-aligned, at most 8).",
    )
    parser.add_argument(
        "--margin",
        type=non_negative_int,
        help="HR context margin (default: lcm(8, scale)).",
    )
    parser.add_argument(
        "--napari", action="store_true", help="Open LR and HR as aligned 3D layers."
    )
    parser.add_argument(
        "--no-view", action="store_true", help="Save results without a viewer."
    )
    args = parser.parse_args(argv)
    section("SR generation check")
    root = prepare_output(args, __file__)
    weight = resolve_export(args.weight or latest_sr()).resolve()
    path_field("SR weights", weight)
    sr = SuperResolutionAPI(weight, args.device, lr_weights=args.lr_weight)
    shared = {
        "seed": args.seed,
        "height_origin": args.height_origin,
        "height_extent": args.height_extent,
    }
    lr_weight = None
    if args.input:
        low = (
            load_probabilities(args.input)
            if args.input.suffix.lower() == ".pt"
            else load_volume(args.input)
        )
    else:
        # None: the LR export stored inside the SR export.
        if args.lr_weight:
            lr_weight = str(resolve_export(args.lr_weight).resolve())
        if args.lr_guidance is None:
            args.lr_guidance = load_generation_settings().guidance
        status("Generating LR")
        low = sr.lr.generate_probs(guidance=args.lr_guidance, **shared)
        gc.collect()
        empty_cache(sr.device)
    tile_size, overlap = sr.default_tiling(args.tile_size)
    if args.overlap is not None:
        overlap = args.overlap
    status(f"Refining LR {tuple(low.shape[-3:])}")
    high = sr.refine(
        low, tile_size=tile_size, overlap=overlap, margin=args.margin, **shared
    )
    low_labels = low.argmax(0) if low.ndim == 4 else low
    save_volume(low_labels, root / "lr.tiff")
    save_volume(high, root / "hr.tiff")
    if low.ndim == 4:
        save_probabilities(low, root / "lr_probs.pt")
    low_probs = lr_fractions(low, sr.num_phases)[None]
    # This compares categorical output occupancy, not an unknown HR ground truth.
    # One phase at a time avoids a full-size HR one-hot volume.
    coarse_error = sum(
        (downsample((high == p)[None, None].float(), low.shape[-3:]) - low_probs[:, p])
        .abs()
        .sum()
        .item()
        for p in range(sr.num_phases)
    )
    report = {
        "sr_weights": str(weight),
        "lr_weights": lr_weight,
        "input": str(args.input.resolve()) if args.input else None,
        "lr_source_verified": not args.input and not args.lr_weight,
        **shared,
        "crop_size": sr.crop_size,
        "lo_res_size": sr.lo_res_size,
        "hi_res_size": sr.hi_res_size,
        "source_pixels_per_hr_voxel": sr.crop_size / sr.hi_res_size,
        "lr_shape": list(low.shape[-3:]),
        "hr_shape": list(high.shape),
        "tile_size": tile_size,
        "overlap": overlap,
        "margin": sr.default_margin if args.margin is None else args.margin,
        "lr_guidance": args.lr_guidance,
        "lr_phase_fractions": low_probs.mean((0, 2, 3, 4)).tolist(),
        "hr_label_phase_fractions": phase_fractions(high, sr.num_phases).tolist(),
        "hr_label_coarse_mae": coarse_error / low_probs.numel(),
        "has_measured_3d_reference": False,
    }
    write_json(root / "report.json", report)
    field("HR shape", tuple(high.shape))
    section("Quality")
    metric("LR phase VF", format_phases(report["lr_phase_fractions"]))
    metric("HR phase VF", format_phases(report["hr_label_phase_fractions"]))
    metric("Coarse MAE", f"{report['hr_label_coarse_mae']:.4f}")
    path_field("Comparison", root / "comparison.png")
    comparison(
        low_labels.cpu(),
        high.cpu(),
        sr.num_phases,
        root / "comparison.png",
        show=not args.no_view and not args.napari,
    )
    if args.napari and not args.no_view:
        import napari

        viewer = napari.Viewer()
        scale = sr.scale_factor
        viewer.add_labels(
            low_labels.cpu().numpy(),
            name="LR",
            scale=(scale,) * 3,
            translate=((scale - 1) / 2,) * 3,
        )
        viewer.add_labels(high.cpu().numpy(), name="Generated HR")
        viewer.dims.ndisplay = 3
        napari.run()


if __name__ == "__main__":
    main()
