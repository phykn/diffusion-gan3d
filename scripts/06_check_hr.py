import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.cli import (
    add_device_argument,
    add_domain_seed_arguments,
    add_height_arguments,
    non_negative_int,
    positive_int,
    result_directory,
)
from scripts.common.diagnostic import format_phases
from src.build.predict import resolve_weights
from src.config.data import validate_sr_source
from src.config.files import load_yaml
from src.console import field, metric, path_field, section, status
from src.device import check_device, empty_cache
from src.evaluate.label import phase_fractions
from src.plane import PLANES
from src.predict.inference import LowResolutionAPI
from src.predict.sr.inference import SuperResolutionAPI, load_sr_export
from src.prepare.resize import downsample, phase_channels
from src.storage import load_probabilities, load_volume, save_probabilities, save_volume
from src.train.run.source import frozen_source_weights


def latest_sr():
    candidates = []
    for directory in (PROJECT_ROOT / "run").glob("*/"):
        config, weight = directory / "train.yaml", directory / "generator.pt"
        if (
            config.is_file()
            and weight.is_file()
            and load_yaml(config).get("stage") == "sr"
        ):
            candidates.append(weight)
    if not candidates:
        raise FileNotFoundError("No trained SR export found in run/. Supply --weight.")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def comparison(low, high, phases, path, show):
    if not show:
        plt.switch_backend("Agg")
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspect SR using its saved LR source; only SR --weight is needed."
    )
    parser.add_argument(
        "--weight",
        type=Path,
        help="SR generator.pt or run directory (default: latest SR run in run/).",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--lr-weight", type=Path, help="Override the LR source stored with SR."
    )
    source.add_argument(
        "--input", type=Path, help="Existing LR label TIFF or fractional .pt."
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output directory (default: a new directory under checks/).",
    )
    add_device_argument(parser)
    add_domain_seed_arguments(parser)
    add_height_arguments(parser)
    parser.add_argument(
        "--lr-guidance",
        type=float,
        help="LR guidance scale (default: config/gen.yaml).",
    )
    parser.add_argument(
        "--guidance", type=float, default=1.0, help="SR guidance scale (default: 1)."
    )
    parser.add_argument(
        "--tile-size",
        type=positive_int,
        help="HR tile edge (default: the SR training size).",
    )
    parser.add_argument(
        "--overlap",
        type=non_negative_int,
        help="HR overlap (default: scale-aligned up to 8).",
    )
    parser.add_argument(
        "--margin",
        type=non_negative_int,
        help="HR context margin (default: model context).",
    )
    parser.add_argument(
        "--napari",
        action="store_true",
        help="Open LR and HR as spatially aligned 3D layers.",
    )
    parser.add_argument(
        "--no-view", action="store_true", help="Save results without opening a viewer."
    )
    args = parser.parse_args(argv)
    section("SR generation check")
    device = check_device(args.device)
    weight = resolve_weights(args.weight or latest_sr())
    cfg = load_sr_export(weight)["config"]
    lr_weight = None
    common = dict(
        domain=args.domain,
        seed=args.seed,
        height_origin=args.height_origin,
        height_extent=args.height_extent,
    )
    if args.input:
        low = (
            load_probabilities(args.input)
            if args.input.suffix.lower() == ".pt"
            else load_volume(args.input)
        )
    else:
        lr_weight = (
            resolve_weights(args.lr_weight)
            if args.lr_weight
            else frozen_source_weights(cfg.get("source", {}))
        )
        path_field("LR weights", lr_weight)
        status("Generating LR")
        lr = LowResolutionAPI(lr_weight, device)
        validate_sr_source(cfg["data"], lr.data)
        if args.lr_guidance is None:
            args.lr_guidance = lr.settings.guidance
        low = lr.generate_probs(guidance=args.lr_guidance, **common)
        del lr
        gc.collect()
        empty_cache(device)
    path_field("SR weights", weight)
    status(f"Refining LR {tuple(low.shape[-3:])}")
    sr = SuperResolutionAPI(weight, device)
    tile_size, overlap = sr.default_tiling(args.tile_size)
    if args.overlap is not None:
        overlap = args.overlap
    high = sr.refine(
        low,
        tile_size=tile_size,
        overlap=overlap,
        margin=args.margin,
        guidance=args.guidance,
        **common,
    )
    root = args.out or result_directory(__file__)
    root.mkdir(parents=True, exist_ok=True)
    low_labels = low.argmax(0) if low.ndim == 4 else low
    save_volume(low_labels, root / "lr.tiff")
    save_volume(high, root / "hr.tiff")
    if low.ndim == 4:
        save_probabilities(low, root / "lr_probs.pt")
    low_probs = low[None] if low.ndim == 4 else phase_channels(low[None], sr.num_phases)
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
        "lr_weights": str(lr_weight) if lr_weight else None,
        "input": str(args.input.resolve()) if args.input else None,
        "lr_source_verified": not args.input and not args.lr_weight,
        **common,
        "crop_size": sr.crop_size,
        "lo_res_size": sr.lo_res_size,
        "hi_res_size": sr.hi_res_size,
        "source_pixels_per_hr_voxel": sr.crop_size / sr.hi_res_size,
        "lr_shape": list(low.shape[-3:]),
        "hr_shape": list(high.shape),
        "tile_size": tile_size,
        "overlap": overlap,
        "margin": args.margin,
        "sr_guidance": args.guidance,
        "lr_guidance": args.lr_guidance,
        "lr_phase_fractions": low_probs.mean((0, 2, 3, 4)).tolist(),
        "hr_label_phase_fractions": phase_fractions(high, sr.num_phases).tolist(),
        "hr_label_coarse_mae": coarse_error / low_probs.numel(),
        "has_measured_3d_reference": False,
    }
    (root / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    path_field("Results", root.resolve())
    field("HR shape", tuple(high.shape))
    section("Quality")
    metric("LR phase VF", format_phases(report["lr_phase_fractions"]))
    metric("HR phase VF", format_phases(report["hr_label_phase_fractions"]))
    metric("Coarse MAE", f"{report['hr_label_coarse_mae']:.4f}")
    path_field("Comparison", root.resolve() / "comparison.png")
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
        viewer.add_labels(
            low_labels.cpu().numpy(),
            name="LR",
            scale=(sr.scale_factor,) * 3,
            translate=((sr.scale_factor - 1) / 2,) * 3,
        )
        viewer.add_labels(high.cpu().numpy(), name="Generated HR")
        viewer.dims.ndisplay = 3
        napari.run()


if __name__ == "__main__":
    main()
