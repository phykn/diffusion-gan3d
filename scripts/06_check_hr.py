import gc
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.cli import check_parser, resolve_weight, result_directory
from src.config.data import validate_sr_source
from src.config.files import load_yaml
from src.config.train import normalize_train_config
from src.predict.inference import InferenceAPI
from src.predict.sr.inference import SuperResolutionAPI
from src.prepare.resize import downsample, phase_channels
from src.storage import load_probabilities, load_volume, save_probabilities, save_volume


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
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Match the field of view; the baseline adds no generated structure.
    nearest = F.interpolate(low[None, None].float(), size=high.shape, mode="nearest")[
        0, 0
    ].long()
    fig, panels = plt.subplots(3, 3, figsize=(10, 10))
    for row, (title, volume) in enumerate(
        (("LR", low), ("Nearest baseline", nearest), ("Generated HR", high))
    ):
        for axis, plane in enumerate(("xy", "xz", "yz")):
            index = volume.shape[axis] // 2
            panels[row, axis].imshow(
                volume.select(axis, index).numpy(),
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
    parser = check_parser(
        __file__, "Inspect SR using its saved LR source; only SR --weight is needed."
    )
    parser.add_argument(
        "--weight",
        type=Path,
        help="SR generator.pt or run directory; default: latest trained SR run.",
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
        help="Output directory; default: a new directory under run/checks.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height-origin", type=float, default=0.0)
    parser.add_argument("--lr-guidance", type=float)
    parser.add_argument("--guidance", type=float, default=1.0, help="SR guidance.")
    parser.add_argument(
        "--tile-size",
        type=int,
        help="HR tile edge; default: up to 64, aligned to the SR scale.",
    )
    parser.add_argument(
        "--overlap", type=int, help="HR overlap; default: scale-aligned up to 8."
    )
    parser.add_argument(
        "--margin", type=int, help="HR context margin; default: model context."
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
    weight = resolve_weight(args.weight or latest_sr())
    payload = torch.load(weight, map_location="cpu", weights_only=True)
    if payload.get("format") != "diffusion-gan3d.sr":
        raise ValueError("--weight must be an exported SR generator.pt.")
    cfg = normalize_train_config(payload["config"], "sr")
    del payload
    lr_weight = None
    common = dict(domain=args.domain, seed=args.seed, height_origin=args.height_origin)
    if args.input:
        low = (
            load_probabilities(args.input)
            if args.input.suffix.lower() == ".pt"
            else load_volume(args.input)
        )
    else:
        stored_source = cfg.get("source", {}).get("weights")
        if args.lr_weight:
            lr_weight = args.lr_weight.resolve()
        elif stored_source:
            lr_weight = Path(stored_source)
            if not lr_weight.is_absolute():
                lr_weight = PROJECT_ROOT / lr_weight
        else:
            raise ValueError(
                "SR export has no LR source. Supply --lr-weight or --input."
            )
        print(f"LR weights: {lr_weight}\nGenerating LR...", flush=True)
        lr = InferenceAPI(lr_weight, args.device)
        validate_sr_source(cfg["data"], lr.data)
        low = lr.generate_probs(guidance=args.lr_guidance, **common)
        del lr
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    print(f"SR weights: {weight}\nRefining LR {tuple(low.shape[-3:])}...", flush=True)
    sr = SuperResolutionAPI(weight, args.device)
    tile_size = args.tile_size
    if tile_size is None and sr.scale_factor.is_integer():
        scale = int(sr.scale_factor)
        tile_size = max(scale, min(64, sr.hi_res_size) // scale * scale)
    overlap = args.overlap
    if overlap is None:
        scale = int(sr.scale_factor) if sr.scale_factor.is_integer() else 1
        overlap = (
            0
            if tile_size is None
            else min(8 // scale, (tile_size - 1) // (2 * scale)) * scale
        )
    high = sr.super_resolve(
        low,
        tile_size=tile_size,
        overlap=overlap,
        margin=args.margin,
        guidance=args.guidance,
        **common,
    )
    root = args.out or result_directory("06_check_hr")
    root.mkdir(parents=True, exist_ok=True)
    low_labels = low.argmax(0) if low.ndim == 4 else low
    save_volume(low_labels, root / "lr.tiff")
    save_volume(high, root / "hr.tiff")
    if low.ndim == 4:
        save_probabilities(low, root / "lr_probs.pt")
    low_probs = low[None] if low.ndim == 4 else phase_channels(low[None], sr.num_phases)
    # This compares categorical output occupancy, not an unknown HR ground truth.
    coarse = downsample(phase_channels(high[None], sr.num_phases), low.shape[-3:])
    report = {
        "sr_weights": str(weight),
        "lr_weights": str(lr_weight) if lr_weight else None,
        "input": str(args.input.resolve()) if args.input else None,
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
        "hr_label_phase_fractions": [
            (high == p).float().mean().item() for p in range(sr.num_phases)
        ],
        "hr_label_coarse_mae": (coarse - low_probs).abs().mean().item(),
        "has_measured_3d_reference": False,
    }
    (root / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Saved HR {tuple(high.shape)} to {root.resolve()}\nOpening/saving comparison.png",
        flush=True,
    )
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
