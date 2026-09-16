import argparse
import json
from pathlib import Path

import torch

from src.config import find_train_config, load_train_config, validate_sr_source
from src.predict.inference import InferenceAPI
from src.predict.sr import SuperResolutionAPI
from src.storage import load_probabilities, load_volume, save_probabilities, save_volume


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate LR 3D and optionally super-resolve it, or super-resolve a saved LR TIFF."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--weights", type=Path, help="Stage-1 generator.pt or its run directory."
    )
    source.add_argument(
        "--input",
        type=Path,
        help="Existing LR label TIFF or fractional C,D,H,W tensor (.pt).",
    )
    parser.add_argument("--sr-weights", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--domain", type=int)
    parser.add_argument(
        "--height-origin",
        type=float,
        default=0.0,
        help="Crop origin along thickness, in source-image pixels.",
    )
    parser.add_argument("--tile-size", type=int)
    parser.add_argument("--overlap", type=int, default=8)
    args = parser.parse_args(argv)
    if args.input and not args.sr_weights:
        raise ValueError("--input requires --sr-weights.")
    sr = SuperResolutionAPI(args.sr_weights, args.device) if args.sr_weights else None
    if args.weights:
        base = InferenceAPI(args.weights, args.device)
        data = load_train_config(find_train_config(base.weights))["data"]
        if sr is not None:
            validate_sr_source(sr.config["data"], data)
        sample = base.generate_probs if sr is not None else base.generate
        low = sample(
            domain=args.domain, seed=args.seed, height_origin=args.height_origin
        )
        del base
        if args.device == "cuda":
            torch.cuda.empty_cache()
    else:
        low = (
            load_probabilities(args.input)
            if args.input.suffix.lower() == ".pt"
            else load_volume(args.input)
        )
    output = (
        low
        if sr is None
        else sr.super_resolve(
            low,
            args.domain,
            args.seed,
            args.tile_size,
            args.overlap,
            args.height_origin,
        )
    )
    save_volume(output, args.output)
    if sr is not None and args.weights:
        save_volume(
            low.argmax(0).to(torch.uint8),
            args.output.with_name(args.output.stem + "_lr.tiff"),
        )
        save_probabilities(
            low, args.output.with_name(args.output.stem + "_lr_probs.pt")
        )
    metadata = {
        "stage1_weights": str(args.weights) if args.weights else None,
        "input": str(args.input) if args.input else None,
        "sr_weights": str(args.sr_weights) if args.sr_weights else None,
        "seed": args.seed,
        "domain": args.domain,
        "lr_shape": list(low.shape[-3:]),
        "height_origin": args.height_origin,
        "output_shape": list(output.shape),
    }
    if sr:
        metadata.update(
            crop_size=sr.crop_size,
            lo_res_size=sr.lo_res_size,
            scale_factor=sr.scale_factor,
            source_pixels_per_hr_voxel=sr.crop_size / sr.hi_res_size,
        )
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {tuple(output.shape)} to {args.output}")


if __name__ == "__main__":
    main()
