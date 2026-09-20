"""Exercise LR conditions, repeated extent growth, SR and preserved HR growth.

Saves categorical TIFFs, fractional LR tensors, slice previews and report.json.
--smoke uses untrained tiny networks: execution evidence, never quality evidence.
"""

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api import InferenceAPI, PlaneAnchor, SuperResolutionAPI, extend_hr
from src.build.model import build_denoiser, build_sr_model
from src.config.files import save_yaml
from src.config.train import load_train_config
from src.storage import save_model, save_probabilities, save_volume


def smoke_weights(root):
    cfg = load_train_config("config/train/low_res.yaml")
    cfg["data"].update(crop_size=16, lo_res_size=8, hi_res_size=16)
    cfg["conditioning"]["height_enabled"] = False
    cfg["model"]["generator"].update(
        channels=[4, 8], embedding_channels=8, latent_channels=4
    )
    cfg["model"]["diffusion"]["num_steps"] = 2
    cfg["model"]["gradient_checkpointing"] = False
    cfg["train"]["mixed_precision"] = False
    lr_path = root / "untrained_low_res/generator.pt"
    save_model(lr_path, build_denoiser(cfg, checkpointing=False))
    save_yaml(lr_path.parent / "train.yaml", cfg)
    sr_cfg = load_train_config("config/train/sr.yaml", "sr")
    sr_cfg["data"] = copy.deepcopy(cfg["data"])
    sr_cfg["model"]["generator"].update(
        channels=[4, 8], embedding_channels=8, latent_channels=4
    )
    sr_cfg["model"]["diffusion"] = copy.deepcopy(cfg["model"]["diffusion"])
    sr_cfg["model"]["gradient_checkpointing"] = False
    sr_cfg["conditioning"]["height_enabled"] = False
    sr_cfg["train"]["mixed_precision"] = False
    sr_path = root / "untrained_sr/generator.pt"
    sr_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "diffusion-gan3d.sr",
            "config": sr_cfg,
            "model": build_sr_model(sr_cfg).state_dict(),
        },
        sr_path,
    )
    return lr_path, sr_path


def check_combinations(
    lr, sr, root, *, seed=0, domain=0, image=None, height_origin=0.0
):
    root.mkdir(parents=True, exist_ok=True)
    if not sr.scale_factor.is_integer():
        raise ValueError("The extent-growth matrix requires an integer SR scale.")
    n, scale = lr.input_size, int(sr.scale_factor)
    labels = (
        (torch.arange(lr.crop_size**2).reshape(lr.crop_size, -1) % lr.num_phases)
        if image is None
        else image
    )
    prepared = lr.prepare_image(labels)
    vf = prepared.mean((-2, -1)).tolist()
    anchor = PlaneAnchor(prepared, 0, n // 2, (0, 0))
    common = dict(domain=domain, seed=seed, height_origin=height_origin)
    report = []

    def save(name, volume, preserved=None):
        categorical = volume.argmax(0).to(torch.uint8) if volume.ndim == 4 else volume
        save_volume(categorical, root / f"{name}.tiff")
        if volume.ndim == 4:
            save_probabilities(volume, root / f"{name}.pt")
            if not torch.isfinite(volume).all() or not torch.allclose(
                volume.sum(0), torch.ones_like(volume[0]), atol=1e-5
            ):
                raise AssertionError(f"Invalid fractions: {name}")
        plane = categorical[len(categorical) // 2].numpy()
        Image.fromarray(
            (plane * (255 / max(1, lr.num_phases - 1))).astype(np.uint8)
        ).save(root / f"{name}.png")
        record = {
            "case": name,
            "shape": list(categorical.shape),
            "fractions": [
                (categorical == p).float().mean().item() for p in range(lr.num_phases)
            ],
        }
        if preserved is not None:
            slices = tuple(slice(0, n) for n in preserved.shape[-3:])
            old = preserved.argmax(0) if preserved.ndim == 4 else preserved
            rate = (categorical[slices] == old).float().mean().item()
            record["preserved_label_fraction"] = rate
            if rate != 1.0:
                raise AssertionError(f"Existing labels changed: {name}")
        report.append(record)
        print(f"{name}: {tuple(categorical.shape)}", flush=True)

    low = None
    for name, anchors, fractions in (
        ("none", (), None),
        ("vf", (), vf),
        ("anchor", (anchor,), None),
        ("anchor_vf", (anchor,), vf),
    ):
        low = lr.generate_probs(anchors=anchors, vf=fractions, **common)
        save(f"lr_{name}", low)
        high = sr.super_resolve(low, margin=0, **common)
        save(f"hr_{name}", high)
    # All extra anchors lie outside the base, avoiding contradictory observations.
    extra = PlaneAnchor(prepared, 0, 2 * n - 1, (0, 0))
    expanded = lr.generate_probs(
        shape=(2 * n, n, n),
        base=low,
        base_offset=(0, 0, 0),
        preserve_base=True,
        anchors=(extra,),
        vf=vf,
        overlap=0,
        **common,
    )
    save("lr_expanded", expanded, low)
    repeated = lr.generate_probs(
        shape=(3 * n, n, n),
        base=expanded,
        base_offset=(0, 0, 0),
        preserve_base=True,
        overlap=0,
        **common,
    )
    save("lr_expanded_again", repeated, expanded)
    high = sr.super_resolve(
        repeated, tile_size=n * scale, overlap=0, margin=0, **common
    )
    save("hr_tiled", high)
    expanded, grown = extend_hr(
        lr,
        sr,
        high,
        (4 * n * scale, n * scale, n * scale),
        low=repeated,
        lr_overlap=0,
        overlap=0,
        margin=0,
        base_height_origin=height_origin,
        **common,
    )
    save("lr_for_hr_extension", expanded, repeated)
    save("hr_extended", grown, high)
    _, again = extend_hr(
        lr,
        sr,
        grown,
        (5 * n * scale, n * scale, n * scale),
        low=expanded,
        lr_overlap=0,
        overlap=0,
        margin=0,
        base_height_origin=height_origin,
        **common,
    )
    save("hr_extended_again", again, grown)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr-weights", type=Path)
    parser.add_argument("--sr-weights", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--height-origin", type=float, default=0.0)
    parser.add_argument("--anchor-image", type=Path)
    args = parser.parse_args(argv)
    root = args.output or Path("run") / (
        datetime.now().strftime("%m%d%H%M%S") + "_prediction_combinations"
    )
    if args.smoke:
        if args.lr_weights or args.sr_weights:
            parser.error("--smoke cannot be combined with trained weights.")
        torch.set_num_threads(1)
        args.lr_weights, args.sr_weights = smoke_weights(root)
    elif not (args.lr_weights and args.sr_weights):
        parser.error("supply --lr-weights and --sr-weights, or --smoke.")
    image = None
    if args.anchor_image:
        with Image.open(args.anchor_image) as source:
            image = torch.from_numpy(np.array(source))
    report = check_combinations(
        InferenceAPI(args.lr_weights, args.device),
        SuperResolutionAPI(args.sr_weights, args.device),
        root,
        seed=args.seed,
        domain=args.domain,
        image=image,
        height_origin=args.height_origin,
    )
    (root / "report.json").write_text(
        json.dumps(
            {
                "untrained_smoke": args.smoke,
                "quality_validated": False,
                "anchor_source": str(args.anchor_image)
                if args.anchor_image
                else "synthetic phase pattern",
                "lr_weights": str(args.lr_weights),
                "sr_weights": str(args.sr_weights),
                "seed": args.seed,
                "cases": report,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Report: {root / 'report.json'}")


if __name__ == "__main__":
    main()
