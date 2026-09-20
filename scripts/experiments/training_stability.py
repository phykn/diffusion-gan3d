"""Bounded LR ablations and LR -> SR execution/held-out-region checks.

This uses one label image under an isotropic assumption. It does not validate
anisotropic acquisition, absolute thickness conditioning, or measured 3D fidelity.
"""

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api import InferenceAPI, PlaneAnchor, SuperResolutionAPI
from src.build.trainer import build_trainer
from src.config.files import save_yaml
from src.config.train import load_train_config
from src.prepare.resize import resize_labels
from src.storage import save_volume
from src.train.run.loop import run_train
from src.train.run.sr import run_sr_train
from src.train.state import resume_training


def section_stats(labels, phases):
    values = labels.long()
    fractions = [(values == phase).float().mean().item() for phase in range(phases)]
    agreement = [
        (values[..., gap:] == values[..., :-gap]).float().mean().item()
        for gap in (1, 2, 4, 8)
        if gap < values.shape[-1]
    ]
    return np.array(fractions + agreement)


def evaluate(api, crops, out, seeds):
    stats = []
    volumes = []
    for index, seed in enumerate(seeds):
        crop = crops[index % len(crops)]
        anchor = api.prepare_image(crop)
        vol = api.generate(
            anchors=(PlaneAnchor(anchor, axis=1, index=api.input_size // 2),),
            seed=seed,
            guidance=1.0,
            anchor_strength=1.0,
        )
        save_volume(vol, out / f"anchor_{seed}.tiff")
        section = vol[:, api.input_size // 2, :]
        anchor = anchor.argmax(0)
        stats.append(
            {
                "seed": seed,
                "anchor_accuracy": float((section == anchor).float().mean()),
                "phase_and_lag_mae": float(
                    np.abs(
                        section_stats(section, api.num_phases)
                        - section_stats(anchor, api.num_phases)
                    ).mean()
                ),
            }
        )
        volumes.append(vol)
    return stats, volumes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=Path("data/sample.png"))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--sr-steps", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--continue-run",
        action="store_true",
        help="Reuse completed checkpoints in --run-dir after an interrupted check.",
    )
    args = parser.parse_args()
    if args.steps < 2 or args.sr_steps < 1:
        raise ValueError("at least two LR steps and one SR step are required.")
    torch.set_num_threads(1)
    root = args.run_dir or Path("run") / datetime.now().strftime(
        "%Y%m%d-%H%M%S-stability"
    )
    if args.continue_run and args.run_dir is None:
        raise ValueError("--continue-run requires --run-dir.")
    root.mkdir(parents=True, exist_ok=args.continue_run)
    with Image.open(args.image) as image:
        labels = np.array(image)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("provide a 2D integer phase-label image.")
    phases = int(labels.max()) + 1
    if phases < 2 or not np.array_equal(np.unique(labels), np.arange(phases)):
        raise ValueError("phase IDs must be contiguous from zero.")
    split = labels.shape[1] * 2 // 3
    crop_size = 128
    if min(labels.shape[0], split, labels.shape[1] - split) < crop_size:
        raise ValueError(
            "image is too small for disjoint 128px train/validation regions."
        )
    train_images = root / "train_images"
    train_images.mkdir(exist_ok=args.continue_run)
    Image.fromarray(labels[:, :split]).save(train_images / "train.png")
    Image.fromarray(labels[:, split:]).save(root / "validation.png")
    crops = [
        torch.from_numpy(labels[row : row + 128, split : split + 128].copy()).long()
        for row in (0, labels.shape[0] - 128)
    ]
    report = {
        "image": str(args.image.resolve()),
        "split_column": split,
        "purpose": "execution and short-run stability; one image, isotropic plane reuse; not 3D reconstruction quality proof",
        "grid": {"crop": 128, "low": 64, "high": 128},
        "variants": {},
    }
    reference = load_train_config("config/train/low_res.yaml")
    reference["data"].update(
        domains={0: {p: [str(train_images.resolve())] for p in ("xy", "xz", "yz")}},
        num_phases=phases,
    )
    reference["model"]["generator"].update(
        channels=[8, 16, 32], embedding_channels=32, latent_channels=16
    )
    reference["model"]["critic"].update(
        channels=[8, 16, 32], plane_groups=[["xy", "xz", "yz"]]
    )
    reference["conditioning"]["anchor"]["ramp_steps"] = 20
    reference["optim"]["ema_decay"] = 0.95
    reference["train"].update(
        total_steps=args.steps,
        real_batch_size=4,
        slice_pairs_per_plane=4,
        weights_every_steps=args.steps,
        archive_every_steps=None,
    )
    for name, embedding, r1, r2 in (
        ("index_r1", "index", 0.2, 0.0),
        ("scaled_r1", "scaled", 0.2, 0.0),
        ("index_r1_r2", "index", 0.1, 0.1),
    ):
        cfg = copy.deepcopy(reference)
        cfg["model"]["diffusion"]["time_embedding"] = embedding
        cfg["loss"].update(r1_weight=r1, r2_weight=r2)
        out = root / name
        out.mkdir(exist_ok=args.continue_run)
        trainer = build_trainer(cfg, torch.device(args.device))
        checkpoint = out / "checkpoints/last.pt"
        reused = args.continue_run and checkpoint.is_file()
        if reused:
            resume_training(
                trainer, torch.load(checkpoint, map_location="cpu", weights_only=True)
            )
        else:
            save_yaml(out / "train.yaml", cfg)
        started = time.monotonic()
        if trainer.completed_steps < args.steps:
            run_train(
                trainer, args.steps, args.steps, out, start_step=trainer.completed_steps
            )
        elapsed = None if reused else time.monotonic() - started
        payload = torch.load(
            out / "checkpoints/last.pt", map_location="cpu", weights_only=True
        )
        del trainer
        restored = build_trainer(cfg, torch.device(args.device))
        resume_training(restored, payload)
        restored.step(args.steps)
        resumed = restored.completed_steps == args.steps + 1
        del restored, payload
        torch.cuda.empty_cache() if args.device == "cuda" else None
        api = InferenceAPI(out / "generator.pt", device=args.device)
        validation, _ = evaluate(api, crops, out, [100, 101])
        del api
        records = [
            json.loads(line)
            for line in (out / "metrics.jsonl").read_text().splitlines()
        ]
        skips = sum(
            value
            for row in records
            for key, value in row["diagnostics"].items()
            if key.startswith("skipped/")
        )
        report["variants"][name] = {
            "seconds": elapsed,
            "steps": args.steps,
            "resume_completed_next_step": resumed,
            "skipped_updates": skips,
            "validation": validation,
            "last_loss": records[-1]["generator_total"],
        }
        (root / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(name, report["variants"][name], flush=True)
        if not resumed:
            raise RuntimeError("LR checkpoint did not resume the next update.")
    # Keep the baseline for the pipeline; a short ablation does not select a winner.
    base = root / "index_r1"
    sr_cfg = load_train_config("config/train/sr.yaml", "sr")
    sr_cfg["data"] = copy.deepcopy(reference["data"])
    sr_cfg["model"]["generator"].update(channels=8, blocks=2)
    sr_cfg["model"]["critic"].update(
        channels=[8, 16, 32], plane_groups=[["xy", "xz", "yz"]]
    )
    sr_cfg["lr_bank"]["samples_per_domain"] = 4
    sr_cfg["optim"]["ema_decay"] = 0.95
    sr_cfg["train"].update(
        total_steps=args.sr_steps,
        weights_every_steps=args.sr_steps,
        slices_per_plane=4,
    )
    save_yaml(root / "sr.yaml", sr_cfg)
    sr_run = run_sr_train(
        device=args.device,
        base_weights=base / "generator.pt",
        config=root / "sr.yaml",
        run_dir=root / "sr",
    )
    api = InferenceAPI(base / "generator.pt", device=args.device)
    _, low_volumes = evaluate(api, crops, root, [200, 201])
    sr = SuperResolutionAPI(sr_run / "generator.pt", device=args.device)
    sr_checks = []
    for index, low in enumerate(low_volumes):
        high = sr.super_resolve(low, seed=300 + index)
        save_volume(high, root / f"sr_{index}.tiff")
        reconstructed = resize_labels(high, low.shape[0], phases)
        sr_checks.append(
            {
                "low_shape": list(low.shape),
                "high_shape": list(high.shape),
                "lr_label_agreement": float((reconstructed == low).float().mean()),
                "phases": torch.unique(high).tolist(),
            }
        )
    report["sr"] = {
        "steps": args.sr_steps,
        "held_out_lr_seeds": [200, 201],
        "checks": sr_checks,
    }
    sr_records = [
        json.loads(line) for line in (sr_run / "metrics.jsonl").read_text().splitlines()
    ]
    report["sr"]["skipped_updates"] = {
        key: sum(row.get(key, 0) for row in sr_records)
        for key in sr_records[0]
        if key.startswith("skipped/")
    }
    report["complete"] = True
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Report:", root / "report.json", flush=True)


if __name__ == "__main__":
    main()
