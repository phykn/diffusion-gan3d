"""Bounded LR ablations and LR -> SR execution/held-out-region checks.

This uses one label image under an isotropic assumption. It does not validate
anisotropic acquisition, absolute height conditioning, or measured 3D fidelity.
"""

import argparse
import contextlib
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.cli import add_device_argument, result_directory
from scripts.common.metrics import read_metrics
from src.api import LowResolutionAPI, PlaneAnchor, SuperResolutionAPI
from src.build.trainer import build_trainer
from src.config.files import PROJECT_ROOT, save_yaml
from src.config.sample import build_train_config, load_sample
from src.config.train import load_train_config
from src.device import check_device, empty_cache
from src.prepare.resize import resize_labels
from src.storage import save_volume
from src.train.checkpoint import resolve_checkpoint
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
        image = api.prepare_image(crops[index % len(crops)])
        vol = api.generate(
            anchors=(PlaneAnchor(image, axis=1, index=api.input_size // 2),),
            seed=seed,
            guidance=1.0,
            anchor_strength=1.0,
        )
        save_volume(vol, out / f"anchor_{seed}.tiff")
        section = vol[:, api.input_size // 2, :]
        target = image.argmax(0)
        stats.append(
            {
                "seed": seed,
                "anchor_accuracy": float((section == target).float().mean()),
                "phase_and_lag_mae": float(
                    np.abs(
                        section_stats(section, api.num_phases)
                        - section_stats(target, api.num_phases)
                    ).mean()
                ),
            }
        )
        volumes.append(vol)
    return stats, volumes


def shrink(cfg):
    """Small models and a short EMA for a bounded run."""
    cfg["model"]["generator"].update(
        channels=[8, 16, 32], embedding_channels=32, latent_channels=16
    )
    cfg["model"]["critic"].update(
        channels=[8, 16, 32], plane_groups=[["xy", "xz", "yz"]]
    )
    cfg["optim"]["ema_decay"] = 0.95


def train_variant(cfg, out, steps, continue_run, device):
    """Train or reuse one LR variant; return its seconds and whether a restored
    checkpoint completes the next update."""
    out.mkdir(exist_ok=continue_run)
    trainer = build_trainer(cfg, device)
    checkpoint = None
    if continue_run:
        with contextlib.suppress(FileNotFoundError):
            checkpoint = resolve_checkpoint(out)
    if checkpoint is None:
        save_yaml(out / "train.yaml", cfg)
    else:
        resume_training(
            trainer, torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
    started = time.monotonic()
    if trainer.completed_steps < steps:
        run_train(trainer, steps, steps, out, start_step=trainer.completed_steps)
    # A reused run's partial time would not be comparable.
    elapsed = None if checkpoint is not None else time.monotonic() - started
    payload = torch.load(resolve_checkpoint(out), map_location="cpu", weights_only=True)
    del trainer
    restored = build_trainer(cfg, device)
    resume_training(restored, payload)
    restored.step(steps)
    resumed = restored.completed_steps == steps + 1
    del restored, payload
    empty_cache(device)
    return elapsed, resumed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=PROJECT_ROOT / "data/sample.png")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--sr-steps", type=int, default=100)
    add_device_argument(parser)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--continue-run",
        action="store_true",
        help="Reuse completed checkpoints in --run-dir after an interrupted check.",
    )
    args = parser.parse_args()
    if args.steps < 2 or args.sr_steps < 1:
        raise ValueError("at least two LR steps and one SR step are required.")
    device = check_device(args.device)
    torch.set_num_threads(1)
    root = args.run_dir or result_directory(__file__)
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
    reference = load_train_config(build_train_config(load_sample("default"), "low_res"))
    data = reference["data"]
    crop_size = data["crop_size"]
    if min(labels.shape[0], split, labels.shape[1] - split) < crop_size:
        raise ValueError(
            f"image is too small for disjoint {crop_size}px train/validation regions."
        )
    train_images = root / "train_images"
    train_images.mkdir(exist_ok=args.continue_run)
    Image.fromarray(labels[:, :split]).save(train_images / "train.png")
    Image.fromarray(labels[:, split:]).save(root / "validation.png")
    crops = [
        torch.from_numpy(
            labels[row : row + crop_size, split : split + crop_size].copy()
        ).long()
        for row in (0, labels.shape[0] - crop_size)
    ]
    report = {
        "image": str(args.image.resolve()),
        "split_column": split,
        "purpose": "execution and short-run stability; one image, isotropic plane reuse; not 3D reconstruction quality proof",
        "grid": {
            "crop": crop_size,
            "low": data["lo_res_size"],
            "high": data["hi_res_size"],
        },
        "variants": {},
    }
    data.update(
        domains={0: {p: [str(train_images.resolve())] for p in ("xy", "xz", "yz")}},
        num_phases=phases,
    )
    shrink(reference)
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
        elapsed, resumed = train_variant(
            cfg, out, args.steps, args.continue_run, device
        )
        api = LowResolutionAPI(out / "generator.pt", device=device)
        validation, _ = evaluate(api, crops, out, [100, 101])
        del api
        records = read_metrics(out / "tensorboard")
        skips = sum(
            value
            for row in records
            for key, value in row.items()
            if key.startswith("skipped/")
        )
        report["variants"][name] = {
            "seconds": elapsed,
            "steps": args.steps,
            "resume_completed_next_step": resumed,
            "skipped_updates": skips,
            "validation": validation,
            "last_loss": records[-1]["loss/generator_total"],
        }
        (root / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(name, report["variants"][name], flush=True)
        if not resumed:
            raise RuntimeError("LR checkpoint did not resume the next update.")
    # Keep the baseline for the pipeline; a short ablation does not select a winner.
    base = root / "index_r1"
    sr_cfg = load_train_config(build_train_config(load_sample("default"), "sr"), "sr")
    sr_cfg["data"] = copy.deepcopy(reference["data"])
    shrink(sr_cfg)
    sr_cfg["lr_bank"]["samples_per_domain"] = 4
    sr_cfg["train"].update(
        total_steps=args.sr_steps,
        weights_every_steps=args.sr_steps,
        slice_pairs_per_plane=4,
    )
    save_yaml(root / "sr.yaml", sr_cfg)
    sr_run = run_sr_train(
        device=device,
        base_weights=base / "generator.pt",
        config=root / "sr.yaml",
        run_dir=root / "sr",
    )
    api = LowResolutionAPI(base / "generator.pt", device=device)
    _, low_volumes = evaluate(api, crops, root, [200, 201])
    sr = SuperResolutionAPI(sr_run / "generator.pt", device=device)
    sr_checks = []
    for index, low in enumerate(low_volumes):
        high = sr.refine(low, seed=300 + index)
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
    sr_records = read_metrics(sr_run / "tensorboard")
    report["sr"]["skipped_updates"] = {
        key: sum(row.get(key, 0) for row in sr_records)
        for key in {tag for row in sr_records for tag in row}
        if key.startswith("skipped/")
    }
    report["complete"] = True
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Report:", root / "report.json", flush=True)


if __name__ == "__main__":
    main()
