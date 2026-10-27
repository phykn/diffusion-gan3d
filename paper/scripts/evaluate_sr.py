"""Evaluate one run's SR stage for the paper: HR structure, baselines, LR agreement.

Writes paper/assets/sr-results.json. LR inputs and HR volumes are cached under
checks/paper-sr/; --reuse recomputes the metrics from them without sampling.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from paper.scripts.evaluate_lr import interface_density, mean_std
from src.build.data import build_datasets
from src.build.predict import build_generator
from src.config.files import load_yaml
from src.config.generation import load_generation_settings
from src.config.train import load_train_config
from src.device import default_device
from src.evaluate.connectivity import spanning_fractions
from src.evaluate.structure import (
    compare_sections,
    label_sections,
    section_errors,
    section_statistics,
)
from src.evaluate.tortuosity import tortuosity
from src.predict.sr.inference import SuperResolutionAPI, load_sr_export
from src.prepare.resize import downsample
from src.random import seeded_rng
from src.train.run.source import frozen_source_weights
from src.train.selection import lr_mse, pick_sections

OUTPUT = PROJECT_ROOT / "paper" / "assets" / "sr-results.json"
CACHE = PROJECT_ROOT / "checks" / "paper-sr"
# Same crops and LR seeds as evaluate_lr.py: independent of the selection
# reference (seed 0), and the LR inputs are the selected LR step's volumes.
REAL_SEED = 1
REAL_CROPS = 128
SAMPLES = 16
CHECKPOINTS = (2500,)
# Second SR seed per LR input, to separate what SR adds from what LR fixes.
RESEED = 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="SR run directory.")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    run = args.run.resolve()
    device = torch.device(args.device)
    cfg = load_sr_export(run / "generator.pt")["config"]
    lr_weights = frozen_source_weights(cfg["source"])
    lr_cfg = load_train_config(lr_weights.parent / "train.yaml")
    settings = load_generation_settings()
    scores = load_yaml(run / "selection" / "scores.yaml")
    best = int(scores["best_step"])
    CACHE.mkdir(parents=True, exist_ok=True)

    real = real_crops(build_datasets(cfg, high_resolution=True)[0][0])
    real_labels = label_sections(real)
    noise = section_errors(real_labels[0::2], real_labels[1::2])

    def sample_lr():
        gen = build_generator(lr_weights, lr_cfg, device)
        volumes = []
        for seed in range(SAMPLES):
            with seeded_rng(seed, device):
                volumes.append(gen.generate_probs(guidance=settings.guidance).cpu())
        return volumes

    low = cached("lr", args.reuse, sample_lr)
    weights = {best: run / "generator.pt"} | {
        step: run / "checkpoints" / f"generator_{step:08d}.pt" for step in CHECKPOINTS
    }

    def refine(step, offset=0):
        api = SuperResolutionAPI(weights[step], device)
        return [api.refine(v, seed=offset + i).cpu() for i, v in enumerate(low)]

    results = {
        "run": str(run.relative_to(PROJECT_ROOT)),
        "lr_run": str(lr_weights.parent.relative_to(PROJECT_ROOT)),
        "best_step": best,
        "lr_guidance": settings.guidance,
        "real": real_summary(real_labels, noise),
        "selection": scores["steps"],
        "steps": {},
        "baselines": {},
    }
    for step in sorted(weights):
        volumes = cached(f"step{step}", args.reuse, lambda step=step: refine(step))
        print(f"step {step}: {len(volumes)} volumes")
        results["steps"][str(step)] = volume_summary(
            volumes, low, real_labels, noise, full=step == best
        )
        if step == best:
            high = volumes
    for name, upsample in (("nearest", upsample_nearest), ("linear", upsample_linear)):
        high = [upsample(v) for v in low]
        results["baselines"][name] = volume_summary(
            high, low, real_labels, noise, full=name == "nearest"
        )

    again = cached(f"step{best}-reseed", args.reuse, lambda: refine(best, RESEED))
    nearest = [upsample_nearest(v) for v in low]
    results["reseed"] = {
        "between_seeds": mean_std(
            [(a != b).float().mean() for a, b in zip(high, again)]
        ),
        "from_nearest": mean_std(
            [(a != b).float().mean() for a, b in zip(high, nearest)]
        ),
    }
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Results: {OUTPUT}")


def cached(name, reuse, make):
    path = CACHE / f"{name}.pt"
    if reuse and path.is_file():
        return torch.load(path, weights_only=True)
    value = make()
    torch.save(value, path)
    return value


def real_crops(dataset) -> torch.Tensor:
    """Phase fractions of fixed random crops at the source resolution."""
    rng = np.random.RandomState(REAL_SEED)
    path = dataset.path_groups[0][0]
    return torch.stack(
        [dataset.sample(path, rng)["image"] for _ in range(REAL_CROPS)]
    ).float()


def upsample_nearest(low: torch.Tensor) -> torch.Tensor:
    labels = low.argmax(0)
    for dim in range(3):
        labels = labels.repeat_interleave(2, dim)
    return labels.to(torch.uint8)


def upsample_linear(low: torch.Tensor) -> torch.Tensor:
    probs = F.interpolate(low[None], scale_factor=2, mode="trilinear")
    return probs[0].argmax(0).to(torch.uint8)


def lr_agreement(high: torch.Tensor, low: torch.Tensor) -> float:
    """Fraction of LR voxels whose majority HR phase is the LR label."""
    onehot = F.one_hot(high.long(), len(low)).movedim(-1, 0)[None].float()
    coarse = downsample(onehot, tuple(low.shape[-3:]))[0]
    return float((coarse.argmax(0) == low.argmax(0)).float().mean())


def real_summary(real_labels: torch.Tensor, noise) -> dict:
    labels = real_labels.argmax(1)
    return {
        "crops": len(labels),
        "size": labels.shape[-1],
        "phase0_fraction": mean_std((labels == 0).float().mean((1, 2))),
        "interface_density": mean_std(
            [interface_density(image, (0, 1)) for image in labels]
        ),
        "half_split_errors": noise._asdict(),
    }


def volume_summary(volumes, low, real_labels, noise, full: bool) -> dict:
    real = section_statistics(real_labels)
    planes = {}
    normalized = []
    for axis, name in enumerate(("xy", "xz", "yz")):
        fake = pick_sections(volumes, axis, real_labels.shape[1])
        errors = compare_sections(real, section_statistics(fake))
        planes[name] = errors._asdict()
        normalized.append(np.mean([v / n for v, n in zip(errors, noise)]))
    summary = {
        "samples": len(volumes),
        "planes": planes,
        "structure": float(np.mean(normalized)),
        "phase0_fraction": mean_std([(v == 0).float().mean() for v in volumes]),
        "interface_density": mean_std(
            [interface_density(v, (0, 1, 2)) for v in volumes]
        ),
        "lr_mse": mean_std([lr_mse(h, v) for h, v in zip(volumes, low)]),
        "lr_agreement": mean_std([lr_agreement(h, v) for h, v in zip(volumes, low)]),
    }
    if not full:
        return summary
    for phase in (0, 1):
        spans = np.asarray([spanning_fractions((v == phase).numpy()) for v in volumes])
        summary[f"percolation_phase{phase}"] = {
            axis: mean_std(spans[:, index]) for index, axis in enumerate("zyx")
        }
    taus = np.asarray(
        [[tortuosity(v, 0, axis, device="cpu") for axis in range(3)] for v in volumes]
    )
    summary["tortuosity_phase0"] = {
        axis: mean_std(taus[:, index]) for index, axis in enumerate("zyx")
    }
    return summary


if __name__ == "__main__":
    main()
