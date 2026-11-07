import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from paper.scripts.common import (
    archived_generators,
    cache_dir,
    cached,
    mean_std,
    parse_args,
    real_crops,
    wall_minutes,
)
from paper.scripts.evaluate_lr import normalized
from src.api import SuperResolutionAPI
from src.config.io import load_generation_settings, load_yaml, stored_path
from src.data.crops import downsample
from src.stats.sections import (
    compare_sections,
    label_sections,
    section_errors,
    section_statistics,
)
from src.stats.volume import interface_density, spanning_fractions, tortuosity
from src.train.selection import lr_mse, pick_sections

DESCRIPTION = (
    "Evaluate one run's SR stage for the paper: HR structure, baselines and LR "
    "agreement. Writes sr-results.json to --out and caches the LR inputs and HR "
    "volumes in its cache/<run>/; --reuse recomputes the metrics from them "
    "without sampling."
)
# The same LR seeds as evaluate_lr.py; the LR inputs come from the LR model
# inside the SR export.
SAMPLES = 16
SMOKE_SAMPLES = 2
# Second SR seed per LR input, to separate what SR adds from what LR fixes.
RESEED = 1000


def main(argv=None) -> None:
    args = parse_args(DESCRIPTION, "SR run directory.", argv)
    run = args.run
    samples = SMOKE_SAMPLES if args.smoke else SAMPLES
    api = SuperResolutionAPI(run, args.device)
    config = api.config
    scale = api.scale_factor
    settings = load_generation_settings()
    scores = load_yaml(run / "selection" / "scores.yaml")
    best = int(scores["best_step"])
    cache = cache_dir(args.out, run, args.smoke)

    real_labels = label_sections(real_crops(config, config.data.hi_res_size))
    noise = section_errors(real_labels[0::2], real_labels[1::2])

    def sample_lr():
        return [
            api.lr.generate_probs(seed=seed, guidance=settings.guidance)
            for seed in range(samples)
        ]

    low = cached(cache / "lr.pt", args.reuse, sample_lr)
    weights = {best: run / "generator.pt"} | archived_generators(run)

    def refine(step, offset=0):
        sr = SuperResolutionAPI(weights[step], args.device)
        return [sr.refine(v, seed=offset + i) for i, v in enumerate(low)]

    results = {
        "run": stored_path(run),
        "lr_run": config.source.lr_run,
        "best_step": best,
        "training_minutes": wall_minutes(run / "tensorboard"),
        "lr_guidance": settings.guidance,
        "real": real_summary(real_labels, noise),
        "selection": scores["steps"],
        "steps": {},
        "baselines": {},
    }
    for step in sorted(weights):
        volumes = cached(
            cache / f"step{step}.pt", args.reuse, lambda step=step: refine(step)
        )
        print(f"step {step}: {len(volumes)} volumes")
        results["steps"][str(step)] = volume_summary(
            volumes, low, real_labels, noise, full=step == best
        )
        if step == best:
            high = volumes
    for name, upsample in (("nearest", upsample_nearest), ("linear", upsample_linear)):
        baseline = [upsample(v, scale) for v in low]
        results["baselines"][name] = volume_summary(
            baseline, low, real_labels, noise, full=name == "nearest"
        )

    again = cached(
        cache / f"step{best}-reseed.pt", args.reuse, lambda: refine(best, RESEED)
    )
    nearest = [upsample_nearest(v, scale) for v in low]
    results["reseed"] = {
        "between_seeds": mean_std(
            [(a != b).float().mean() for a, b in zip(high, again)]
        ),
        "from_nearest": mean_std(
            [(a != b).float().mean() for a, b in zip(high, nearest)]
        ),
    }
    output = args.out / "sr-results.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Results: {output}")


def upsample_nearest(low: torch.Tensor, scale: int) -> torch.Tensor:
    labels = low.argmax(0)
    for dim in range(3):
        labels = labels.repeat_interleave(scale, dim)
    return labels.to(torch.uint8)


def upsample_linear(low: torch.Tensor, scale: int) -> torch.Tensor:
    probs = F.interpolate(low[None], scale_factor=scale, mode="trilinear")
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
        # Separate regions of each phase per section pixel.
        "cluster_density": section_statistics(real_labels).clusters[0].tolist(),
    }


def volume_summary(volumes, low, real_labels, noise, full: bool) -> dict:
    real = section_statistics(real_labels)
    planes = {}
    structure = []
    clusters = []
    for axis, name in enumerate(("xy", "xz", "yz")):
        fake = section_statistics(pick_sections(volumes, axis, real_labels.shape[1]))
        errors = compare_sections(real, fake)
        planes[name] = errors._asdict()
        structure.append(normalized(errors, noise))
        clusters.append(fake.clusters[0])
    summary = {
        "samples": len(volumes),
        "planes": planes,
        "structure": float(np.mean(structure)),
        "cluster_density": np.mean(clusters, 0).tolist(),
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
