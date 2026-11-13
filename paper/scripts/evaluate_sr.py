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
    connectivity,
    describe_volumes,
    mean_std,
    parse_args,
    real_crops,
    wall_minutes,
)
from paper.scripts.evaluate_lr import FULL, SMOKE
from src.api import SuperResolutionAPI
from src.config.io import load_generation_settings, load_yaml, stored_path
from src.data.crops import downsample
from src.stats.sections import (
    label_sections,
    section_errors,
    section_statistics,
)
from src.stats.volume import interface_density
from src.train.selection import lr_mse

DESCRIPTION = (
    "Evaluate one run's SR stage for the paper: HR structure, baselines and LR "
    "agreement. Writes sr-results.json to --out and caches the LR inputs and HR "
    "volumes in its cache/; --reuse recomputes the metrics from them "
    "without sampling."
)
# Second SR seed per LR input, to separate what SR adds from what LR fixes.
RESEED = 1000


def main(argv=None):
    evaluate(parse_args(DESCRIPTION, "SR run directory.", argv))


def evaluate(args):
    run = args.run
    # The same LR seeds as evaluate_lr.py; the LR inputs come from the LR model
    # inside the SR export.
    samples = (SMOKE if args.smoke else FULL).samples
    api = SuperResolutionAPI(run, args.device)
    config = api.config
    scale = api.scale_factor
    settings = load_generation_settings()
    scores = load_yaml(run / "selection" / "scores.yaml")
    best = int(scores["best_step"])
    cache = cache_dir(args.out, run, args.smoke)

    real_labels = label_sections(real_crops(config, config.data.hi_res_size))
    noise = section_errors(real_labels[0::2], real_labels[1::2])
    reference = section_statistics(real_labels)
    phases = real_labels.shape[1]

    low = cached(
        cache / "lr.pt",
        args.reuse,
        lambda: [
            api.lr.generate_probs(seed=seed, guidance=settings.guidance)
            for seed in range(samples)
        ],
    )
    weights = {best: run / "generator.pt"} | archived_generators(run)

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
    for step, export in sorted(weights.items()):
        volumes = cached(
            cache / f"step{step}.pt",
            args.reuse,
            lambda: refine_samples(export, args.device, low),
        )
        print(f"step {step}: {len(volumes)} volumes")
        results["steps"][str(step)] = volume_summary(
            volumes, low, reference, phases, noise, full=step == best
        )
        if step == best:
            high = volumes
    for name, upsample in (("nearest", upsample_nearest), ("linear", upsample_linear)):
        baseline = [upsample(v, scale) for v in low]
        results["baselines"][name] = volume_summary(
            baseline, low, reference, phases, noise, full=name == "nearest"
        )

    again = cached(
        cache / f"step{best}-reseed.pt",
        args.reuse,
        lambda: refine_samples(weights[best], args.device, low, RESEED),
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


def refine_samples(weights, device, volumes, offset=0):
    model = SuperResolutionAPI(weights, device)
    return [
        model.refine(volume, seed=seed) for seed, volume in enumerate(volumes, offset)
    ]


def upsample_nearest(low, scale):
    labels = low.argmax(0).to(torch.uint8)
    for dim in range(labels.ndim):
        labels = labels.repeat_interleave(scale, dim)
    return labels


def upsample_linear(low, scale):
    probs = F.interpolate(low[None], scale_factor=scale, mode="trilinear")
    return probs[0].argmax(0).to(torch.uint8)


def lr_agreement(high, low):
    """Fraction of LR voxels whose majority HR phase is the LR label."""
    onehot = F.one_hot(high.long(), len(low)).movedim(-1, 0)[None].float()
    coarse = downsample(onehot, tuple(low.shape[-3:]))[0]
    return float((coarse.argmax(0) == low.argmax(0)).float().mean())


def real_summary(real_labels, noise):
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


def volume_summary(volumes, low, reference, phases, noise, full):
    summary, clusters = describe_volumes(volumes, reference, phases, noise)
    summary.update(
        cluster_density=np.mean(clusters, 0).tolist(),
        lr_mse=mean_std(
            [lr_mse(high, source) for high, source in zip(volumes, low, strict=True)]
        ),
        lr_agreement=mean_std(
            [
                lr_agreement(high, source)
                for high, source in zip(volumes, low, strict=True)
            ]
        ),
    )
    if not full:
        return summary
    summary |= connectivity(volumes)
    return summary


if __name__ == "__main__":
    main()
