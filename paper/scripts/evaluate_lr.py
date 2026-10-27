"""Evaluate one run's LR stage for the paper: structure, selection and anchors.

Writes paper/assets/lr-results.json. Generated volumes are cached under
checks/paper-lr/; --reuse recomputes the metrics from them without sampling.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.common.diagnostic import generate_with_baseline
from src.build.data import build_datasets
from src.build.predict import build_generator
from src.config.files import load_yaml
from src.config.generation import load_generation_settings
from src.config.train import load_train_config
from src.device import default_device
from src.evaluate.anchor import measure_boundaries, measure_slice_smoothness
from src.evaluate.connectivity import spanning_fractions
from src.evaluate.structure import (
    compare_sections,
    label_sections,
    section_errors,
    section_statistics,
)
from src.evaluate.tilt import tilt_balance
from src.evaluate.tortuosity import tortuosity
from src.predict.anchor import PlaneAnchor
from src.prepare.resize import phase_channels, resize_crop, resize_phases
from src.random import seeded_rng
from src.train.selection import pick_sections

OUTPUT = PROJECT_ROOT / "paper" / "assets" / "lr-results.json"
CACHE = PROJECT_ROOT / "checks" / "paper-lr"
# Independent of the selection reference (seed 0), so the best step is not
# scored on the crops that chose it.
REAL_SEED = 1
REAL_CROPS = 128
SAMPLES = 16
CHECKPOINTS = (2500, 5000, 7500, 10000)
ANCHOR_SEEDS = tuple(range(8))
REFERENCE_OFFSET = 1000
COVERAGE_SEEDS = tuple(range(4))
COVERAGE_COUNTS = (0, 1, 2, 4, 8, 16, 32, 64)
# (left, top) of the training-image crop used as the external anchor.
EXTERNAL_CROP = (281, 58)
DISTANCES = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="LR run directory.")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    run = args.run.resolve()
    device = torch.device(args.device)
    cfg = load_train_config(run / "train.yaml")
    settings = load_generation_settings()
    scores = load_yaml(run / "selection" / "scores.yaml")
    best = int(scores["best_step"])
    CACHE.mkdir(parents=True, exist_ok=True)

    dataset = build_datasets(cfg)[0][0]
    real = real_crops(dataset)
    real_labels = label_sections(real)
    noise = section_errors(real_labels[0::2], real_labels[1::2])
    weights = {best: run / "generator.pt"} | {
        step: run / "checkpoints" / f"generator_{step:08d}.pt" for step in CHECKPOINTS
    }

    def sample(step):
        gen = build_generator(weights[step], cfg, device)
        volumes = []
        for seed in range(SAMPLES):
            with seeded_rng(seed, device):
                volumes.append(gen.generate(guidance=settings.guidance))
        return volumes

    results = {
        "run": str(run.relative_to(PROJECT_ROOT)),
        "best_step": best,
        "guidance": settings.guidance,
        "anchor_strength": settings.anchor_strength,
        "real": real_summary(real_labels, noise) | {"tilt_phase1": real_tilt(dataset)},
        "selection": scores["steps"],
        "steps": {},
    }
    for step in sorted(weights):
        volumes = cached(f"step{step}", args.reuse, lambda step=step: sample(step))
        print(f"step {step}: {len(volumes)} volumes")
        results["steps"][str(step)] = volume_summary(
            volumes, real_labels, noise, full=step == best
        )

    gen = build_generator(weights[best], cfg, device)
    results["anchor"] = anchor_experiments(gen, settings, device, dataset, args.reuse)
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
    """LR phase fractions of fixed random crops of the training image."""
    rng = np.random.RandomState(REAL_SEED)
    path = dataset.path_groups[0][0]
    return torch.stack(
        [dataset.sample(path, rng)["image"] for _ in range(REAL_CROPS)]
    ).float()


def external_anchor(dataset) -> torch.Tensor:
    """The fixed external crop on the LR grid, as uint8 labels."""
    source = dataset.decode(dataset.path_groups[0][0].resolve())
    left, top = EXTERNAL_CROP
    size = dataset.crop_size
    labels = torch.from_numpy(source[top : top + size, left : left + size].copy())
    image = resize_crop(labels.long(), dataset.patch_size, dataset.num_phases)
    return label_sections(image[None])[0].argmax(0).to(torch.uint8)


def real_tilt(dataset) -> dict:
    """Solid tilt balance of the whole training image on the LR grid."""
    source = torch.from_numpy(dataset.decode(dataset.path_groups[0][0].resolve()))
    shape = tuple(n * dataset.patch_size // dataset.crop_size for n in source.shape)
    probs = resize_phases(
        phase_channels(source.long()[None], dataset.num_phases), shape
    )
    labels = label_sections(probs)[0].argmax(0).numpy()
    tilt = tilt_balance([labels[None]], phase=1)["yx"]
    return {key: tilt[key] for key in ("count", "balance", "mean", "sem")}


def interface_density(labels: torch.Tensor, dims) -> float:
    values = [
        (
            labels.narrow(dim, 1, labels.shape[dim] - 1)
            != labels.narrow(dim, 0, labels.shape[dim] - 1)
        )
        .float()
        .mean()
        for dim in dims
    ]
    return float(torch.stack(values).mean())


def mean_std(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def real_summary(real_labels: torch.Tensor, noise) -> dict:
    labels = real_labels.argmax(1)
    return {
        "crops": len(labels),
        "phase0_fraction": mean_std((labels == 0).float().mean((1, 2))),
        "interface_density": mean_std(
            [interface_density(image, (0, 1)) for image in labels]
        ),
        "half_split_errors": noise._asdict(),
    }


def volume_summary(volumes, real_labels, noise, full: bool) -> dict:
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
    tilt = tilt_balance([v.numpy() for v in volumes], phase=1)
    summary["tilt_phase1"] = {
        name: {key: tilt[name][key] for key in ("count", "balance", "mean", "sem")}
        for name in tilt
    }
    return summary


def nested_plane_order(size: int) -> tuple[int, ...]:
    """Farthest-point plane order, so every prefix covers the volume broadly."""
    selected: list[int] = []
    remaining = set(range(size))
    while remaining:
        if not selected:
            choice = size // 2
        else:
            choice = max(
                remaining,
                key=lambda index: (
                    min(abs(index - prior) for prior in selected),
                    -abs(index - (size - 1) / 2),
                    -index,
                ),
            )
        selected.append(choice)
        remaining.remove(choice)
    return tuple(selected)


def pool_similarity(first: torch.Tensor, second: torch.Tensor, cell: int = 4) -> float:
    """One minus the mean total variation of 4x4-cell phase fractions."""
    pool = torch.nn.functional.avg_pool2d

    def fractions(labels):
        onehot = torch.nn.functional.one_hot(labels.long(), 2).movedim(-1, 0).float()
        return pool(onehot[None], cell)[0]

    return float(1 - 0.5 * (fractions(first) - fractions(second)).abs().sum(0).mean())


def plane_metrics(anchored, baseline, plane, axis, index, reference=None) -> dict:
    """How one anchored plane is met, against a same-noise baseline."""
    got = anchored.movedim(axis, 0)
    base = baseline.movedim(axis, 0)
    boundary = measure_boundaries(anchored, (index,), axis)
    base_boundary = measure_boundaries(baseline, (index,), axis)
    smooth = measure_slice_smoothness(anchored, (index,), axis, baseline)
    size = got.shape[0]
    positions = torch.arange(size)
    distance = (positions - index).abs()
    row = {
        "exact": float((got[index] == plane).float().mean()),
        "exact_baseline": float((base[index] == plane).float().mean()),
        "pool4": pool_similarity(got[index], plane),
        "pool4_baseline": pool_similarity(base[index], plane),
        "first_change_ratio": boundary.change_ratio,
        "first_change_ratio_baseline": base_boundary.change_ratio,
        "p95_flicker_ratio": smooth.p95_ratio,
        "max_flicker_ratio": smooth.max_ratio,
        "divergence": {},
    }
    if reference is not None:
        row["reference"] = {}
        ref = reference.movedim(axis, 0)
    for d in DISTANCES:
        rows = distance == d
        if not rows.any():
            continue
        row["divergence"][d] = float((got[rows] != base[rows]).float().mean())
        if reference is not None:
            row["reference"][d] = {
                "anchored": float((got[rows] == ref[rows]).float().mean()),
                "baseline": float((base[rows] == ref[rows]).float().mean()),
            }
    return row


def anchor_experiments(gen, settings, device, dataset, reuse) -> dict:
    size = gen.patch_size
    center = size // 2
    strength = settings.anchor_strength

    def anchored_pair(seed, anchors):
        with seeded_rng(seed, device):
            return generate_with_baseline(
                gen, anchors, strength, guidance=settings.guidance
            )

    def reference(seed):
        with seeded_rng(REFERENCE_OFFSET + seed, device):
            return gen.generate(guidance=settings.guidance)

    def internal():
        rows = []
        for seed in ANCHOR_SEEDS:
            ref = reference(seed)
            for axis in range(3):
                plane = ref.movedim(axis, 0)[center]
                anchored, baseline = anchored_pair(
                    seed, (PlaneAnchor(image=plane, axis=axis, index=center),)
                )
                rows.append(
                    {
                        "seed": seed,
                        "axis": axis,
                        "reference": ref,
                        "anchored": anchored,
                        "baseline": baseline,
                    }
                )
        return rows

    def external():
        plane = external_anchor(dataset)
        rows = []
        for seed in ANCHOR_SEEDS:
            for index in (center, 0):
                anchored, baseline = anchored_pair(
                    seed, (PlaneAnchor(image=plane, axis=0, index=index),)
                )
                rows.append(
                    {
                        "seed": seed,
                        "index": index,
                        "anchored": anchored,
                        "baseline": baseline,
                    }
                )
        return {"plane": plane, "rows": rows}

    def coverage():
        order = nested_plane_order(size)
        rows = []
        for seed in COVERAGE_SEEDS:
            ref = reference(seed)
            for count in COVERAGE_COUNTS:
                indices = order[:count]
                anchors = tuple(
                    PlaneAnchor(image=ref[index], axis=0, index=index)
                    for index in indices
                )
                with seeded_rng(seed, device):
                    volume = gen.generate(
                        anchors=anchors,
                        anchor_strength=strength if count else 0.0,
                        guidance=settings.guidance,
                    )
                rows.append(
                    {
                        "seed": seed,
                        "count": count,
                        "indices": list(indices),
                        "reference": ref,
                        "volume": volume,
                    }
                )
        return rows

    internal_rows = cached("anchor-internal", reuse, internal)
    external_data = cached("anchor-external", reuse, external)
    coverage_rows = cached("anchor-coverage", reuse, coverage)
    print("anchor volumes ready")

    result = {"internal": [], "external": [], "coverage": []}
    for row in internal_rows:
        plane = row["reference"].movedim(row["axis"], 0)[center]
        metrics = plane_metrics(
            row["anchored"],
            row["baseline"],
            plane,
            row["axis"],
            center,
            row["reference"],
        )
        result["internal"].append({"seed": row["seed"], "axis": row["axis"], **metrics})
    for row in external_data["rows"]:
        metrics = plane_metrics(
            row["anchored"], row["baseline"], external_data["plane"], 0, row["index"]
        )
        result["external"].append(
            {"seed": row["seed"], "index": row["index"], **metrics}
        )
    for row in coverage_rows:
        volume, ref, indices = row["volume"], row["reference"], row["indices"]
        free = torch.ones(size, dtype=torch.bool)
        free[indices] = False
        result["coverage"].append(
            {
                "seed": row["seed"],
                "count": row["count"],
                "whole": float((volume == ref).float().mean()),
                "anchored_planes": None
                if not indices
                else float((volume[indices] == ref[indices]).float().mean()),
                "free_planes": None
                if not free.any()
                else float((volume[free] == ref[free]).float().mean()),
            }
        )
    return result


if __name__ == "__main__":
    main()
