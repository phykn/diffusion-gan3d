import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from paper.scripts.common import (
    TILT_KEYS,
    archived_generators,
    cache_dir,
    cached,
    connectivity,
    first_image,
    mean_std,
    normalized,
    parse_args,
    real_crops,
    wall_minutes,
)
from src.api import LowResolutionAPI, PlaneAnchor
from src.config.io import load_config, load_generation_settings, load_yaml, stored_path
from src.data.crops import phase_channels, resize_phases
from src.stats.anchor import measure_boundaries, measure_slice_smoothness
from src.stats.sections import (
    compare_sections,
    label_sections,
    section_errors,
    section_statistics,
    two_point,
)
from src.stats.volume import (
    interface_density,
    tilt_balance,
)
from src.train.selection import pick_sections

DESCRIPTION = (
    "Evaluate one run's LR stage for the paper: structure, selection and anchors. "
    "Writes lr-results.json to --out and caches the generated volumes in its "
    "cache/<run>/; --reuse recomputes the metrics from them without sampling."
)
REFERENCE_OFFSET = 1000
DISTANCES = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32)


class Budget(NamedTuple):
    samples: int
    anchor_seeds: tuple[int, ...]
    coverage_seeds: tuple[int, ...]
    coverage_counts: tuple[int, ...]
    # (left, top) of the training-image crop used as the external anchor.
    external_crop: tuple[int, int]


FULL = Budget(
    16, tuple(range(8)), tuple(range(4)), (0, 1, 2, 4, 8, 16, 32, 64), (281, 58)
)
SMOKE = Budget(2, (0,), (0,), (0, 1), (0, 0))


def main(argv=None) -> None:
    args = parse_args(DESCRIPTION, "LR run directory.", argv)
    run = args.run
    budget = SMOKE if args.smoke else FULL
    config = load_config(run / "train.yaml")
    settings = load_generation_settings()
    scores = load_yaml(run / "selection" / "scores.yaml")
    best = int(scores["best_step"])
    cache = cache_dir(args.out, run, args.smoke)

    real_labels = label_sections(real_crops(config, config.data.lo_res_size))
    noise = section_errors(real_labels[0::2], real_labels[1::2])
    weights = {best: run / "generator.pt"} | archived_generators(run)
    source = first_image(config)[1]

    def sample(step):
        api = LowResolutionAPI(weights[step], args.device)
        return [
            api.generate(seed=seed, guidance=settings.guidance)
            for seed in range(budget.samples)
        ]

    results = {
        "run": stored_path(run),
        "best_step": best,
        "training_minutes": wall_minutes(run / "tensorboard"),
        "guidance": settings.guidance,
        "anchor_strength": settings.anchor_strength,
        "real": real_summary(real_labels, noise)
        | {"tilt_phase1": real_tilt(source, config)},
        "selection": scores["steps"],
        "steps": {},
    }
    for step in sorted(weights):
        volumes = cached(
            cache / f"step{step}.pt", args.reuse, lambda step=step: sample(step)
        )
        print(f"step {step}: {len(volumes)} volumes")
        results["steps"][str(step)] = volume_summary(
            volumes, real_labels, noise, full=step == best
        )

    api = LowResolutionAPI(weights[best], args.device)
    results["anchor"] = anchor_experiments(
        api, settings, source, budget, cache, args.reuse
    )
    output = args.out / "lr-results.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Results: {output}")


def external_anchor(api: LowResolutionAPI, source: np.ndarray, crop) -> torch.Tensor:
    """The external crop of the training image on the LR grid, as uint8 labels."""
    left, top = crop
    size = api.crop_size
    labels = torch.tensor(source[top : top + size, left : left + size]).long()
    image = api.prepare_image(labels)
    return label_sections(image[None])[0].argmax(0).to(torch.uint8)


def real_tilt(source: np.ndarray, config) -> dict:
    """Solid tilt balance of the whole training image on the LR grid."""
    data = config.data
    shape = tuple(n * data.lo_res_size // data.crop_size for n in source.shape)
    labels = torch.tensor(source).long()[None]
    probs = resize_phases(phase_channels(labels, data.num_phases), shape)
    labels = label_sections(probs)[0].argmax(0).numpy()
    tilt = tilt_balance([labels[None]], phase=1)["yx"]
    return {key: tilt[key] for key in TILT_KEYS}


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
    structure = []
    for axis, name in enumerate(("xy", "xz", "yz")):
        fake = pick_sections(volumes, axis, real_labels.shape[1])
        errors = compare_sections(real, section_statistics(fake))
        planes[name] = errors._asdict()
        structure.append(normalized(errors, noise))
    fraction = mean_std([(v == 0).float().mean() for v in volumes])
    summary = {
        "samples": len(volumes),
        "planes": planes,
        "structure": float(np.mean(structure)),
        "phase0_fraction": fraction,
        "interface_density": mean_std(
            [interface_density(v, (0, 1, 2)) for v in volumes]
        ),
    }
    if not full:
        return summary
    summary["two_point_from_fraction"] = fraction_offset_error(
        real_labels, fraction["mean"]
    )
    summary |= connectivity(volumes)
    tilt = tilt_balance([v.numpy() for v in volumes], phase=1)
    summary["tilt_phase1"] = {
        name: {key: tilt[name][key] for key in TILT_KEYS} for name in tilt
    }
    return summary


def fraction_offset_error(real_labels: torch.Tensor, phase0: float) -> float:
    """Two-point error a phase-fraction change alone would cause.

    Keeps the real normalized correlation (S2 - p^2) / (p - p^2) of each phase
    and moves only its fraction p, so the result is the part of the two-point
    error explained by the fraction offset.
    """
    errors = []
    for curves in two_point(real_labels):
        moved = []
        for curve, target in zip(curves, (phase0, 1 - phase0)):
            p = curve[0]
            shape = (curve - p * p) / (p - p * p)
            moved.append(target * target + (target - target * target) * shape)
        errors.append(float((torch.stack(moved) - curves).abs().mean()))
    return float(np.mean(errors))


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
    distance = (torch.arange(got.shape[0]) - index).abs()
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


def anchor_experiments(api, settings, source, budget, cache, reuse) -> dict:
    size = api.input_size
    center = size // 2
    strength = settings.anchor_strength
    guidance = settings.guidance

    def anchored_pair(seed, anchors):
        # The same seed gives the same noise, so the baseline shows what the
        # anchors changed.
        anchored = api.generate(
            anchors, seed=seed, anchor_strength=strength, guidance=guidance
        )
        return anchored, api.generate(seed=seed, guidance=guidance)

    def reference(seed):
        return api.generate(seed=REFERENCE_OFFSET + seed, guidance=guidance)

    def internal():
        rows = []
        for seed in budget.anchor_seeds:
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
        crop = budget.external_crop
        plane = external_anchor(api, source, crop)
        rows = []
        for seed in budget.anchor_seeds:
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
        return {"crop": list(crop), "plane": plane, "rows": rows}

    def coverage():
        order = nested_plane_order(size)
        rows = []
        for seed in budget.coverage_seeds:
            ref = reference(seed)
            for count in budget.coverage_counts:
                indices = order[:count]
                anchors = tuple(
                    PlaneAnchor(image=ref[index], axis=0, index=index)
                    for index in indices
                )
                volume = api.generate(
                    anchors,
                    seed=seed,
                    anchor_strength=strength if count else 0.0,
                    guidance=guidance,
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

    internal_rows = cached(cache / "anchor-internal.pt", reuse, internal)
    external_data = cached(cache / "anchor-external.pt", reuse, external)
    coverage_rows = cached(cache / "anchor-coverage.pt", reuse, coverage)
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
