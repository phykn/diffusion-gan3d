import argparse
import csv
import importlib
import json
import sys
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from make_assets import OUTPUT_DIR, ROI_COLOR
from provenance import (
    build_provenance,
    describe_files,
    validate_output_paths,
)

from src.build import load_generator
from src.config import load_generation_settings
from src.scale import ScaledGenerator

OVERLAPS = (0, 4, 8, 12, 16)
SEEDS = tuple(range(20_260_808, 20_260_813))
BLOCKS = (2, 1, 1)
SEAM_EXCLUSION_RADIUS = 4
TEMP_DIR = PROJECT_ROOT / "temp"
RAW_CSV = TEMP_DIR / "overlap_ablation_raw.csv"
SUMMARY_CSV = TEMP_DIR / "overlap_ablation_summary.csv"
MANIFEST = TEMP_DIR / "overlap_ablation_manifest.json"
FIGURE = OUTPUT_DIR / "07-overlap-ablation.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weight",
        type=Path,
        required=True,
    )
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float)
    args = parser.parse_args()
    weights = args.weight.resolve()
    settings = load_generation_settings(weights)
    guidance_scale = (
        settings.guidance_scale if args.guidance_scale is None else args.guidance_scale
    )
    provenance = build_provenance(
        weights,
        guidance_scale,
        generation={
            "seeds": list(SEEDS),
            "domain": args.domain,
            "overlaps": list(OVERLAPS),
            "blocks": list(BLOCKS),
            "scale_geometry": "fixed_blocks_inward_margins",
            "default_overlap": settings.overlap,
            "crop_margin": settings.crop_margin,
            "seam_exclusion_radius": SEAM_EXCLUSION_RADIUS,
        },
    )
    validate_output_paths(
        provenance,
        (RAW_CSV, SUMMARY_CSV, MANIFEST, FIGURE),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = load_generator(weights, device=device)
    scaled = ScaledGenerator(generator)
    diagnostic = importlib.import_module("scripts.04_check_scale_up")
    rows: list[dict[str, float | int]] = []

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nOverlap ablation")
    print("----------------")
    print(f"Weights : {weights}")
    print(f"Device  : {device}")
    print(f"Guidance: {guidance_scale}")
    for seed in SEEDS:
        for overlap in OVERLAPS:
            set_seed(seed, device)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            start = perf_counter()
            volume = scaled.generate(
                blocks=BLOCKS,
                overlap=overlap,
                crop_margin=settings.crop_margin,
                progress=False,
                guidance_scale=guidance_scale,
                domain=args.domain,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = perf_counter() - start
            assert scaled.stats is not None
            plan = scaled.stats
            if len(plan.seams[0]) != 1 or plan.seams[1:] != ((), ()):
                raise RuntimeError("two-block ablation must produce one axis-0 seam.")
            quality = diagnostic.measure_seams(
                volume,
                plan.seams,
                overlap,
                generator.num_phases,
            )
            exact_ratio = exact_seam_change_ratio(
                volume,
                plan.seams[0][0],
            )
            row: dict[str, float | int] = {
                "seed": seed,
                "overlap": overlap,
                "output_axis0": plan.shape[0],
                "seam_axis0": plan.seams[0][0],
                "guidance_scale": guidance_scale,
                "exact_seam_change_ratio": exact_ratio,
                "band_change_ratio": optional_float(quality.change_ratio[0]),
                "transition_tv": optional_float(quality.transition_tv[0]),
                "continuation_delta": optional_float(quality.continuation_delta[0]),
                "elapsed_seconds": elapsed,
                "peak_memory_gib": (
                    torch.cuda.max_memory_allocated(device) / 1024**3
                    if device.type == "cuda"
                    else float("nan")
                ),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            del volume

    summary = summarize(rows, guidance_scale)
    write_csv(rows, RAW_CSV)
    write_csv(summary, SUMMARY_CSV)
    render(summary, FIGURE)
    MANIFEST.write_text(
        json.dumps(
            {
                **provenance,
                "outputs": describe_files((RAW_CSV, SUMMARY_CSV, FIGURE)),
                "seeds": list(SEEDS),
                "overlaps": list(OVERLAPS),
                "default_overlap": settings.overlap,
                "crop_margin": settings.crop_margin,
                "blocks": list(BLOCKS),
                "output_shapes": {
                    str(overlap): list(scaled.shape_from_blocks(BLOCKS, overlap))
                    for overlap in OVERLAPS
                },
                "seam_exclusion_radius": SEAM_EXCLUSION_RADIUS,
                "exact_seam_change_ratio": (
                    "phase-change rate at the exact tile boundary divided by "
                    "the median rate outside the exclusion band; ideal is 1"
                ),
                "transition_tv": (
                    "maximum total-variation distance between transition-pair "
                    "distributions in the seam band and the interior"
                ),
                "continuation_delta": (
                    "maximum absolute phase-continuation difference between "
                    "the seam band and the interior"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Raw     : {RAW_CSV.resolve()}")
    print(f"Summary : {SUMMARY_CSV.resolve()}")
    print(f"Manifest: {MANIFEST.resolve()}")
    print(f"Figure  : {FIGURE.resolve()}")


def set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def exact_seam_change_ratio(volume: torch.Tensor, seam: int) -> float:
    exact = float((volume[seam - 1] != volume[seam]).to(torch.float32).mean())
    rates = []
    for index in range(volume.shape[0] - 1):
        if seam - SEAM_EXCLUSION_RADIUS - 1 <= index <= seam + SEAM_EXCLUSION_RADIUS:
            continue
        rates.append(
            float((volume[index] != volume[index + 1]).to(torch.float32).mean())
        )
    if not rates:
        return float("nan")
    interior = float(torch.tensor(rates).median())
    return exact / interior if interior > 0.0 else float("nan")


def optional_float(value: float | None) -> float:
    return float("nan") if value is None else float(value)


def summarize(
    rows: list[dict[str, float | int]],
    guidance_scale: float,
) -> list[dict[str, float | int]]:
    metrics = (
        "exact_seam_change_ratio",
        "band_change_ratio",
        "transition_tv",
        "continuation_delta",
        "elapsed_seconds",
        "peak_memory_gib",
    )
    summary = []
    for overlap in OVERLAPS:
        selected = [row for row in rows if row["overlap"] == overlap]
        output_sizes = {int(row["output_axis0"]) for row in selected}
        seam_positions = {int(row["seam_axis0"]) for row in selected}
        if len(output_sizes) != 1 or len(seam_positions) != 1:
            raise ValueError("geometry must be constant across seeds.")
        result: dict[str, float | int] = {
            "overlap": overlap,
            "output_axis0": output_sizes.pop(),
            "seam_axis0": seam_positions.pop(),
            "guidance_scale": guidance_scale,
            "samples": len(selected),
        }
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected], dtype=float)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std(ddof=1))
        summary.append(result)
    return summary


def write_csv(rows: list[dict], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(rows: list[dict[str, float | int]], output: Path) -> None:
    overlaps = np.asarray([row["overlap"] for row in rows], dtype=float)
    figure, panels = plt.subplots(1, 3, figsize=(12.2, 3.8), facecolor="white")
    plot_errorbar(
        panels[0],
        overlaps,
        rows,
        "exact_seam_change_ratio",
        "Exact seam / interior change",
    )
    panels[0].axhline(1.0, color="#64748b", linestyle="--", linewidth=1.2)
    plot_errorbar(panels[1], overlaps, rows, "transition_tv", "Transition TV")
    plot_errorbar(
        panels[1],
        overlaps,
        rows,
        "continuation_delta",
        "Continuation delta",
        color="#2563eb",
    )
    panels[1].set_ylabel("Boundary discrepancy")
    panels[1].legend(frameon=False, fontsize=9)
    plot_errorbar(
        panels[2],
        overlaps,
        rows,
        "elapsed_seconds",
        "Elapsed time (s)",
    )
    memory = panels[2].twinx()
    plot_errorbar(
        memory,
        overlaps,
        rows,
        "peak_memory_gib",
        "Peak memory (GiB)",
        color="#2563eb",
        decorate=False,
    )
    memory.tick_params(colors="#2563eb")
    memory.spines["right"].set_color("#2563eb")
    memory.set_ylabel("Peak memory (GiB)", color="#2563eb")
    for panel, letter in zip(panels, "abc", strict=True):
        panel.text(
            0.02,
            0.96,
            f"({letter})",
            transform=panel.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    figure.tight_layout(w_pad=1.6)
    figure.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def plot_errorbar(
    panel: plt.Axes,
    overlaps: np.ndarray,
    rows: list[dict[str, float | int]],
    metric: str,
    label: str,
    color: str = ROI_COLOR,
    decorate: bool = True,
) -> None:
    means = np.asarray([row[f"{metric}_mean"] for row in rows], dtype=float)
    stds = np.asarray([row[f"{metric}_std"] for row in rows], dtype=float)
    panel.errorbar(
        overlaps,
        means,
        yerr=stds,
        color=color,
        marker="o",
        linewidth=2,
        capsize=3,
        label=label,
    )
    if decorate:
        panel.set_ylabel(label)
        panel.set_xlabel("Overlap per side (voxels)")
        panel.set_xticks(OVERLAPS)
        panel.grid(axis="y", color="#cbd5e1", linewidth=0.7, alpha=0.65)
        panel.spines[["top", "right"]].set_visible(False)


if __name__ == "__main__":
    main()
