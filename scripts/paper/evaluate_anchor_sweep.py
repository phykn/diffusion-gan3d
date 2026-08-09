"""Generate and evaluate the axis-0 anchor coverage sweep for PAPER.md."""

import argparse
import csv
import json
import sys
from importlib.metadata import version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import taufactor as tau
import tifffile
import torch
from torchmetrics.image.kid import KernelInceptionDistance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from make_assets import OUTPUT_DIR, ROI_COLOR
from provenance import (
    build_provenance,
    describe_files,
    sha256_file,
    validate_manifest,
    validate_output_paths,
    verify_provenance_inputs,
)

from src.anchor import PlaneAnchor
from src.build import load_generator

REFERENCE_PATH = PROJECT_ROOT / "scripts" / "gt_128.tiff"
TEMP_DIR = PROJECT_ROOT / "temp"
MANIFEST_PATH = TEMP_DIR / "anchor_sweep_manifest.json"
COUNTS = (128, 64, 32, 16, 8, 4, 2, 1, 0)
AXIS = 0
PORE_PHASE = 0
SEED = 0
TAUFACTOR_CONVERGENCE = 1e-3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="reuse existing sweep TIFF files instead of regenerating them",
    )
    args = parser.parse_args()

    reference_digest = sha256_file(REFERENCE_PATH)
    reference = load_volume(REFERENCE_PATH)
    if sha256_file(REFERENCE_PATH) != reference_digest:
        raise ValueError("paper reference changed while it was being loaded.")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weights = args.weight.resolve()
    generation = {
        "seed": SEED,
        "axis": AXIS,
        "anchor_counts": list(COUNTS),
        "reference_shape": list(reference.shape),
        "pore_phase": PORE_PHASE,
    }
    provenance = build_provenance(
        weights,
        args.guidance_scale,
        generation=generation,
        reference=REFERENCE_PATH,
        source_files=(__file__,),
    )
    if provenance["reference_sha256"] != reference_digest:
        raise ValueError("paper reference changed before provenance was recorded.")
    cached_paths = [volume_path(count) for count in COUNTS]
    csv_path = TEMP_DIR / "anchor_sweep_metrics.csv"
    figure_path = OUTPUT_DIR / "06-anchor-sweep-metrics.png"
    validate_output_paths(
        provenance,
        (*cached_paths, csv_path, MANIFEST_PATH, figure_path),
    )

    if args.reuse:
        validate_manifest(
            MANIFEST_PATH,
            provenance,
            label="anchor sweep reuse",
            cached_paths=cached_paths,
        )
    else:
        generate_volumes(reference, weights, args.guidance_scale, provenance)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kid = KernelInceptionDistance(
        feature=2048,
        subsets=100,
        subset_size=50,
        reset_real_features=False,
        normalize=False,
    ).to(device)
    kid.update(metric_images(get_slices(reference, AXIS), device), real=True)
    reference_porosity = porosity(reference)
    reference_tortuosity = tortuosity(reference, AXIS, device)
    rows = []
    for count in COUNTS:
        path = volume_path(count)
        volume = load_volume(path)
        accuracy = voxel_accuracy(volume, reference)
        kid.update(metric_images(get_slices(volume, AXIS), device), real=False)
        torch.manual_seed(SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED)
        kid_mean, kid_std = kid.compute()
        row = {
            "anchor_count": count,
            "coverage": count / reference.shape[AXIS],
            "guidance_scale": args.guidance_scale,
            "kid": float(kid_mean.cpu()),
            "kid_std": float(kid_std.cpu()),
            "porosity": porosity(volume),
            "tortuosity_axis0": tortuosity(volume, AXIS, device),
            "voxel_accuracy": accuracy,
            "tiff": path.name,
        }
        rows.append(row)
        kid.reset()
        score = f"{accuracy:.4%}"
        print(
            f"count={count:2d} coverage={row['coverage']:.2%} "
            f"KID={row['kid']:.6f}±{row['kid_std']:.6f} "
            f"porosity={row['porosity']:.4f} "
            f"tortuosity={row['tortuosity_axis0']:.4f} accuracy={score}"
        )

    write_csv(rows, csv_path)
    render_chart(
        rows,
        reference_porosity=reference_porosity,
        reference_tortuosity=reference_tortuosity,
        output=figure_path,
    )
    write_manifest(
        provenance=provenance,
        cached_paths=cached_paths,
        derived_outputs=(csv_path, figure_path),
        reference=reference,
        reference_porosity=reference_porosity,
        reference_tortuosity=reference_tortuosity,
        output=MANIFEST_PATH,
    )
    print(f"Metrics : {csv_path.resolve()}")
    print(f"Manifest: {MANIFEST_PATH.resolve()}")
    print(f"Figure  : {figure_path.resolve()}")


def generate_volumes(
    reference: np.ndarray,
    weights: Path,
    guidance_scale: float,
    provenance: dict[str, object],
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = load_generator(weights, device=device)
    verify_provenance_inputs(provenance)
    if (
        generator.patch_size != reference.shape[0]
        or reference.shape != (generator.patch_size,) * 3
    ):
        raise ValueError("reference volume must match the generator patch size.")
    if not generator.anchor_enabled:
        raise ValueError("selected weights were trained with anchors disabled.")
    reference_tensor = torch.from_numpy(reference).to(torch.long)
    slices = reference_tensor.movedim(AXIS, 0)

    print("\nAnchor coverage sweep")
    print("---------------------")
    print(f"Weights : {weights.resolve()}")
    print(f"Reference: {REFERENCE_PATH.resolve()}")
    print(f"Device  : {device}")
    print(f"Guidance: {guidance_scale}")
    for count in COUNTS:
        indices = select_indices(reference.shape[AXIS], count)
        anchors = tuple(
            PlaneAnchor(image=slices[index], axis=AXIS, index=index)
            for index in indices
        )
        torch.manual_seed(SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(SEED)
        print(
            f"Status  : generating count={count:2d} "
            f"coverage={count / reference.shape[AXIS]:.2%}...",
            flush=True,
        )
        volume = generator.generate(
            anchors=anchors,
            guidance_scale=guidance_scale,
        )
        path = volume_path(count)
        tifffile.imwrite(path, volume.numpy())
        print(f"Output  : {path.resolve()}")


def select_indices(size: int, count: int) -> tuple[int, ...]:
    """Match the coverage selection used by scripts/03_check_anchor.py."""
    if count == 0:
        return ()
    return tuple(((2 * index + 1) * size) // (2 * count) for index in range(count))


def volume_path(count: int) -> Path:
    return TEMP_DIR / f"anchor_axis0_count_{count:02d}.tiff"


def load_volume(path: Path) -> np.ndarray:
    volume = np.asarray(tifffile.imread(path))
    if volume.ndim != 3 or volume.dtype != np.uint8 or volume.size == 0:
        raise ValueError(f"volume must be a non-empty 3D uint8 array: {path}")
    if int(volume.max()) > 1:
        raise ValueError(f"volume contains a phase outside 0 and 1: {path}")
    return volume


def get_slices(volume: np.ndarray, axis: int) -> np.ndarray:
    return np.moveaxis(volume, axis, 0)


def porosity(volume: np.ndarray) -> float:
    return float(np.mean(volume == PORE_PHASE))


def voxel_accuracy(
    volume: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Fraction of all generated voxels equal to the reference volume."""
    return float(np.mean(volume == reference))


def tortuosity(volume: np.ndarray, axis: int, device: torch.device) -> float:
    """Return TauFactor's diffusion-based tortuosity factor along ``axis``."""
    # TauFactor solves along the first dimension and expects conductive voxels
    # to equal 1, so phase-0 pores are converted to its conductive mask.
    conductive = np.moveaxis(volume == PORE_PHASE, axis, 0).astype(np.uint8)
    solver = tau.Solver(conductive, device=device.type)
    value = solver.solve(verbose=False, conv_crit=TAUFACTOR_CONVERGENCE)
    return float(np.asarray(value).reshape(-1)[0])


def metric_images(slices: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert binary slices to TorchMetrics' uint8 RGB image input."""
    images = torch.from_numpy(slices.copy()).to(torch.uint8).mul_(255)
    return images[:, None].repeat(1, 3, 1, 1).to(device)


def write_csv(rows: list[dict], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(
    provenance: dict[str, object],
    cached_paths: list[Path],
    derived_outputs: tuple[Path, ...],
    reference: np.ndarray,
    reference_porosity: float,
    reference_tortuosity: float,
    output: Path,
) -> None:
    data = {
        **provenance,
        "cached_outputs": describe_files(cached_paths),
        "outputs": describe_files(derived_outputs),
        "reference_shape": list(reference.shape),
        "axis": AXIS,
        "pore_phase": PORE_PHASE,
        "seed": SEED,
        "anchor_counts": list(COUNTS),
        "reference_porosity": reference_porosity,
        "reference_tortuosity_axis0": reference_tortuosity,
        "kid": {
            "reference_slices": reference.shape[0],
            "generated_slices": reference.shape[0],
            "implementation": "torchmetrics.image.kid.KernelInceptionDistance",
            "package_version": version("torchmetrics"),
            "feature_extractor": "torch-fidelity Inception-v3 pool 3",
            "feature_dimensions": 2048,
            "subsets": 100,
            "subset_size": 50,
            "polynomial_kernel_degree": 3,
            "input": "uint8 RGB, phase 0 = 0 and phase 1 = 255",
            "axis": AXIS,
        },
        "tortuosity": {
            "method": "TauFactor steady-state diffusion Solver (classical tortuosity factor)",
            "package_version": version("taufactor"),
            "conductive_phase": PORE_PHASE,
            "conductive_value_supplied_to_taufactor": 1,
            "convergence_criterion": TAUFACTOR_CONVERGENCE,
            "boundary_conditions": "Dirichlet along axis 0; no-flux on transverse boundaries",
            "axis": AXIS,
        },
    }
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")


def render_chart(
    rows: list[dict],
    reference_porosity: float,
    reference_tortuosity: float,
    output: Path,
) -> None:
    positions = np.asarray([row["anchor_count"] for row in rows], dtype=float)
    figure, panels = plt.subplots(2, 2, figsize=(11, 7.3), facecolor="white")
    figure.subplots_adjust(
        left=0.08,
        right=0.97,
        bottom=0.12,
        top=0.97,
        hspace=0.30,
        wspace=0.28,
    )
    specs = (
        ("voxel_accuracy", None, "Voxel accuracy"),
        ("kid", None, "KID"),
        ("porosity", reference_porosity, "Porosity"),
        (
            "tortuosity_axis0",
            reference_tortuosity,
            "Tortuosity",
        ),
    )
    for panel, letter, (key, reference, ylabel) in zip(
        panels.ravel(),
        "abcd",
        specs,
        strict=True,
    ):
        values = np.asarray(
            [np.nan if row[key] is None else row[key] for row in rows],
            dtype=float,
        )
        panel.plot(
            positions,
            values,
            color=ROI_COLOR,
            linewidth=2.2,
            marker="o",
            markersize=5.5,
        )
        if reference is not None and np.isfinite(reference):
            panel.axhline(
                reference,
                color="#64748b",
                linewidth=1.2,
                linestyle="--",
                label=f"GT: {reference:.3f}",
            )
            panel.legend(frameon=False, fontsize=9, loc="best")
        panel.set_ylabel(ylabel, fontsize=10)
        maximum = max(COUNTS)
        panel.set_xscale("symlog", base=2, linthresh=1)
        panel.set_xlim(0, maximum)
        panel.set_xticks(tuple(sorted(COUNTS)))
        panel.set_xticklabels(tuple(map(str, sorted(COUNTS))), fontsize=8)
        panel.text(
            0.01,
            0.97,
            f"({letter})",
            transform=panel.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
        panel.grid(axis="y", color="#cbd5e1", linewidth=0.7, alpha=0.65)
        panel.spines[["top", "right"]].set_visible(False)
        panel.spines[["left", "bottom"]].set_color("#94a3b8")
        panel.tick_params(colors="#475569")
    panels[0, 0].set_ylim(0.5, 1)
    figure.supxlabel("Number of supplied axis-0 anchor planes", fontsize=11, y=0.025)
    figure.savefig(
        output,
        dpi=180,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
