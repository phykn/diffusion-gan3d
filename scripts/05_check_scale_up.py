"""Generate and inspect a jointly denoised tiled volume."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import tifffile
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.build import load_sampler
from src.generate.sample import find_weights
from src.generate.scale import generate_scaled

BLOCKS = (4, 4, 4)
OVERLAP = 32


@dataclass(frozen=True)
class SeamQuality:
    change_ratio: tuple[float | None, float | None, float | None]
    transition_tv: tuple[float | None, float | None, float | None]
    continuation_delta: tuple[float | None, float | None, float | None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--napari",
        action="store_true",
        help="show the complete scaled label volume in Napari",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = find_weights(PROJECT_ROOT / "run")
    sampler = load_sampler(weights, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    started = perf_counter()
    labels, stats = generate_scaled(
        sampler,
        blocks=BLOCKS,
        overlap=OVERLAP,
    )
    elapsed = perf_counter() - started
    shape_name = "x".join(str(length) for length in stats.shape)
    output = weights.parent / f"scaled_joint_{shape_name}.tiff"
    tifffile.imwrite(output, labels.numpy())
    seam_quality = _measure_seams(
        labels,
        stats.seams,
        stats.overlap,
        sampler.num_phases,
    )

    if args.napari:
        _show_napari(labels)
    else:
        _show_slices(
            labels,
            sampler.num_phases,
            stats.seams,
        )

    fractions = torch.bincount(
        labels.to(torch.long).flatten(),
        minlength=sampler.num_phases,
    ).to(torch.float64)
    fractions = fractions / fractions.sum()
    print(f"weights={weights.resolve()}")
    print(f"output={output.resolve()}")
    print(f"shape={tuple(labels.shape)}")
    print(f"block_grid={stats.block_grid} block_count={stats.block_count}")
    print(f"phase_fractions={[round(float(value), 4) for value in fractions]}")
    print(f"seam_change_ratio={_format_scores(seam_quality.change_ratio)}")
    print(f"seam_transition_tv={_format_scores(seam_quality.transition_tv)}")
    print(f"seam_continuation_delta={_format_scores(seam_quality.continuation_delta)}")
    print(f"elapsed_seconds={elapsed:.1f}")
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"peak_cuda_memory_gib={peak:.2f}")


def _measure_seams(
    labels: torch.Tensor,
    seams: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    overlap: int,
    num_phases: int,
) -> SeamQuality:
    change_ratios = []
    transition_tvs = []
    continuation_deltas = []
    for axis, locations in enumerate(seams):
        if not locations:
            change_ratios.append(None)
            transition_tvs.append(None)
            continuation_deltas.append(None)
            continue
        previous = labels.narrow(axis, 0, labels.shape[axis] - 1)
        following = labels.narrow(axis, 1, labels.shape[axis] - 1)
        changed = (previous != following).to(torch.float32)
        dimensions = tuple(value for value in range(3) if value != axis)
        rates = changed.mean(dim=dimensions)
        band_indices = torch.tensor(
            sorted(
                {
                    index
                    for location in locations
                    for index in range(
                        max(0, location - overlap // 2 - 1),
                        min(
                            rates.shape[0],
                            location - overlap // 2 + overlap,
                        ),
                    )
                }
            ),
            dtype=torch.long,
        )
        keep = torch.ones(rates.shape[0], dtype=torch.bool)
        keep[band_indices] = False
        interior_indices = torch.where(keep)[0]
        if interior_indices.numel() == 0:
            change_ratios.append(None)
            transition_tvs.append(None)
            continuation_deltas.append(None)
            continue
        interior_rate = float(rates[keep].median())
        if interior_rate > 0.0:
            ratios = rates.index_select(0, band_indices) / interior_rate
            worst = int((ratios - 1.0).abs().argmax())
            change_ratios.append(float(ratios[worst]))
        else:
            change_ratios.append(None)

        interior_counts = _count_transitions(
            previous.index_select(axis, interior_indices),
            following.index_select(axis, interior_indices),
            num_phases,
        )
        interior_distribution = interior_counts / interior_counts.sum()
        interior_continuation = _calc_continuation(interior_counts)
        axis_tvs = []
        axis_continuation_deltas = []
        for band_index in band_indices:
            selected = band_index.reshape(1)
            seam_counts = _count_transitions(
                previous.index_select(axis, selected),
                following.index_select(axis, selected),
                num_phases,
            )
            seam_distribution = seam_counts / seam_counts.sum()
            axis_tvs.append(
                float(0.5 * (seam_distribution - interior_distribution).abs().sum())
            )
            seam_continuation = _calc_continuation(seam_counts)
            axis_continuation_deltas.append(
                float((seam_continuation - interior_continuation).abs().max())
            )
        transition_tvs.append(max(axis_tvs))
        continuation_deltas.append(max(axis_continuation_deltas))
    return SeamQuality(
        change_ratio=tuple(change_ratios),
        transition_tv=tuple(transition_tvs),
        continuation_delta=tuple(continuation_deltas),
    )


def _count_transitions(
    previous: torch.Tensor,
    following: torch.Tensor,
    num_phases: int,
) -> torch.Tensor:
    pairs = previous.to(torch.long) * num_phases + following.to(torch.long)
    return (
        torch.bincount(
            pairs.flatten(),
            minlength=num_phases * num_phases,
        )
        .to(torch.float64)
        .reshape(num_phases, num_phases)
    )


def _calc_continuation(counts: torch.Tensor) -> torch.Tensor:
    totals = counts.sum(dim=1)
    return counts.diagonal() / totals.clamp_min(1.0)


def _show_slices(
    labels: torch.Tensor,
    num_phases: int,
    seams: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
) -> None:
    middle = tuple(size // 2 for size in labels.shape)
    images = (
        labels[middle[0], :, :],
        labels[:, middle[1], :],
        labels[:, :, middle[2]],
    )
    figure, panels = plt.subplots(1, 3, figsize=(11, 4))
    for axis, (panel, image) in enumerate(zip(panels, images, strict=True)):
        panel.imshow(
            image.numpy(),
            cmap="gray",
            vmin=-0.5,
            vmax=num_phases - 0.5,
            interpolation="nearest",
        )
        visible_axes = tuple(value for value in range(3) if value != axis)
        for location in seams[visible_axes[0]]:
            panel.axhline(location - 0.5, color="#e63946", linewidth=0.6)
        for location in seams[visible_axes[1]]:
            panel.axvline(location - 0.5, color="#e63946", linewidth=0.6)
        panel.set_title(f"axis {axis}")
        panel.axis("off")
    figure.suptitle("Joint tiled diffusion scale-up")
    figure.tight_layout()
    plt.show()


def _show_napari(labels: torch.Tensor) -> None:
    import napari

    viewer = napari.Viewer()
    viewer.add_labels(labels.numpy(), name="scaled phases")
    viewer.dims.ndisplay = 3
    napari.run()


def _format_scores(
    values: tuple[float | None, float | None, float | None],
) -> list[float | None]:
    return [None if value is None else round(value, 4) for value in values]


if __name__ == "__main__":
    main()
