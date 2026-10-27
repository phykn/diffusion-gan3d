import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from paper.scripts.evaluate_lr import (
    interface_density,
    nested_plane_order,
    pool_similarity,
)
from paper.scripts.evaluate_sr import lr_agreement, upsample_linear, upsample_nearest

ROOT = Path(__file__).resolve().parents[1]


def test_paper_numbers_and_figures_match_lr_results() -> None:
    paper = (ROOT / "paper/README.md").read_text(encoding="utf-8")
    results = json.loads(
        (ROOT / "paper/assets/lr-results.json").read_text(encoding="utf-8")
    )
    best = results["steps"][str(results["best_step"])]
    assert f"**{best['structure']:.2f}**" in paper
    exact = 100 * np.asarray([row["exact"] for row in results["anchor"]["internal"]])
    assert f"{exact.mean():.1f} ± {exact.std(ddof=1):.1f}%" in paper
    sixteen = [
        row["whole"] for row in results["anchor"]["coverage"] if row["count"] == 16
    ]
    assert f"{100 * np.mean(sixteen):.1f}%" in paper
    for tilt in (best["tilt_phase1"]["yx"], results["real"]["tilt_phase1"]):
        balance = f"{tilt['balance']:.2f}".replace("-", "−")
        assert f"{balance} over {tilt['count']} particles" in paper

    figures = re.findall(r'<img src="([^"]+)"', paper)
    assert figures
    assert all((ROOT / "paper" / figure).is_file() for figure in figures)


def test_paper_numbers_match_sr_results() -> None:
    paper = (ROOT / "paper/README.md").read_text(encoding="utf-8")
    results = json.loads(
        (ROOT / "paper/assets/sr-results.json").read_text(encoding="utf-8")
    )
    best = results["steps"][str(results["best_step"])]
    assert f"**{best['structure']:.2f}**" in paper
    for name in ("nearest", "linear"):
        assert f"| {results['baselines'][name]['structure']:.2f} |" in paper
    assert f"| {results['steps']['2500']['structure']:.2f} |" in paper
    for summary in (results["real"], best):
        fraction = summary["phase0_fraction"]
        assert f"{fraction['mean']:.3f} ± {fraction['std']:.3f}" in paper
    assert f"{100 * best['lr_agreement']['mean']:.1f}% of LR voxels" in paper
    assert f"differ in {100 * results['reseed']['between_seeds']['mean']:.1f}%" in paper
    tau = best["tortuosity_phase0"]
    assert (
        f"{tau['z']['mean']:.2f} ± {tau['z']['std']:.2f}, "
        f"{tau['y']['mean']:.2f} ± {tau['y']['std']:.2f}, and "
        f"{tau['x']['mean']:.2f} ± {tau['x']['std']:.2f}"
    ) in paper
    for phase, label in ((0, "pore"), (1, "solid")):
        span = 100 * best[f"percolation_phase{phase}"]["z"]["mean"]
        assert f"{span:.1f}% of {label}" in paper


def test_readme_figures_exist() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    figures = re.findall(r'<img src="([^"]+)"', readme)
    assert "paper/assets/sr-volume.png" in figures
    assert all((ROOT / figure).is_file() for figure in figures)


def test_upsampling_baselines_keep_lr_labels() -> None:
    low = torch.nn.functional.one_hot(torch.randint(2, (4, 4, 4)), 2)
    low = low.movedim(-1, 0).float()

    nearest = upsample_nearest(low)
    assert nearest.shape == (8, 8, 8)
    assert lr_agreement(nearest, low) == 1.0
    assert upsample_linear(low).shape == (8, 8, 8)


def test_nested_plane_order_covers_every_plane_in_nested_prefixes() -> None:
    order = nested_plane_order(8)

    assert sorted(order) == list(range(8))
    assert order[0] == 4
    assert set(order[:2]) <= set(order[:4])


def test_pool_similarity_rewards_coarse_match() -> None:
    target = torch.zeros((8, 8), dtype=torch.long)
    shifted = target.clone()
    shifted[0, 0] = 1

    assert pool_similarity(target, target) == 1.0
    assert pool_similarity(shifted, target) == pytest.approx(1 - 1 / 64)


def test_interface_density_averages_given_axes() -> None:
    labels = torch.tensor([[0, 1], [0, 1]])

    assert interface_density(labels, (0,)) == 0.0
    assert interface_density(labels, (1,)) == 1.0
    assert interface_density(labels, (0, 1)) == 0.5
