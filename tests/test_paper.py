import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.paper.evaluate_lr import (
    interface_density,
    nested_plane_order,
    pool_similarity,
)

ROOT = Path(__file__).resolve().parents[1]


def test_paper_numbers_and_figures_match_lr_results() -> None:
    paper = (ROOT / "PAPER.md").read_text(encoding="utf-8")
    results = json.loads(
        (ROOT / "assets/paper/lr-results.json").read_text(encoding="utf-8")
    )
    best = results["steps"][str(results["best_step"])]
    assert f"**{best['structure']:.2f}**" in paper
    exact = 100 * np.asarray([row["exact"] for row in results["anchor"]["internal"]])
    assert f"{exact.mean():.1f} ± {exact.std(ddof=1):.1f}%" in paper
    sixteen = [
        row["whole"] for row in results["anchor"]["coverage"] if row["count"] == 16
    ]
    assert f"{100 * np.mean(sixteen):.1f}%" in paper

    figures = re.findall(r'<img src="([^"]+)"', paper)
    assert figures
    assert all((ROOT / figure).is_file() for figure in figures)


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
