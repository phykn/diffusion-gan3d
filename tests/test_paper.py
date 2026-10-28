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


def read_paper() -> str:
    return (ROOT / "paper/README.md").read_text(encoding="utf-8")


def read_results(stage: str) -> dict:
    path = ROOT / f"paper/assets/{stage}-results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def table(paper: str, header: str) -> dict[str, list[str]]:
    """Cells of the paper table whose first column is header, by row name."""
    lines = paper.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"| {header} |"))
    rows = {}
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        name, *cells = (cell.strip().strip("*") for cell in line.strip("|").split("|"))
        rows[name] = cells
    return rows


def spread(values, digits: int, scale: float = 1.0, unit: str = "") -> str:
    """Mean ± standard deviation of a summary or of raw values, as the paper prints it."""
    if isinstance(values, dict):
        mean, std = values["mean"], values["std"]
    else:
        mean, std = np.mean(values), np.std(values, ddof=1)
    return f"{scale * mean:.{digits}f} ± {scale * std:.{digits}f}{unit}"


def section_cells(summary: dict) -> list[str]:
    planes = summary["planes"].values()
    error = {
        key: np.mean([plane[key] for plane in planes])
        for key in ("two_point_mae", "chord_tv", "cluster_error")
    }
    return [
        f"{summary['structure']:.2f}",
        f"{error['two_point_mae']:.4f}",
        f"{error['chord_tv']:.4f}",
        f"{error['cluster_error']:.3f}",
        spread(summary["phase0_fraction"], 3),
        spread(summary["interface_density"], 4),
    ]


def real_cells(real: dict) -> list[str]:
    return ["—"] * 4 + [
        spread(real["phase0_fraction"], 3),
        spread(real["interface_density"], 4),
    ]


def test_paper_numbers_and_figures_match_lr_results() -> None:
    paper = read_paper()
    results = read_results("lr")
    best = results["steps"][str(results["best_step"])]
    assert f"**{best['structure']:.2f}**" in paper
    for tilt in (best["tilt_phase1"]["yx"], results["real"]["tilt_phase1"]):
        balance = f"{tilt['balance']:.2f}".replace("-", "−")
        assert f"{balance} over {tilt['count']} particles" in paper

    figures = re.findall(r'<img src="([^"]+)"', paper)
    assert figures
    assert all((ROOT / "paper" / figure).is_file() for figure in figures)


def test_paper_numbers_match_sr_results() -> None:
    paper = read_paper()
    results = read_results("sr")
    best = results["steps"][str(results["best_step"])]
    assert f"**{best['structure']:.2f}**" in paper
    for name in ("nearest", "linear"):
        assert f"| {results['baselines'][name]['structure']:.2f} |" in paper
    assert f"| {results['steps']['2500']['structure']:.2f} |" in paper
    for summary in (results["real"], best):
        fraction = summary["phase0_fraction"]
        assert f"{fraction['mean']:.3f} ± {fraction['std']:.3f}" in paper
    assert f"{100 * best['lr_agreement']['mean']:.1f}% of LR voxels" in paper
    reseed = results["reseed"]
    assert f"changes {100 * reseed['from_nearest']['mean']:.1f}% of HR" in paper
    assert f"differ in {100 * reseed['between_seeds']['mean']:.1f}%" in paper
    tau = best["tortuosity_phase0"]
    assert (
        f"{tau['z']['mean']:.2f} ± {tau['z']['std']:.2f}, "
        f"{tau['y']['mean']:.2f} ± {tau['y']['std']:.2f}, and "
        f"{tau['x']['mean']:.2f} ± {tau['x']['std']:.2f}"
    ) in paper
    for phase, label in ((0, "pore"), (1, "solid")):
        span = 100 * best[f"percolation_phase{phase}"]["z"]["mean"]
        assert f"{span:.1f}% of {label}" in paper


def test_lr_table_matches_results() -> None:
    results = read_results("lr")
    rows = table(read_paper(), "Step")
    best, last = results["best_step"], max(map(int, results["steps"]))
    labels = {last: " (last)"}
    labels[best] = " (selected, last)" if best == last else " (selected)"

    assert rows.pop("Real crops") == real_cells(results["real"])
    for step, summary in results["steps"].items():
        assert rows.pop(step + labels.get(int(step), "")) == section_cells(summary)
    assert not rows


def test_sr_table_matches_results() -> None:
    results = read_results("sr")
    rows = table(read_paper(), "Volumes")

    def agreement(summary):
        value = 100 * summary["lr_agreement"]["mean"]
        return "100%" if value == 100 else f"{value:.1f}%"

    assert rows.pop("Real crops") == real_cells(results["real"]) + ["—"]
    for name, label in (("nearest", "nearest"), ("linear", "trilinear")):
        summary = results["baselines"][name]
        assert rows.pop(f"LR, {label}") == section_cells(summary) + [agreement(summary)]
    for step, summary in results["steps"].items():
        selected = " (selected)" if int(step) == results["best_step"] else ""
        assert rows.pop(f"SR, step {step}{selected}") == section_cells(summary) + [
            agreement(summary)
        ]
    assert not rows


def test_anchor_table_matches_results() -> None:
    anchor = read_results("lr")["anchor"]
    rows = table(read_paper(), "Anchor")
    center = max(row["index"] for row in anchor["external"])
    cases = {
        "Internal": anchor["internal"],
        "External, center": [r for r in anchor["external"] if r["index"] == center],
        "External, boundary": [r for r in anchor["external"] if r["index"] == 0],
    }

    for name, group in cases.items():

        def column(key, scale=1.0, digits=2, unit=""):
            return spread([row[key] for row in group], digits, scale, unit)

        assert rows.pop(name) == [
            str(len(group)),
            column("exact_baseline", 100, 1, "%"),
            column("exact", 100, 1, "%"),
            column("pool4", 100, 1, "%"),
            column("first_change_ratio"),
            column("first_change_ratio_baseline"),
            column("p95_flicker_ratio"),
        ]
    assert not rows


def test_paper_text_numbers_match_results() -> None:
    paper = read_paper()
    lr, sr = read_results("lr"), read_results("sr")
    assert f"({lr['training_minutes']:.0f} minutes" in paper
    assert f"({sr['training_minutes'] / 60:.1f} hours)" in paper
    for results, text in ((lr, "within"), (sr, "by")):
        best = results["steps"][str(results["best_step"])]
        real = results["real"]["interface_density"]["mean"]
        gap = 100 * abs(1 - best["interface_density"]["mean"] / real)
        assert f"{text} {gap:.1f}%" in paper

    best = lr["best_step"]
    selected = next(row for row in lr["selection"] if row["step"] == best)
    scores = (
        f"{lr['steps'][str(best)]['structure']:.2f} against {selected['structure']:.2f}"
    )
    assert f"({scores} for step {best})" in paper

    anchor = lr["anchor"]
    reference = [row["reference"] for row in anchor["internal"]]
    for distance, text in ((1, "one voxel away"), (4, "at four"), (12, "at twelve")):
        agreement = np.mean([r[str(distance)]["anchored"] for r in reference])
        assert f"{100 * agreement:.1f}% {text}" in paper
    divergence = [
        100 * np.mean([row["divergence"][d] for row in anchor["internal"]])
        for d in anchor["internal"][0]["divergence"]
    ]
    assert f"differ in {min(divergence):.0f}–{max(divergence):.0f}% of voxels" in paper
    for count, text in ((0, "no anchors"), (8, "eight planes"), (16, "sixteen")):
        whole = [r["whole"] for r in anchor["coverage"] if r["count"] == count]
        assert f"{100 * np.mean(whole):.1f}% with {text}" in paper


def test_anode_table_matches_results() -> None:
    results = read_results("anode")
    rows = table(read_paper(), "Measure")

    for axis in "zyx":
        lr, sr = (results[s]["tortuosity_phase0"][axis] for s in ("lr", "sr"))
        assert rows.pop(f"Pore tortuosity, {axis}") == [
            spread(lr["generated"], 2),
            f"{lr['real']['mean']:.2f}",
            spread(sr["generated"], 2),
            f"{sr['real']['mean']:.2f}",
        ]
        for phase, name in (("1", "Graphite"), ("2", "Silicon")):
            lr, sr = (results[s]["chord_mean"][phase][axis] for s in ("lr", "sr"))
            assert rows.pop(f"{name} chord, {axis}") == [
                spread(lr["generated"], 2),
                f"{lr['real']['mean']:.2f}",
                spread(sr["generated"], 2),
                f"{sr['real']['mean']:.2f}",
            ]
    for pair in ("zy", "zx", "yx"):
        balances = [
            f"{results[stage]['tilt_phase1'][source][pair]['balance']:.2f}".replace(
                "-", "−"
            )
            for stage in ("lr", "sr")
            for source in ("generated", "real")
        ]
        assert rows.pop(f"Graphite tilt balance, d{pair[0]}/d{pair[1]}") == balances
    assert not rows


def test_paper_numbers_match_anode_results() -> None:
    paper = read_paper()
    results = read_results("anode")
    lr, sr = results["lr"], results["sr"]
    assert f"({results['lr_training_minutes']:.0f} minutes)" in paper
    assert f"({results['training_minutes'] / 60:.1f} hours)" in paper
    assert f"with step {results['lr_best_step']} selected" in paper

    def band(stage, phase, source):
        return f"{stage['profile']['bands'][source][phase][-1]:.3f}"

    assert f"{band(lr, 2, 'generated')} silicon against {band(lr, 2, 'real')}" in paper
    assert f"({band(lr, 1, 'generated')} against {band(lr, 1, 'real')})" in paper
    assert f"({band(sr, 2, 'generated')} against {band(sr, 2, 'real')})" in paper
    silicon = sr["phase_fractions"]["2"]
    assert (
        f"{silicon['generated']['mean']:.3f} against {silicon['real']['mean']:.3f}"
        in paper
    )
    assert f"{100 * sr['lr_agreement']['mean']:.1f}% of LR voxels" in paper
    graphite = sr["phase_fractions"]["1"]["real_values"]
    assert f"({graphite[0]:.3f} and {graphite[1]:.3f})" in paper

    sections = lr["sections"]
    assert (
        f"{sections['structure_xy']:.2f} in xy and "
        f"{sections['structure_side']:.2f} on the side planes"
    ) in paper
    baselines = results["baselines"]
    assert (
        f"SR scores {sr['sections']['structure']:.2f} at source resolution, against "
        f"{baselines['nearest']['sections']['structure']:.2f} and "
        f"{baselines['linear']['sections']['structure']:.2f}"
    ) in paper

    def worst(stage):
        return 100 * max(
            abs(entry["generated"]["mean"] / entry["real"]["mean"] - 1)
            for entry in stage["tortuosity_phase0"].values()
        )

    assert (
        f"at most {worst(lr):.1f}% on the LR grid and {worst(sr):.1f}% at source"
        in paper
    )
    assert f"tortuosity within {worst(sr):.1f}%" in paper


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
