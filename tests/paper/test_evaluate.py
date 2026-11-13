"""Smoke tests: the paper scripts on tiny runs write the asset keys and figures."""

import json
from pathlib import Path

import pytest

from paper.scripts import evaluate_anode, evaluate_lr, evaluate_sr, make_figures
from paper.scripts.common import cache_dir, cached

ROOT = Path(__file__).parents[2]
ASSETS = ROOT / "paper" / "assets"
# Keys that are numbers: step numbers and anchor distances.
NUMBERED = {"steps", "divergence", "reference"}
FIGURES = {
    "01-training-data.png",
    "02-generated-volume.png",
    "03-generated-slices.png",
    "lr-training.png",
    "lr-anchor.png",
    "sr-training.png",
    "sr-sections.png",
    "sr-volume.png",
    "anode-profile.png",
    "anode-sections.png",
    "anode-volume.png",
}


def key_paths(value, path=()):
    """Every key path of nested results; list items share the path "[]"."""
    if isinstance(value, list):
        return set().union(set(), *(key_paths(item, (*path, "[]")) for item in value))
    if not isinstance(value, dict):
        return set()
    paths = set()
    for key, item in value.items():
        name = "<n>" if path and path[-1] in NUMBERED else key
        paths |= {(*path, name)} | key_paths(item, (*path, name))
    return paths


@pytest.fixture(scope="module")
def results(tiny_lr_run, tiny_sr_run, tiny_anode_sr_run, tmp_path_factory):
    out = tmp_path_factory.mktemp("paper")
    common = ("--device", "cpu", "--smoke", "--out", str(out))
    evaluate_lr.main(["--run", str(tiny_lr_run), *common])
    evaluate_sr.main(["--run", str(tiny_sr_run), *common])
    evaluate_anode.main(["--run", str(tiny_anode_sr_run), *common])
    return out


@pytest.mark.parametrize("stage", ["lr", "sr", "anode"])
def test_smoke_results_have_the_asset_keys(results, stage):
    name = f"{stage}-results.json"
    written = json.loads((results / name).read_text(encoding="utf-8"))
    expected = json.loads((ASSETS / name).read_text(encoding="utf-8"))

    assert key_paths(written) == key_paths(expected)


def test_reuse_reads_the_cached_volumes(results, tiny_lr_run, monkeypatch):
    def sample(*args, **options):
        raise AssertionError("--reuse generated a volume.")

    monkeypatch.setattr(evaluate_lr.LowResolutionAPI, "generate", sample)
    path = results / "lr-results.json"
    first = path.read_text(encoding="utf-8")
    evaluate_lr.main(
        ["--run", str(tiny_lr_run), "--device", "cpu", "--smoke", "--reuse"]
        + ["--out", str(results)]
    )

    assert path.read_text(encoding="utf-8") == first


def test_figures_from_smoke_results(results, tmp_path):
    make_figures.main(["--results", str(results), "--out", str(tmp_path)])

    assert {path.name for path in tmp_path.iterdir()} == FIGURES


def test_a_full_evaluation_refuses_a_smoke_cache(results, tiny_lr_run):
    with pytest.raises(SystemExit, match="--smoke volumes"):
        evaluate_lr.main(
            ["--run", str(tiny_lr_run), "--device", "cpu", "--reuse"]
            + ["--out", str(results)]
        )


def close(actual, expected, path=()):
    """Nested results equal up to float rounding of the stored JSON."""
    if isinstance(expected, dict):
        assert set(actual) == set(expected), path
        for key in expected:
            close(actual[key], expected[key], (*path, key))
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            close(a, b, (*path, index))
    else:
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-9), path


def test_the_real_crops_are_those_behind_the_assets():
    """Structure scores divide by the real crops' half-split errors, so the
    evaluation must draw the same crops as the runs in paper/assets."""
    from paper.scripts.common import first_image, real_crops
    from src.config.sample import build_stage_configs, load_sample
    from src.stats.sections import label_sections, section_errors

    configs = build_stage_configs(load_sample("default"))
    for stage, module, size in (
        ("lr", evaluate_lr, configs["lr"].data.lo_res_size),
        ("sr", evaluate_sr, configs["sr"].data.hi_res_size),
    ):
        config = configs[stage]
        labels = label_sections(real_crops(config, size))
        noise = section_errors(labels[0::2], labels[1::2])
        actual = module.real_summary(labels, noise)
        if stage == "lr":
            source = first_image(config)[1]
            actual |= {"tilt_phase1": evaluate_lr.real_tilt(source, config)}
        expected = json.loads((ASSETS / f"{stage}-results.json").read_text())["real"]
        close(actual, expected, (stage,))


def test_a_full_evaluation_clears_the_smoke_marker(tmp_path):
    run = tmp_path / "run"
    cache = cache_dir(tmp_path, run, smoke=True)
    cache_dir(tmp_path, run, smoke=False)
    (cache / "lr.pt").touch()
    # The full volumes now in the cache open as full volumes again.
    assert cache_dir(tmp_path, run, smoke=False) == cache
    with pytest.raises(SystemExit, match="holds full volumes"):
        cache_dir(tmp_path, run, smoke=True)


def test_reuse_keeps_same_named_runs_separate(tmp_path):
    first = tmp_path / "a" / "run-sr"
    second = tmp_path / "b" / "run-sr"
    first_cache = cache_dir(tmp_path, first, smoke=True) / "lr.pt"
    second_cache = cache_dir(tmp_path, second, smoke=True) / "lr.pt"
    assert cached(first_cache, True, lambda: [1]) == [1]
    assert cached(second_cache, True, lambda: [2]) == [2]
    assert cached(first_cache, True, lambda: [3]) == [1]
