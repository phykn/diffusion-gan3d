import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tifffile
import torch
from PIL import Image

from src.build import build_models
from src.config import GenerationSettings
from src.evaluate import (
    BoundaryQuality,
    SliceSmoothness,
    measure_boundaries,
    measure_distance_changes,
    measure_distance_divergence,
    measure_slice_smoothness,
)
from src.train.weights import save_weights
from src.utils import save_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_boundary_continuation_defaults_to_an_unconditioned_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _config(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_yaml(run_dir / "train.yaml", cfg)
    model, _, _ = build_models(cfg)
    weights = save_weights(run_dir, model)
    output_path = tmp_path / "continuation.tiff"

    module = _load_script("05_check_continuation.py")
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "05_check_continuation.py",
            "--weight",
            str(weights),
            "--out",
            str(output_path),
            "--no-view",
        ],
    )

    module.main()

    assert tifffile.imread(output_path).shape == (8, 8, 8)
    output = capsys.readouterr().out
    assert "Anchor  : unconditioned reference boundary" in output
    assert "Generating unconditioned reference" in output


def test_boundary_continuation_script_uses_a_real_image_at_the_start_plane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _config(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_yaml(run_dir / "train.yaml", cfg)
    model, _, _ = build_models(cfg)
    weights = save_weights(run_dir, model)
    source = np.indices((18, 20)).sum(axis=0) % cfg["data"]["num_phase"]
    anchor_path = tmp_path / "anchor.png"
    Image.fromarray(source.astype(np.uint8)).save(anchor_path)
    output_path = tmp_path / "continuation.tiff"
    figure_path = tmp_path / "continuation.png"

    module = _load_script("05_check_continuation.py")
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "05_check_continuation.py",
            "--weight",
            str(weights),
            "--anchor",
            str(anchor_path),
            "--out",
            str(output_path),
            "--figure",
            str(figure_path),
            "--no-view",
        ],
    )

    module.main()

    assert tifffile.imread(output_path).shape == (8, 8, 8)
    assert figure_path.is_file()
    output = capsys.readouterr().out
    assert "crop (1, 2, 16, 16) -> input 8 x 8" in output
    assert "Boundary: axis 0, start plane 0" in output
    assert "Anchor match" in output
    assert "First change" in output
    assert "farthest" in output


def test_continuation_anchor_keeps_indexed_labels_and_uses_nearest_resize(
    tmp_path: Path,
) -> None:
    source = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 2, 2, 0],
            [0, 1, 1, 2, 2, 0],
            [0, 2, 2, 0, 0, 0],
            [0, 2, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    image = Image.fromarray(source, mode="P")
    palette = [0] * 768
    palette[:9] = [255, 0, 0, 0, 255, 0, 0, 0, 255]
    image.putpalette(palette)
    path = tmp_path / "indexed-anchor.png"
    image.save(path)

    module = _load_script("05_check_continuation.py")
    labels, crop = module.load_anchor_image(path, 4, 2, 3)

    assert crop == (1, 1, 4, 4)
    assert labels.tolist() == [[1, 2], [2, 0]]


def test_boundary_continuation_napari_shows_generated_volume_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("05_check_continuation.py")
    layers = []
    ran = False

    class FakeViewer:
        def __init__(self) -> None:
            self.dims = SimpleNamespace(ndisplay=2)

        def add_labels(self, values, **kwargs) -> None:
            layers.append((np.asarray(values), kwargs))

    def run() -> None:
        nonlocal ran
        ran = True

    monkeypatch.setitem(
        sys.modules,
        "napari",
        SimpleNamespace(Viewer=FakeViewer, run=run),
    )
    volume = torch.zeros((4, 4, 4), dtype=torch.long)

    module.show_napari(volume)

    assert ran
    assert [layer[1]["name"] for layer in layers] == ["Generated output"]
    assert np.array_equal(layers[0][0], volume.numpy())


@pytest.mark.parametrize(
    "count",
    (0, 1, 3, 8),
)
def test_anchor_check_script_runs_with_generated_reference(
    count: int,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    output_path = tmp_path / f"generated_{count}.tiff"

    cfg = _config(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_yaml(run_dir / "train.yaml", cfg)
    model, _, _ = build_models(cfg)
    with torch.no_grad():
        model.anchor_input.weight.fill_(0.01)
    weights = save_weights(run_dir, model)

    filename = "03_check_anchor.py"
    module = _load_script(filename)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            filename,
            "--weight",
            str(weights),
            "--count",
            str(count),
            "--out",
            str(output_path),
        ],
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    module.main()
    plt.close("all")
    assert output_path.is_file()
    assert tifffile.imread(output_path).shape == (8, 8, 8)

    output = capsys.readouterr().out
    assert "Input   : 8 x 8 x 8, cpu" in output
    assert f"Anchors : {count} planes on axis 0" in output
    assert "Anchor match  :" in output
    assert "Smoothness" in output
    assert "Surface bumps" in output
    assert "Volume match" not in output
    assert "Phase IoU" not in output
    assert "Boundary ratio" not in output
    assert "Distance change" not in output
    assert "Coverage" not in output
    assert "Phase recall" not in output
    if count == 0:
        assert "Anchor effect" not in output
        assert "strength 0.90" not in output
    else:
        assert "Anchor effect" in output
        assert "strength 0.90" in output
    assert "Indices" not in output
    assert "Center slice" not in output


def test_anchor_selection_is_center_first_and_evenly_distributed() -> None:
    module = _load_script("03_check_anchor.py")

    assert module.select_indices(64, 0) == ()
    assert module.select_indices(64, 2) == (16, 48)
    assert module.select_indices(64, 3) == (10, 32, 53)
    assert module.select_indices(64, 4) == (8, 24, 40, 56)
    assert module.select_indices(8, 8) == tuple(range(8))


def test_anchor_graph_uses_nearest_selected_plane() -> None:
    module = _load_script("03_check_anchor.py")

    assert module.select_display_index(64, ()) == 32
    assert module.select_display_index(64, (16, 48)) == 16
    assert module.select_display_index(64, (10, 32, 53)) == 32


def test_anchor_effect_profile_reports_parameter_free_distances() -> None:
    module = _load_script("03_check_anchor.py")

    assert module.format_profile((0.2, 0.4, 0.8)) == (
        "anchor 20.00%, mean 46.67%, farthest 80.00% at distance 2"
    )
    assert module.format_profile((0.2, 0.4, None)) == (
        "anchor 20.00%, mean 30.00%, farthest 40.00% at distance 1"
    )


def test_zero_strength_is_unanchored_for_anchor_disabled_weights(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script("03_check_anchor.py")
    calls = []

    class FakeGenerator:
        patch_size = 4
        num_phases = 3
        default_margin = 8

        class diffusion:
            timesteps = 10

        def generate(
            self,
            *,
            anchors,
            anchor_strength,
            guidance,
            domain,
            margin,
        ):
            calls.append(
                (
                    anchors,
                    anchor_strength,
                    guidance,
                    domain,
                    margin,
                )
            )
            return torch.zeros((4, 4, 4), dtype=torch.uint8)

    monkeypatch.setattr(module, "load_generator", lambda _path, device: FakeGenerator())
    monkeypatch.setattr(
        module,
        "load_generation_settings",
        lambda: GenerationSettings(),
    )
    monkeypatch.setattr(module, "show_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "03_check_anchor.py",
            "--weight",
            str(tmp_path / "generator.pt"),
            "--domain",
            "1",
            "--count",
            "3",
            "--anchor-strength",
            "0",
            "--guidance",
            "1.75",
        ],
    )

    module.main()

    output = capsys.readouterr().out
    assert calls == [
        ((), 0.0, 1.75, 1, 8),
        ((), 0.0, 1.75, 1, 8),
        ((), 0.0, 1.75, 1, 8),
    ]
    assert "Anchors : 0 planes on axis 0" in output
    assert "Guidance" not in output


def test_anchor_check_uses_generated_reference_and_same_rng_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script("03_check_anchor.py")
    reference = torch.arange(64, dtype=torch.uint8).reshape(4, 4, 4) % 3
    calls = []

    class FakeGenerator:
        patch_size = 4
        num_phases = 3
        default_margin = 8

        def generate(self, *, anchors, **_kwargs):
            calls.append((anchors, float(torch.rand(()))))
            return reference.clone() if len(calls) == 1 else torch.zeros_like(reference)

    monkeypatch.setattr(module, "load_generator", lambda _path, device: FakeGenerator())
    monkeypatch.setattr(
        module,
        "load_generation_settings",
        lambda: GenerationSettings(),
    )
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "03_check_anchor.py",
            "--weight",
            str(tmp_path / "generator.pt"),
            "--seed",
            "17",
            "--axis",
            "1",
            "--count",
            "1",
            "--no-view",
        ],
    )

    module.main()

    assert len(calls) == 3
    assert calls[0][0] == ()
    assert len(calls[1][0]) == 1
    anchor = calls[1][0][0]
    assert anchor.axis == 1
    assert anchor.index == 2
    assert torch.equal(anchor.image, reference.movedim(1, 0)[2])
    assert calls[2][0] == ()
    assert calls[1][1] == calls[2][1]


@pytest.mark.parametrize(
    ("extra_args", "domain"),
    (((), 0), (("--domain", "2"), 2)),
)
def test_unconditioned_check_routes_only_domain_condition(
    extra_args: tuple[str, ...],
    domain: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script("02_check_generated.py")
    calls = []
    output_path = tmp_path / "nested" / "generated.tiff"

    class FakeGenerator:
        patch_size = 4
        num_phases = 3
        default_margin = 8

        def generate(self, *, vf, domain, margin):
            calls.append((vf, domain, margin))
            return torch.zeros((4, 4, 4), dtype=torch.uint8)

    monkeypatch.setattr(module, "load_generator", lambda _path, device: FakeGenerator())
    monkeypatch.setattr(module, "show_slices", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "02_check_generated.py",
            "--weight",
            str(tmp_path / "generator.pt"),
            "--out",
            str(output_path),
            *extra_args,
        ],
    )

    module.main()
    assert output_path.is_file()
    assert tifffile.imread(output_path).shape == (4, 4, 4)

    assert calls == [(None, domain, 8)]
    output = capsys.readouterr().out
    assert f"domain {domain}, cpu" in output
    assert "Anchor/VF" not in output


def test_anchor_boundary_quality_compares_both_sides_with_ordinary_planes() -> None:
    checker = torch.tensor([[0, 1], [0, 1]], dtype=torch.uint8)
    vol = torch.stack(
        (
            torch.zeros_like(checker),
            checker,
            torch.ones_like(checker),
            torch.zeros_like(checker),
        )
    )

    quality = measure_boundaries(vol, (2,), axis=0, num_phases=2)

    assert quality.anchor_change == pytest.approx(0.75)
    assert quality.ordinary_change == pytest.approx(0.5)
    assert quality.change_ratio == pytest.approx(1.5)
    assert quality.transition_tv == pytest.approx(0.75)
    assert quality.continuation_delta == pytest.approx(0.5)


def test_anchor_boundary_quality_is_empty_without_anchors() -> None:
    vol = torch.zeros((4, 2, 2), dtype=torch.uint8)

    quality = measure_boundaries(vol, (), axis=0, num_phases=2)

    assert quality == BoundaryQuality(None, None, None, None, None)


def test_slice_smoothness_detects_a_delayed_jump() -> None:
    baseline = torch.zeros((5, 2, 2), dtype=torch.uint8)
    baseline[1:, 0, 0] = 1
    baseline[2:, 0, 1] = 1
    baseline[3:, 1, 0] = 1
    baseline[4:, 1, 1] = 1
    abrupt = torch.zeros_like(baseline)
    abrupt[3:] = 1

    quality = measure_slice_smoothness(abrupt, (1,), axis=0, baseline=baseline)

    assert quality.acceleration_p95 == pytest.approx(1.0)
    assert quality.acceleration_max == pytest.approx(1.0)
    assert quality.baseline_p95 == pytest.approx(0.0)
    assert quality.baseline_max == pytest.approx(0.0)
    assert quality.p95_ratio is None
    assert quality.max_ratio is None
    assert quality.reversal_rate == pytest.approx(0.0)
    assert quality.baseline_reversal_rate == pytest.approx(0.0)
    assert quality.reversal_ratio is None
    assert quality.peak_index == 2
    assert quality.peak_anchor_distance == 1


def test_slice_smoothness_counts_one_slice_bumps() -> None:
    vol = torch.zeros((3, 2, 2), dtype=torch.uint8)
    vol[1, 0, 0] = 1

    quality = measure_slice_smoothness(vol, (), axis=0)

    assert quality.reversal_rate == pytest.approx(0.25)


def test_slice_smoothness_is_empty_for_two_slices() -> None:
    quality = measure_slice_smoothness(
        torch.zeros((2, 3, 3), dtype=torch.uint8),
        (),
        axis=0,
    )

    assert quality == SliceSmoothness(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def test_anchor_distance_profile_groups_slice_changes_by_nearest_anchor() -> None:
    checker = torch.tensor([[0, 1], [0, 1]], dtype=torch.uint8)
    vol = torch.stack(
        (
            torch.zeros_like(checker),
            checker,
            torch.ones_like(checker),
            torch.zeros_like(checker),
        )
    )

    profile = measure_distance_changes(
        vol,
        (2,),
        axis=0,
        max_distance=2,
    )

    assert profile[0] == pytest.approx(0.75)
    assert profile[1] == pytest.approx(0.5)
    assert profile[2] is None


def test_anchor_divergence_profile_compares_same_distance_slices() -> None:
    baseline = torch.zeros((5, 2, 2), dtype=torch.uint8)
    anchored = baseline.clone()
    anchored[2] = 1
    anchored[1, 0] = 1

    profile = measure_distance_divergence(
        anchored,
        baseline,
        (2,),
        axis=0,
        max_distance=3,
    )

    assert profile[0] == pytest.approx(1.0)
    assert profile[1] == pytest.approx(0.25)
    assert profile[2] == pytest.approx(0.0)
    assert profile[3] is None


def _load_script(filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load test script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(root: Path) -> dict:
    return {
        "data": {
            "domains": {0: {axis: (root / str(axis),) for axis in (0, 1, 2)}},
            "num_phase": 3,
            "crop_partial": False,
            "crop_size": 16,
            "input_size": 8,
            "augment": False,
            "augment_prob": 1.0,
            "domain_prob": 1.0,
            "batch_size": 2,
            "num_workers": 0,
        },
        "model": {
            "grad_checkpoint": False,
            "generator": {
                "channels": [4, 8],
                "condition_channels": 8,
                "latent_channels": 4,
            },
            "critic": {
                "channels": [4, 8],
                "local_loss_weight": 0.5,
                "r1_weight": 0.0,
                "r1_interval": 2,
            },
        },
        "diffusion": {"steps": 1, "beta_min": 0.1, "beta_max": 2.0},
        "anchor": {
            "multiscale_input": False,
            "start_step": 0,
            "ramp_steps": 0,
            "train_prob": 1.0,
            "cross_domain_prob": 0.0,
            "pixel_weight": 0.05,
            "connectivity": {
                "volume_count": 1,
                "refresh_every": 500,
                "weight": 0.0,
                "phase_transition_weight": 0.0,
            },
        },
        "vf": {"max_samples": 4, "weight": 1.0},
        "condition_dropout": {"joint_each_prob": 0.05},
        "optim": {
            "generator_lr": 1e-3,
            "critic_lr": 1e-3,
            "adam_betas": [0.0, 0.9],
            "ema_decay": 0.9,
        },
        "train": {
            "init_weights": None,
            "steps": 1,
            "volume_batch_size": 1,
            "pairs_per_axis": 2,
            "amp": False,
            "update_weights_every": 1,
            "archive_every": 1,
        },
    }
