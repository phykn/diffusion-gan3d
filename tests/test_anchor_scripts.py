import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tifffile
import torch

from src.build import build_models
from src.config import GenerationSettings
from src.generate import Generator
from src.train.weights import save_weights
from src.utils import save_yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "count",
    (0, 1, 3, 8),
)
def test_anchor_check_script_runs_with_fixed_volume(
    count: int,
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    volume_path = tmp_path / "volume_000.tiff"
    output_path = tmp_path / f"generated_{count}.tiff"
    volume = (np.indices((10, 12, 14)).sum(axis=0) % 3).astype(np.uint8)
    tifffile.imwrite(volume_path, volume)

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
            "--gt",
            str(volume_path),
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
    assert "Shape    : 8 × 8 × 8" in output
    assert f"Anchors  : {count} planes" in output
    assert "Anchor match      :" in output
    assert "Margin  : 8 per outer face" in output
    assert "Complete volume :" in output
    assert "Phase IoU" in output
    assert "Anchor boundary" in output
    assert "Anchor sides" in output
    assert "Ordinary planes" in output
    if count == 0:
        assert "Anchor sides       : n/a" in output
        assert "Distance profile" not in output
    else:
        assert "Distance profile" in output
        assert "Distance  0" in output
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

        class diffusion:
            timesteps = 10

        def generate(
            self,
            *,
            anchors,
            anchor_strength,
            anchor_sigma,
            anchor_residual_blur,
            anchor_residual_blur_early,
            guidance,
            domain,
            margin,
        ):
            calls.append(
                (
                    anchors,
                    anchor_strength,
                    anchor_sigma,
                    anchor_residual_blur,
                    anchor_residual_blur_early,
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
    monkeypatch.setattr(
        module,
        "load_volume",
        lambda *_args, **_kwargs: torch.zeros((4, 4, 4), dtype=torch.uint8),
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
            "--gt",
            str(tmp_path / "gt.tiff"),
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
    assert calls == [((), 0.0, 2.0, 1.5, 2.0, 1.75, 1, 8)]
    assert "Anchors  : 0 planes" in output
    assert "Conditioning : none" in output


@pytest.mark.parametrize(
    ("extra_args", "expected", "domain"),
    (((), 1.0, 0), (("--guidance", "1.5", "--domain", "2"), 1.5, 2)),
)
def test_unconditioned_check_routes_guidance(
    extra_args: tuple[str, ...],
    expected: float,
    domain: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script("02_check_generated.py")
    calls = []
    output_path = tmp_path / "nested" / "generated.tiff"

    class FakeGenerator:
        patch_size = 4
        num_phases = 3

        def generate(self, *, vf, guidance, domain, margin):
            calls.append((vf, guidance, domain, margin))
            return torch.zeros((4, 4, 4), dtype=torch.uint8)

    monkeypatch.setattr(module, "load_generator", lambda _path, device: FakeGenerator())
    monkeypatch.setattr(
        module,
        "load_generation_settings",
        lambda: GenerationSettings(),
    )
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

    assert calls == [(None, expected, domain, 8)]


def test_unconditioned_check_propagates_generator_guidance_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_script("02_check_generated.py")
    generator = Generator.__new__(Generator)
    generator.patch_size = 4

    monkeypatch.setattr(module, "load_generator", lambda _path, device: generator)
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
            "02_check_generated.py",
            "--weight",
            str(tmp_path / "generator.pt"),
            "--guidance",
            "-0.1",
        ],
    )

    with pytest.raises(ValueError, match="guidance must be"):
        module.main()


def test_anchor_boundary_quality_compares_both_sides_with_ordinary_planes() -> None:
    module = _load_script("03_check_anchor.py")
    checker = torch.tensor([[0, 1], [0, 1]], dtype=torch.uint8)
    vol = torch.stack(
        (
            torch.zeros_like(checker),
            checker,
            torch.ones_like(checker),
            torch.zeros_like(checker),
        )
    )

    quality = module.measure_boundaries(vol, (2,), axis=0, num_phases=2)

    assert quality.anchor_change == pytest.approx(0.75)
    assert quality.ordinary_change == pytest.approx(0.5)
    assert quality.change_ratio == pytest.approx(1.5)
    assert quality.transition_tv == pytest.approx(0.75)
    assert quality.continuation_delta == pytest.approx(0.5)


def test_anchor_boundary_quality_is_empty_without_anchors() -> None:
    module = _load_script("03_check_anchor.py")
    vol = torch.zeros((4, 2, 2), dtype=torch.uint8)

    quality = module.measure_boundaries(vol, (), axis=0, num_phases=2)

    assert quality == module.BoundaryQuality(None, None, None, None, None)


def test_anchor_distance_profile_groups_slice_changes_by_nearest_anchor() -> None:
    module = _load_script("03_check_anchor.py")
    checker = torch.tensor([[0, 1], [0, 1]], dtype=torch.uint8)
    vol = torch.stack(
        (
            torch.zeros_like(checker),
            checker,
            torch.ones_like(checker),
            torch.zeros_like(checker),
        )
    )

    profile = module.measure_distance_changes(
        vol,
        (2,),
        axis=0,
        max_distance=2,
    )

    assert profile[0] == pytest.approx(0.75)
    assert profile[1] == pytest.approx(0.5)
    assert profile[2] is None


def test_anchor_divergence_profile_compares_same_distance_slices() -> None:
    module = _load_script("03_check_anchor.py")
    baseline = torch.zeros((5, 2, 2), dtype=torch.uint8)
    anchored = baseline.clone()
    anchored[2] = 1
    anchored[1, 0] = 1

    profile = module.measure_distance_divergence(
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
            "crop_size": 16,
            "input_size": 8,
            "num_phases": 3,
            "augment": False,
            "augment_prob": 1.0,
            "batch_size": 2,
            "num_workers": 0,
        },
        "model": {
            "base_channels": 4,
            "channel_multipliers": (1, 2),
            "embedding_channels": 8,
            "latent_channels": 4,
            "critic_channels": (4, 8),
            "gradient_checkpointing": False,
        },
        "diffusion": {"timesteps": 1, "beta_min": 0.1, "beta_max": 2.0},
        "anchor": {
            "start_step": 0,
            "ramp_steps": 0,
            "training_probability": 1.0,
            "multi_anchor_prob": 0.5,
            "max_density": 0.05,
            "min_spacing": 2,
            "mixed_axis_prob": 0.5,
            "teacher_bank_size_mib": 1,
            "loss_weight": 1.0,
        },
        "conditioning": {
            "cfg_dropout": {
                "drop_each_prob": 0.05,
                "single_condition_drop_prob": 0.1,
            }
        },
        "connectivity": {
            "loss_weight": 0.0,
            "normal_transition_loss_weight": 0.0,
            "replay_triplets_per_axis": 1,
            "replay_capacity_per_axis": 2,
            "max_triplets_per_step": 1,
            "reversal_invariant": True,
        },
        "vf": {
            "loss_weight": 1.0,
        },
        "optim": {
            "denoiser_lr": 1e-3,
            "critic_lr": 1e-3,
            "beta1": 0.0,
            "beta2": 0.9,
            "r1_gamma": 0.0,
            "r1_interval": 2,
            "local_loss_weight": 0.5,
        },
        "train": {
            "total_steps": 1,
            "volume_batch_size": 1,
            "slice_pairs_per_axis": 2,
            "mixed_precision": False,
            "ema_decay": 0.9,
            "save_every_steps": 1,
            "checkpoint_every_steps": 1,
        },
    }
