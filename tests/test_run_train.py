import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from run_train import make_run_dir
from src.train.engine import Metrics
from src.train.runner import write_metrics
from src.utils import save_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_run_directory_uses_minute_name_and_numeric_collision_suffix(
    tmp_path: Path,
) -> None:
    with patch("run_train.datetime") as current:
        current.now.return_value.astimezone.return_value.strftime.return_value = (
            "08052314"
        )
        first = make_run_dir(tmp_path)
        second = make_run_dir(tmp_path)

    assert first.name == "08052314"
    assert second.name == "0805231402"


def test_metrics_separate_multi_plane_anchor_quality() -> None:
    writer = Mock()
    metrics = Metrics(
        generator=1.0,
        generator_total=1.2,
        critic=2.0,
        r1=0.0,
        transition=1,
        volume_size=8,
        domain=2,
        critic_axes=(0.5, 0.7, 0.8),
        anchor_planes=3,
        anchor_conflict_rate=0.02,
        anchor_loss=0.2,
        anchor_accuracy=0.95,
        generator_connectivity=0.3,
        critic_connectivity=0.4,
        connectivity_r1=0.05,
        anchor_ramp=0.5,
        connectivity_triplets=3,
        prior_volumes=8,
        prior_mebibytes=16.0,
        prior_ready=True,
        generator_global=0.6,
        generator_local=0.8,
        critic_global=1.2,
        critic_local=1.6,
        vf_loss=0.15,
        vf_active=True,
        target_vfs=(0.5, 0.1, 0.4),
        target_vf_stds=(0.02, 0.01, 0.03),
        soft_vfs=(0.48, 0.12, 0.4),
        hard_vfs=(0.46, 0.14, 0.4),
        hard_vf_mae=0.026,
    )

    write_metrics(writer, 10, metrics)

    tags = {call.args[0] for call in writer.add_scalar.call_args_list}
    assert "conditioning/anchor_planes" in tags
    assert "conditioning/anchor_conflict_rate" in tags
    assert "loss/anchor_3_planes" in tags
    assert "conditioning/anchor_accuracy_3_planes" in tags
    assert "loss/generator_global" in tags
    assert "loss/generator_local_raw" in tags
    assert "loss/critic_global" in tags
    assert "loss/critic_local_raw" in tags
    assert "loss/generator_connectivity" in tags
    assert "loss/critic_connectivity" in tags
    assert "loss/connectivity_r1_raw" in tags
    assert "conditioning/anchor_ramp" in tags
    assert "train/connectivity_triplets" in tags
    assert "train/prior_volumes" in tags
    assert "train/prior_mebibytes" in tags
    assert "train/prior_ready" in tags
    assert "train/prior_updates" in tags
    assert "loss/vf" in tags
    assert "loss/normal_transition" in tags
    assert "loss/anchor_coarse" in tags
    assert "loss/anchor_pixel" in tags
    assert "conditioning/anchor_shared" in tags
    assert "conditioning/state_joint_null_fraction" in tags
    assert "train/volume_size" in tags
    assert "train/domain" in tags
    assert "conditioning/vf_active" in tags
    assert "conditioning/vf_hard_mae" in tags
    assert "conditioning/vf_target_0" in tags
    assert "conditioning/vf_target_std_0" in tags
    assert "conditioning/vf_soft_0" in tags
    assert "conditioning/vf_hard_0" in tags
    writer.add_image.assert_not_called()


@pytest.mark.parametrize("axes", ((0, 1, 2), (0,)))
def test_cpu_entrypoint_saves_complete_anchor_run(
    tmp_path: Path,
    axes: tuple[int, ...],
) -> None:
    folders = {}
    shapes = {
        0: (8, 8),
        1: (4, 8),
        2: (8, 6),
    }
    for axis in axes:
        folder = tmp_path / "slices" / str(axis)
        folder.mkdir(parents=True)
        Image.fromarray(
            np.random.randint(
                0,
                3,
                size=shapes[axis],
                dtype=np.uint8,
            )
        ).save(folder / "sample.png")
        folders[axis] = [str(folder)]

    config = tmp_path / "train.yaml"
    run_root = tmp_path / "run"
    save_yaml(
        config,
        {
            "data": {
                "domains": {0: folders},
                "num_phase": 3,
                "crop_partial": True,
                "crop_size": 8,
                "input_size": 8,
                "augment": "anisotropic",
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
            "diffusion": {
                "steps": 2,
                "beta_min": 0.1,
                "beta_max": 2.0,
            },
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
                    "weight": 0.25,
                    "phase_transition_weight": 0.1,
                },
            },
            "vf": {"max_samples": 4, "weight": 1.0},
            "condition_dropout": {"joint_each_prob": 0.0},
            "optim": {
                "generator_lr": 0.001,
                "critic_lr": 0.001,
                "adam_betas": [0.0, 0.9],
                "ema_decay": 0.9,
            },
            "train": {
                # Step one builds the prior; step two exercises anchor training.
                "init_weights": None,
                "steps": 2,
                "volume_batch_size": 1,
                "pairs_per_axis": 2,
                "amp": False,
                "update_weights_every": 1,
                "archive_every": 1,
            },
        },
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    runner = (
        "import sys; "
        "from pathlib import Path; "
        "import run_train; "
        "run_train.RUN_ROOT = Path(sys.argv.pop(1)); "
        "run_train.main()"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            runner,
            str(run_root),
            "--config",
            str(config),
            "--device",
            "cpu",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    run_dirs = tuple(run_root.iterdir())
    assert len(run_dirs) == 1
    assert len(run_dirs[0].name) == 8
    assert run_dirs[0].name.isdigit()
    weights = run_dirs[0] / "generator.pt"
    assert weights.is_file()
    expected_critics = tuple(f"critic_{axis}.pt" for axis in axes) + ("critic_c.pt",)
    assert (
        tuple(path.name for path in sorted(run_dirs[0].glob("critic_*.pt")))
        == expected_critics
    )
    assert (run_dirs[0] / "train.yaml").is_file()
    checkpoint = run_dirs[0] / "checkpoints" / "step_00000001"
    assert (checkpoint / "generator.pt").is_file()
    assert (
        tuple(path.name for path in sorted(checkpoint.glob("critic_*.pt")))
        == expected_critics
    )
    values = __import__("torch").load(weights, weights_only=True)
    assert values
    assert all(
        isinstance(value, __import__("torch").Tensor) for value in values.values()
    )


def test_dataset_check_script_accepts_one_axis(tmp_path: Path) -> None:
    folder = tmp_path / "axis_0"
    folder.mkdir()
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(folder / "sample.png")
    config = tmp_path / "train.yaml"
    save_yaml(
        config,
        {
            "data": {
                "domains": {0: {0: [str(folder)]}},
                "num_phase": 2,
                "crop_partial": False,
                "crop_size": 8,
                "input_size": 8,
            }
        },
    )
    environment = dict(os.environ)
    environment["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "01_check_dataset.py"),
            "--config",
            str(config),
            "--domain",
            "0",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
