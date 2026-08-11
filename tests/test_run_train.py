import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
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
        connectivity_replay=12,
        anchor_teacher=True,
        teacher_volumes=8,
        teacher_mebibytes=16.0,
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
    assert "train/connectivity_replay" in tags
    assert "train/teacher_volumes" in tags
    assert "train/teacher_mebibytes" in tags
    assert "conditioning/anchor_teacher" in tags
    assert "loss/vf" in tags
    assert "loss/normal_transition" in tags
    assert "conditioning/state_joint_null_fraction" in tags
    assert "train/volume_size" in tags
    assert "conditioning/vf_active" in tags
    assert "conditioning/vf_hard_mae" in tags
    assert "conditioning/vf_target_0" in tags
    assert "conditioning/vf_target_std_0" in tags
    assert "conditioning/vf_soft_0" in tags
    assert "conditioning/vf_hard_0" in tags


def test_cpu_entrypoint_saves_one_complete_step(tmp_path: Path) -> None:
    folders = {}
    for axis in (0, 1, 2):
        folder = tmp_path / "slices" / str(axis)
        folder.mkdir(parents=True)
        Image.fromarray(
            np.random.randint(
                0,
                3,
                size=(8, 8),
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
                "folders": folders,
                "crop_size": 8,
                "input_size": 8,
                "augment": "directional",
                "augment_prob": 1.0,
                "num_phases": 3,
                "batch_size": 2,
                "num_workers": 0,
            },
            "model": {
                "base_channels": 4,
                "channel_multipliers": [1, 2],
                "embedding_channels": 8,
                "latent_channels": 4,
                "critic_channels": [4, 8],
                "gradient_checkpointing": False,
            },
            "diffusion": {
                "timesteps": 2,
                "beta_min": 0.1,
                "beta_max": 2.0,
            },
            "anchor": {
                "training_probability": 1.0,
                "start_step": 0,
                "ramp_steps": 0,
                "multi_anchor_prob": 0.5,
                "max_density": 0.05,
                "min_spacing": 2,
                "mixed_axis_prob": 0.5,
                "teacher_bank_size_mib": 1,
                "loss_weight": 1.0,
            },
            "conditioning": {
                "cfg_dropout": {
                    "drop_each_prob": 0.0,
                    "single_condition_drop_prob": 0.0,
                }
            },
            "connectivity": {
                "loss_weight": 0.25,
                "normal_transition_loss_weight": 0.1,
                "replay_triplets_per_axis": 1,
                "replay_capacity_per_axis": 2,
                "max_triplets_per_step": 1,
                "reversal_invariant": True,
            },
            "vf": {
                "loss_weight": 1.0,
            },
            "optim": {
                "denoiser_lr": 0.001,
                "critic_lr": 0.001,
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
    assert tuple(path.name for path in sorted(run_dirs[0].glob("critic_*.pt"))) == (
        "critic_0.pt",
        "critic_1.pt",
        "critic_2.pt",
        "critic_c.pt",
    )
    assert (run_dirs[0] / "train.yaml").is_file()
    checkpoint = run_dirs[0] / "checkpoints" / "step_00000001"
    assert (checkpoint / "generator.pt").is_file()
    assert tuple(path.name for path in sorted(checkpoint.glob("critic_*.pt"))) == (
        "critic_0.pt",
        "critic_1.pt",
        "critic_2.pt",
        "critic_c.pt",
    )
    values = __import__("torch").load(weights, weights_only=True)
    assert values
    assert all(
        isinstance(value, __import__("torch").Tensor) for value in values.values()
    )
