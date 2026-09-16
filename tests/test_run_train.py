import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from run_train_1st import make_run_dir
from src.config import save_yaml
from src.plane import PLANES
from src.train.run import run_train, write_metrics
from src.train.trainer import Metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("steps", 0),
        ("steps", -1),
        ("save_every", 0),
        ("save_every", -1),
        ("checkpoint_every", 0),
        ("checkpoint_every", -1),
    ],
)
def test_invalid_schedule_fails_before_training_or_creating_logs(
    tmp_path, field, value
):
    trainer = Mock(critics={})
    schedule = {"steps": 1, "save_every": 1, "checkpoint_every": None}
    schedule[field] = value
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        run_train(trainer, run_dir=tmp_path, **schedule)
    trainer.step.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_run_directory_uses_minute_name_and_numeric_collision_suffix(
    tmp_path: Path,
) -> None:
    with patch("run_train_1st.datetime") as current:
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
        generator_global=0.6,
        generator_local=0.8,
        critic_global=1.2,
        critic_local=1.6,
        vf_loss=0.15,
        vf_active=True,
    )

    write_metrics(writer, 10, metrics)

    tags = {call.args[0] for call in writer.add_scalar.call_args_list}
    assert tags == {
        "loss/generator",
        "loss/generator_total",
        "loss/critic",
        "loss/r1",
        "loss/generator_connectivity",
        "loss/critic_connectivity",
        "loss/connectivity_r1",
        "loss/normal_transition",
        "loss/anchor",
        "loss/vf",
        "conditioning/anchor_fraction",
        "conditioning/vf_fraction",
        "conditioning/anchor_ramp",
        "conditioning/connectivity_ramp",
        "conditioning/anchor_planes",
        "conditioning/anchor_accuracy",
        "sampling/transition",
        "timestep/1/generator",
        "timestep/1/critic",
        "critic_plane/xy",
        "critic_plane/xz",
        "critic_plane/yz",
    }
    writer.add_image.assert_not_called()


@pytest.mark.parametrize("axes", ((0, 1, 2), (0,)))
def test_cpu_entrypoint_saves_complete_anchor_run(
    tmp_path: Path,
    axes: tuple[int, ...],
) -> None:
    folders = {}
    shapes = {
        0: (8, 8),
        1: (8, 8),
        2: (8, 8),
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
        folders[PLANES[axis]] = [str(folder)]

    config = tmp_path / "train.yaml"
    run_root = tmp_path / "run"
    save_yaml(
        config,
        {
            "data": {
                "domains": {0: folders},
                "crop_size": 8,
                "num_phases": 3,
                "lo_res_size": 8,
            },
            "model": {
                "generator": {
                    "channels": [4, 8],
                    "latent_channels": 4,
                    "embedding_channels": 8,
                    "anchor_multiscale_input": False,
                },
                "critic": {
                    "channels": [4, 8],
                    "plane_groups": [
                        [plane]
                        for plane in ("xy", "xz", "yz")
                        if any((plane in planes for planes in {0: folders}.values()))
                    ],
                },
                "gradient_checkpointing": False,
                "diffusion": {"num_steps": 2, "beta_min": 0.1, "beta_max": 2.0},
            },
            "optim": {
                "generator_lr": 0.001,
                "critic_lr": 0.001,
                "adam_betas": [0.0, 0.9],
                "ema_decay": 0.9,
            },
            "train": {
                "volume_batch_size": 1,
                "real_batch_size": 2,
                "num_workers": 0,
                "total_steps": 2,
                "mixed_precision": False,
                "slice_pairs_per_plane": 2,
                "initial_weights": None,
                "weights_every_steps": 1,
                "archive_every_steps": 1,
            },
            "conditioning": {
                "domain_keep_probability": 1.0,
                "anchor": {
                    "probability": 1.0,
                    "start_step": 0,
                    "ramp_steps": 0,
                    "borrowed_plane_probability": 0.0,
                },
                "dropout_probability_per_case": 0.0,
            },
            "augmentation": {
                "probability": 1.0,
                "planes": {
                    "xy": {"flip_axes": ["x"], "rotate_90": False},
                    "xz": {"flip_axes": ["x"], "rotate_90": False},
                    "yz": {"flip_axes": ["y"], "rotate_90": False},
                },
            },
            "loss": {
                "anchor_pixel_weight": 0.05,
                "connectivity": {
                    "adversarial_weight": 0.25,
                    "normal_transition_weight": 0.1,
                },
                "volume_fraction_weight": 1.0,
                "critic_local_weight": 0.5,
                "r1_weight": 0.0,
                "r1_every_steps": 2,
            },
        },
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    runner = (
        "import sys; "
        "from pathlib import Path; "
        "import run_train_1st; "
        "run_train_1st.RUN_ROOT = Path(sys.argv.pop(1)); "
        "run_train_1st.main()"
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
    expected_critics = ("critic_c.pt",) + tuple(
        f"critic_{PLANES[axis]}.pt" for axis in axes
    )
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
                "domains": {0: {"xy": [str(folder)]}},
                "crop_size": 8,
                "num_phases": 2,
                "lo_res_size": 8,
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
