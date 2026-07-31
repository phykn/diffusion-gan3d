import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image

from src.train.engine import Metrics, Trainer
from src.utils import load_yaml, save_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_metrics_separate_multi_plane_anchor_quality() -> None:
    writer = Mock()
    metrics = Metrics(
        generator=1.0,
        generator_total=1.2,
        critic=2.0,
        r1=0.0,
        transition=1,
        critic_axes=(0.5, 0.7, 0.8),
        anchor_planes=3,
        anchor_conflict_rate=0.02,
        anchor_loss=0.2,
        anchor_accuracy=0.95,
        generator_global=0.6,
        generator_local=0.8,
        critic_global=1.2,
        critic_local=1.6,
    )

    Trainer._write_metrics(writer, 10, metrics)

    tags = {call.args[0] for call in writer.add_scalar.call_args_list}
    assert "conditioning/anchor_planes" in tags
    assert "conditioning/anchor_conflict_rate" in tags
    assert "loss/anchor_3_planes" in tags
    assert "conditioning/anchor_accuracy_3_planes" in tags
    assert "loss/generator_global" in tags
    assert "loss/generator_local_raw" in tags
    assert "loss/critic_global" in tags
    assert "loss/critic_local_raw" in tags


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
        folders[axis] = str(folder)

    config = tmp_path / "train.yaml"
    run_root = tmp_path / "run"
    save_yaml(
        config,
        {
            "data": {
                "folder": folders,
                "crop_size": 8,
                "patch_size": 8,
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
                "probability": 1.0,
                "loss_weight": 1.0,
                "max_planes": 3,
            },
            "optim": {
                "generator_lr": 0.001,
                "critic_lr": 0.001,
                "beta1": 0.0,
                "beta2": 0.9,
                "r1_gamma": 0.0,
                "r1_interval": 2,
            },
            "train": {
                "steps": 1,
                "volume_batch_size": 1,
                "slices_per_axis": 2,
                "mixed_precision": False,
                "ema_decay": 0.9,
                "save_every_steps": 1,
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
    weights = run_dirs[0] / "model.pt"
    assert weights.is_file()
    assert tuple(path.name for path in sorted(run_dirs[0].glob("critic_*.pt"))) == (
        "critic_0.pt",
        "critic_1.pt",
        "critic_2.pt",
    )
    assert (run_dirs[0] / "train.yaml").is_file()
    values = __import__("torch").load(weights, weights_only=True)
    assert values
    assert all(
        isinstance(value, __import__("torch").Tensor) for value in values.values()
    )

    values = load_yaml(config, label="test config")
    values["train"]["checkpoint"] = {
        "model": str(run_dirs[0] / "model.pt"),
        "critic_0": str(run_dirs[0] / "critic_0.pt"),
        "critic_1": str(run_dirs[0] / "critic_1.pt"),
        "critic_2": str(run_dirs[0] / "critic_2.pt"),
    }
    save_yaml(config, values)
    resumed = subprocess.run(
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

    assert resumed.returncode == 0, resumed.stderr
    assert len(tuple(run_root.iterdir())) == 2
