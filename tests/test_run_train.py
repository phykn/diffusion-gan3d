import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.misc import save_mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cpu_entrypoint_saves_one_complete_step(tmp_path: Path) -> None:
    folders = {}
    for axis in (0, 1, 2):
        folder = tmp_path / "slices" / str(axis)
        folder.mkdir(parents=True)
        Image.fromarray(
            np.random.default_rng(axis).integers(
                0,
                3,
                size=(8, 8),
                dtype=np.uint8,
            )
        ).save(folder / "sample.png")
        folders[axis] = str(folder)

    config = tmp_path / "train.yaml"
    run_root = tmp_path / "run"
    save_mapping(
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
            "optim": {
                "generator_lr": 0.001,
                "critic_lr": 0.001,
                "beta1": 0.0,
                "beta2": 0.9,
                "r1_gamma": 0.0,
                "r1_interval": 2,
            },
            "train": {
                "checkpoint": None,
                "steps": 1,
                "volume_batch_size": 1,
                "slices_per_axis": 2,
                "mixed_precision": False,
                "ema_decay": 0.9,
                "save_every_steps": 1,
            },
            "output": {"run_root": str(run_root)},
        },
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "run_train.py"),
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
    checkpoint = run_dirs[0] / "last.pt"
    assert checkpoint.is_file()
    assert (run_dirs[0] / "train.yaml").is_file()
    values = __import__("torch").load(checkpoint, weights_only=True)
    assert values["step"] == 1
