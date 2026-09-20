import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.config.files import load_yaml, save_yaml
from src.plane import PLANES
from src.train.metrics import Metrics, write_metrics
from src.train.run.loop import make_run_dir, run_train

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("stage", ["low_res", "sr"])
@pytest.mark.parametrize("interrupt", [False, True])
def test_shared_run_records_resumed_steps_and_preserves_save_schedule(
    tmp_path, monkeypatch, stage, interrupt
):
    metrics = Metrics(
        generator=1.0,
        generator_total=1.0,
        critic=2.0,
        r1=0.0,
        transition=1,
        volume_size=8,
        domain=0,
        critic_axes=(2.0, 2.0, 2.0),
        anchor_planes=0,
        anchor_conflict_rate=0.0,
        anchor_loss=0.0,
        anchor_accuracy=0.0,
        generator_connectivity=0.0,
        critic_connectivity=0.0,
        connectivity_r1=0.0,
        anchor_ramp=0.0,
    )
    trainer = SimpleNamespace(cfg={"stage": stage}, completed_steps=1)
    prepared, trained, exports, checkpoints = [], [], [], []

    def step(index):
        assert prepared[-1] == index
        if interrupt and index == 3:
            raise KeyboardInterrupt
        trained.append(index)
        trainer.completed_steps = index + 1
        return metrics

    trainer.step = step
    monkeypatch.setattr("src.train.run.loop.describe_data", lambda _: {"sources": []})
    monkeypatch.setattr(
        "src.train.run.loop.save_weights",
        lambda trainer, path, stage: exports.append(
            (trainer.completed_steps, path.relative_to(tmp_path).as_posix(), stage)
        ),
    )
    monkeypatch.setattr(
        "src.train.run.loop.save_training",
        lambda path, trainer: checkpoints.append(("low_res", trainer.completed_steps)),
    )
    monkeypatch.setattr(
        "src.train.run.loop.save_sr_training",
        lambda trainer, path: checkpoints.append(("sr", trainer.completed_steps)),
    )
    with pytest.raises(KeyboardInterrupt) if interrupt else nullcontext():
        run_train(
            trainer,
            steps=5,
            save_every=3,
            checkpoint_every=2,
            run_dir=tmp_path,
            start_step=1,
            before_step=prepared.append,
        )

    expected_steps = [2, 3] if interrupt else [2, 3, 4, 5]
    records = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert [record["step"] for record in records] == expected_steps
    assert [index + 1 for index in trained] == expected_steps
    assert prepared == ([1, 2, 3] if interrupt else [1, 2, 3, 4])
    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    assert [value.step for value in events.Scalars("loss/generator")] == expected_steps
    assert load_yaml(tmp_path / "train.yaml") == trainer.cfg
    assert json.loads((tmp_path / "data_manifest.json").read_text()) == {"sources": []}
    assert checkpoints == ([(stage, 3)] if interrupt else [(stage, 3), (stage, 5)])
    assert exports == [
        (2, "checkpoints/step_00000002", stage),
        (3, ".", stage),
        *(
            [(3, ".", stage)]
            if interrupt
            else [
                (4, "checkpoints/step_00000004", stage),
                (5, ".", stage),
            ]
        ),
    ]


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


@pytest.mark.parametrize("stage", ["low_res", "sr"])
@pytest.mark.parametrize("nickname", ["", "실험_A"])
def test_run_directory_preserves_stage_and_nickname_on_collision(
    tmp_path: Path,
    stage: str,
    nickname: str,
) -> None:
    with patch("src.train.run.loop.datetime") as current:
        current.now.return_value.astimezone.return_value.strftime.return_value = (
            "08052314"
        )
        first = make_run_dir(tmp_path, stage, nickname=nickname)
        marker = first / "keep.txt"
        marker.write_text("existing run")
        second = make_run_dir(tmp_path, stage, nickname=nickname)

    label = f"_{nickname}" if nickname else ""
    assert first.name == f"08052314_{stage}{label}"
    assert second.name == f"08052314_02_{stage}{label}"
    assert marker.read_text() == "existing run"


def test_explicit_run_directory_is_preserved_and_never_reused(tmp_path):
    target = tmp_path / "experiment"
    assert (
        make_run_dir(tmp_path / "unused", "sr", target, "ignored") == target.resolve()
    )
    with pytest.raises(FileExistsError):
        make_run_dir(tmp_path / "unused", "sr", target)


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
            "nickname": "coarse",
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
        "import src.train.run.low_res; "
        "src.train.run.low_res.PROJECT_ROOT = Path(sys.argv.pop(1)); "
        "run_train_1st.main()"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            runner,
            str(tmp_path),
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
    timestamp, stage = run_dirs[0].name.split("_", 1)
    assert len(timestamp) == 8 and timestamp.isdigit()
    assert stage == "low_res_coarse"
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
