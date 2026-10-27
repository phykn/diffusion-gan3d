import os
import signal
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import src.train.run.low_res as low_res
from src.config.files import load_yaml, save_yaml
from src.plane import PLANES
from src.train.metrics import Metrics, write_metrics
from src.train.run.loop import make_run_dir, run_train
from src.train.run.low_res import run_low_res_train

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
        domain=0,
        critic_axes=(2.0, 2.0, 2.0),
    )
    trainer = SimpleNamespace(cfg={"stage": stage}, completed_steps=1, selection=None)
    prepared, trained, exports, checkpoints = [], [], [], []

    def step(index):
        assert prepared[-1] == index
        if interrupt and index == 3:
            signal.raise_signal(signal.SIGINT)
        trained.append(index)
        trainer.completed_steps = index + 1
        return metrics

    trainer.step = step
    monkeypatch.setattr(
        "src.train.run.loop.save_weights",
        lambda trainer, path: exports.append(
            (trainer.completed_steps, path.relative_to(tmp_path).as_posix(), stage)
        ),
    )
    monkeypatch.setattr(
        "src.train.run.loop.save_checkpoint",
        lambda trainer, path: checkpoints.append((stage, trainer.completed_steps)),
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

    expected_steps = [2, 3, 4] if interrupt else [2, 3, 4, 5]
    assert not (tmp_path / "metrics.jsonl").exists()
    assert not (tmp_path / "data_manifest.json").exists()
    assert [index + 1 for index in trained] == expected_steps
    assert prepared == ([1, 2, 3] if interrupt else [1, 2, 3, 4])
    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    assert [value.step for value in events.Scalars("loss/generator")] == expected_steps
    assert load_yaml(tmp_path / "train.yaml") == trainer.cfg
    assert checkpoints == [
        (stage, step) for step in ([2, 4] if interrupt else [2, 4, 5])
    ]
    assert exports == [
        (3, ".", stage),
        (4 if interrupt else 5, ".", stage),
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


def test_metrics_write_loss_conditioning_and_plane_tags() -> None:
    writer = Mock()
    metrics = Metrics(
        generator=1.0,
        generator_total=1.2,
        critic=2.0,
        r1=0.0,
        transition=1,
        domain=2,
        critic_axes=(0.5, 0.7, 0.8),
        real_transition_ramp=0.5,
        vf_loss=0.15,
        vf_active=True,
    )

    write_metrics(writer, 10, metrics)

    tags = {call.args[0] for call in writer.add_scalar.call_args_list}
    assert tags >= {
        "loss/generator",
        "loss/generator_total",
        "loss/critic",
        "loss/r1",
        "loss/vf",
        "conditioning/vf_fraction",
        "loss/real_transition_ramp",
        "sampling/transition",
        "timestep/1/generator",
        "timestep/1/critic",
        "critic_plane/xy",
        "critic_plane/xz",
        "critic_plane/yz",
    }
    writer.add_image.assert_not_called()


@pytest.mark.parametrize("axes", ((0, 1, 2), (0,)))
def test_cpu_entrypoint_saves_complete_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
                "cfg_dropout_probability": 0.0,
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
                "real_transition": {"weight": 0.25, "ramp_steps": 0},
                "volume_fraction_weight": 1.0,
                "r1_weight": 0.0,
                "r1_every_steps": 2,
            },
        },
    )
    monkeypatch.setattr(low_res, "PROJECT_ROOT", tmp_path)
    low_res.run_low_res_train("cpu", config=config)

    run_dirs = tuple(run_root.iterdir())
    assert len(run_dirs) == 1
    timestamp, stage = run_dirs[0].name.split("_", 1)
    assert len(timestamp) == 8 and timestamp.isdigit()
    assert stage == "low_res_coarse"
    weights = run_dirs[0] / "generator.pt"
    assert weights.is_file()
    expected_critics = tuple(f"critic_{PLANES[axis]}.pt" for axis in axes)
    assert (
        tuple(path.name for path in sorted(run_dirs[0].glob("critic_*.pt")))
        == expected_critics
    )
    assert load_yaml(run_dirs[0] / "train.yaml") == load_yaml(config)
    checkpoints = sorted((run_dirs[0] / "checkpoints").glob("step_*.pt"))
    assert len(checkpoints) == 2
    for step, checkpoint in enumerate(checkpoints, start=1):
        payload = torch.load(checkpoint, weights_only=True)
        assert payload["step"] == step
        assert payload["format"] == "diffusion-gan3d.lr.train"
        assert payload["generator_optim"]["state"]
        assert checkpoint.name == f"step_{step:08d}.pt"
        assert checkpoint.with_suffix(".complete").is_file()
        exported = torch.load(
            checkpoint.with_name(f"generator_{step:08d}.pt"), weights_only=True
        )
        torch.testing.assert_close(exported, payload["ema"])
    assert not list((run_dirs[0] / "checkpoints").glob("*/generator.pt"))
    values = torch.load(weights, weights_only=True)
    assert values
    assert all(isinstance(value, torch.Tensor) for value in values.values())


def test_lr_run_saves_the_supplied_yaml_and_step_override_only(
    tmp_path: Path, monkeypatch
) -> None:
    folder = tmp_path / "slices"
    folder.mkdir()
    for index in range(2):
        Image.fromarray(
            ((np.indices((8, 8)).sum(0) + index) % 3).astype(np.uint8)
        ).save(folder / f"{index}.png")
    source = load_yaml(PROJECT_ROOT / "tests/fixtures/config/train/low_res.yaml")
    source["data"] = {
        "domains": {0: {plane: [str(folder)] for plane in PLANES}},
        "crop_size": 8,
        "num_phases": 3,
        "lo_res_size": 8,
    }
    source["model"]["generator"].update(
        channels=[4, 8], embedding_channels=8, latent_channels=4
    )
    source["model"]["critic"]["channels"] = [4, 8]
    source["model"]["gradient_checkpointing"] = False
    source["model"]["diffusion"]["num_steps"] = 2
    source["train"].update(
        real_batch_size=2,
        slice_pairs_per_plane=2,
        mixed_precision=False,
        weights_every_steps=1,
        archive_every_steps=1,
    )
    config = tmp_path / "low_res.yaml"
    save_yaml(config, source)
    monkeypatch.setattr("src.train.run.low_res.PROJECT_ROOT", tmp_path)
    first = run_low_res_train("cpu", config=config, run_dir=tmp_path / "first", steps=1)
    expected = dict(source)
    expected["train"] = {**source["train"], "total_steps": 1}
    assert load_yaml(first / "train.yaml") == expected
    resumed = run_low_res_train(
        "cpu", resume=first / "checkpoints", run_dir=tmp_path / "resumed", steps=2
    )
    payload = torch.load(
        resumed / "checkpoints" / "step_00000002.pt", weights_only=True
    )
    # A resume saves the resolved checkpoint settings.
    assert load_yaml(resumed / "train.yaml") == payload["config"]
    assert payload["config"]["data"]["domains"] == {
        0: {plane: [str(folder.resolve())] for plane in PLANES}
    }


def test_dataset_check_script_accepts_one_axis(tmp_path: Path) -> None:
    folder = tmp_path / "axis_0"
    folder.mkdir()
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(folder / "sample.png")
    sample = tmp_path / "one_axis.yaml"
    save_yaml(
        sample,
        {
            "images": {"xy": str(folder)},
            "structure": {
                "isotropic": True,
                "depth_gradient": False,
                "mirror_symmetric": True,
            },
            "num_phases": 2,
            "resolution": {"crop_size": 8, "lo_res_size": 8, "sr_scale": 2},
        },
    )
    environment = dict(os.environ)
    environment["MPLBACKEND"] = "Agg"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "01_check_dataset.py"),
            "--sample",
            str(sample),
            "--domain",
            "0",
            "--out",
            str(tmp_path / "crops.png"),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_second_interrupt_stops_the_step_at_once(capsys):
    from src.train.run.loop import deferred_sigint

    with deferred_sigint() as stop:
        signal.raise_signal(signal.SIGINT)
        assert stop.is_set()
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)
    assert "press Ctrl+C again" in capsys.readouterr().out


def test_interrupt_while_saving_stops_without_saving_again(tmp_path, monkeypatch):
    trainer = SimpleNamespace(
        cfg={"stage": "low_res"}, completed_steps=0, selection=None
    )

    def step(index):
        trainer.completed_steps = index + 1

    trainer.step = step
    saves = []

    def interrupted_save(trainer, root):
        saves.append(trainer.completed_steps)
        raise KeyboardInterrupt

    monkeypatch.setattr("src.train.run.loop.write_metrics", lambda *args: None)
    monkeypatch.setattr("src.train.run.loop.save_checkpoint", interrupted_save)
    with pytest.raises(KeyboardInterrupt):
        run_train(trainer, steps=1, save_every=1, run_dir=tmp_path)
    assert saves == [1]
