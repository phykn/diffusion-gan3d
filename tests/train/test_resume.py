"""Checkpoints, exact resume, interrupts and run-name collisions (T8)."""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import train
from src.config.io import PROJECT_ROOT
from src.config.schema import update
from src.train.checkpoint import complete_checkpoint, resolve_checkpoint
from src.train.run import resume_stage, train_stage

LOSSES = ("loss/generator", "loss/generator_total", "loss/critic", "loss/vf")


def _config(base, steps, **values):
    config = update(base, "train.total_steps", steps)
    for path, value in {
        "train.weights_every_steps": 5,
        "train.archive_every_steps": 5,
        "train.structure_every_steps": 0,
        **values,
    }.items():
        config = update(config, path, value)
    return config


def _losses(run_dir: Path) -> dict:
    events = EventAccumulator(str(run_dir / "tensorboard"))
    events.Reload()
    return {
        tag: {event.step: event.value for event in events.Scalars(tag)}
        for tag in LOSSES
    }


@pytest.fixture(scope="module")
def ten_steps(tiny_configs, tmp_path_factory):
    root = tmp_path_factory.mktemp("resume")
    return train_stage(_config(tiny_configs["lr"], 10), "cpu", root / "ten-lr")


def test_resume_matches_an_unbroken_run(tiny_configs, ten_steps, tmp_path):
    whole = train_stage(_config(tiny_configs["lr"], 20), "cpu", tmp_path / "whole")
    stage, resumed = resume_stage(
        ten_steps / "checkpoints", "cpu", tmp_path / "resumed", steps=20
    )
    assert stage == "lr"
    expected, first, second = _losses(whole), _losses(ten_steps), _losses(resumed)
    for tag in LOSSES:
        joined = first[tag] | second[tag]
        assert sorted(joined) == list(range(1, 21))
        assert joined == expected[tag], tag


def test_incomplete_steps_are_skipped(ten_steps, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(ten_steps, copy)
    last = copy / "checkpoints" / "step_00000010.pt"
    last.with_suffix(".complete").unlink()
    assert resolve_checkpoint(copy).name == "step_00000005.pt"
    with pytest.raises(ValueError, match="incomplete"):
        resolve_checkpoint(last)
    last.with_suffix(".complete").write_text("1\n", encoding="ascii")
    assert resolve_checkpoint(copy).name == "step_00000005.pt"


def _rewrite(source: Path, target: Path, change) -> Path:
    payload = torch.load(source, map_location="cpu", weights_only=True)
    change(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    complete_checkpoint(target)
    return target


def test_resume_rejects_changed_settings(ten_steps, tmp_path):
    source = resolve_checkpoint(ten_steps)

    def add(payload):
        payload["config"]["train"]["extra"] = 1

    def remove(payload):
        del payload["config"]["train"]["seed"]

    for name, change, message in (
        ("added", add, "train.extra"),
        ("removed", remove, "missing train.seed"),
    ):
        path = _rewrite(source, tmp_path / name / "step_00000010.pt", change)
        with pytest.raises(ValueError, match=message):
            resume_stage(path, "cpu", tmp_path / f"{name}-run", steps=20)
    with pytest.raises(ValueError, match="matches no saved path: /nowhere"):
        resume_stage(
            source, "cpu", tmp_path / "mapped", steps=20, path_map=[("/nowhere", "/x")]
        )
    with pytest.raises(SystemExit):
        train.parse_args(["--resume", str(source)])


def test_resume_rejects_changed_images(tiny_configs, tmp_path):
    folder = tmp_path / "images"
    shutil.copytree(PROJECT_ROOT / "tests/fixtures/tiny/default", folder)
    config = update(
        _config(tiny_configs["lr"], 5),
        "data.images",
        {plane: [str(folder)] for plane in ("xy", "xz", "yz")},
    )
    run = train_stage(config, "cpu", tmp_path / "run")
    image = folder / "sample.png"
    values = __import__("numpy").array(Image.open(image))
    values[0, 0] = 1 - values[0, 0]
    Image.fromarray(values).save(image)
    with pytest.raises(ValueError, match="changed since the checkpoint: xy/sample.png"):
        resume_stage(run, "cpu", tmp_path / "again", steps=10)


def test_path_map_moves_the_image_folders(tiny_configs, ten_steps, tmp_path):
    moved = tmp_path / "moved"
    shutil.copytree(PROJECT_ROOT / "tests/fixtures/tiny", moved)
    stage, run = resume_stage(
        ten_steps,
        "cpu",
        tmp_path / "relocated",
        steps=15,
        path_map=[("tests/fixtures/tiny", str(moved))],
    )
    saved = (run / "train.yaml").read_text()
    assert str(moved / "default") in saved


def test_unfinished_stage_prints_the_resume_command(tiny_sample, ten_steps, tmp_path):
    name = tmp_path / "pending"
    shutil.copytree(ten_steps, name.with_name("pending-lr"))
    (name.with_name("pending-lr") / ".done").unlink()
    with pytest.raises(SystemExit, match="--resume .*pending-lr/checkpoints"):
        train.main([str(tiny_sample), "--run-dir", str(name), "--device", "cpu"])


def test_lr_resume_prints_the_sr_command(ten_steps, tmp_path, capsys):
    name = tmp_path / "next"
    train.main(
        [
            "--resume",
            str(ten_steps),
            "--run-dir",
            str(name),
            "--steps",
            "12",
            "--device",
            "cpu",
        ]
    )
    out = capsys.readouterr().out
    assert f"--run-dir {name}" in out and "(SR)" in out
    assert (name.with_name("next-lr") / ".done").is_file()
    assert not name.with_name("next-sr").exists()


def _start(tiny_sample, run: Path) -> subprocess.Popen:
    process = subprocess.Popen(
        [
            sys.executable,
            "train.py",
            str(tiny_sample),
            "--run-dir",
            str(run),
            "--lr-only",
            "--steps",
            "100000",
            "--device",
            "cpu",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    events = run.with_name(f"{run.name}-lr") / "tensorboard"
    deadline = time.monotonic() + 120
    while not any(events.glob("events.*")) and time.monotonic() < deadline:
        time.sleep(0.2)
    time.sleep(2.0)
    return process


def test_one_interrupt_saves_and_stops(tiny_sample, tmp_path):
    run = tmp_path / "stop"
    process = _start(tiny_sample, run)
    process.send_signal(signal.SIGINT)
    output = process.communicate(timeout=120)[0].decode()
    assert process.returncode == 130, output
    lr = run.with_name("stop-lr")
    assert (lr / "generator.pt").is_file()
    step = resolve_checkpoint(lr)
    assert step.with_suffix(".complete").is_file()
    assert not (lr / ".done").exists()


def test_two_interrupts_stop_at_once(tiny_sample, tmp_path):
    run = tmp_path / "now"
    process = _start(tiny_sample, run)
    process.send_signal(signal.SIGINT)
    process.send_signal(signal.SIGINT)
    started = time.monotonic()
    process.communicate(timeout=60)
    assert process.returncode == 130
    assert time.monotonic() - started < 30
