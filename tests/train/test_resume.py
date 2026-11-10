"""Checkpoints, exact resume, interrupts and run-name collisions."""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
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
        "train.weights_every_steps": 1,
        "train.archive_every_steps": 1,
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


def test_resume_matches_an_unbroken_run(tiny_configs, tmp_path):
    whole = train_stage(_config(tiny_configs["lr"], 2), "cpu", tmp_path / "whole")
    first = train_stage(_config(tiny_configs["lr"], 1), "cpu", tmp_path / "first")
    stage, second = resume_stage(first, "cpu", tmp_path / "second", steps=2)
    assert stage == "lr"
    expected, before, after = _losses(whole), _losses(first), _losses(second)
    for tag in LOSSES:
        assert before[tag] | after[tag] == expected[tag], tag


def test_incomplete_steps_are_skipped(tiny_lr_run, tmp_path):
    copy = Path(shutil.copytree(tiny_lr_run, tmp_path / "copy"))
    last = resolve_checkpoint(copy)
    last.with_suffix(".complete").unlink()
    with pytest.raises(FileNotFoundError, match="no completed"):
        resolve_checkpoint(copy)
    with pytest.raises(ValueError, match="incomplete"):
        resolve_checkpoint(last)


def _rewrite(source: Path, target: Path, change) -> Path:
    payload = torch.load(source, map_location="cpu", weights_only=True)
    change(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, target)
    complete_checkpoint(target)
    return target


def test_resume_rejects_changed_settings_and_images(tiny_lr_run, tmp_path):
    source = resolve_checkpoint(tiny_lr_run)

    def add(payload):
        payload["config"]["train"]["extra"] = 1

    def remove(payload):
        del payload["config"]["train"]["seed"]

    def image(payload):
        payload["data_fingerprint"]["xy"]["sample.png"] = "0" * 64

    for name, change, message in (
        ("added", add, "train.extra"),
        ("removed", remove, "missing train.seed"),
        ("image", image, "changed since the checkpoint: xy/sample.png"),
    ):
        path = _rewrite(source, tmp_path / name / source.name, change)
        with pytest.raises(ValueError, match=message):
            resume_stage(path, "cpu", tmp_path / f"{name}-run", steps=20)
    with pytest.raises(ValueError, match="matches no saved path: /nowhere"):
        resume_stage(
            source, "cpu", tmp_path / "mapped", steps=20, path_map=[("/nowhere", "/x")]
        )
    with pytest.raises(SystemExit):
        train.parse_args(["--resume", str(source)])


def test_path_map_moves_the_image_folders(tiny_lr_run, tmp_path):
    moved = tmp_path / "moved"
    shutil.copytree(PROJECT_ROOT / "tests/fixtures/tiny", moved)
    _, run = resume_stage(
        tiny_lr_run,
        "cpu",
        tmp_path / "relocated",
        steps=2,
        path_map=[("tests/fixtures/tiny", str(moved))],
    )
    assert str(moved / "default") in (run / "train.yaml").read_text()


def test_unfinished_stage_prints_the_resume_command(tiny_sample, tiny_lr_run, tmp_path):
    name = tmp_path / "pending"
    shutil.copytree(tiny_lr_run, name.with_name("pending-lr"))
    (name.with_name("pending-lr") / ".done").unlink()
    with pytest.raises(SystemExit, match="--resume .*pending-lr/checkpoints"):
        train.main([str(tiny_sample), "--run-dir", str(name), "--device", "cpu"])


def test_lr_resume_prints_the_sr_command(tiny_lr_run, tmp_path, capsys):
    name = tmp_path / "next"
    train.main(
        [
            "--resume",
            str(tiny_lr_run),
            "--run-dir",
            str(name),
            "--steps",
            "2",
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
