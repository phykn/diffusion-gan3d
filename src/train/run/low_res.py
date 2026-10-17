import copy
from pathlib import Path

import torch

from src.build.trainer import build_trainer
from src.config.files import PROJECT_ROOT, load_yaml
from src.config.train import (
    load_train_config,
    normalize_train_config,
    validate_training_settings,
)
from src.device import check_device
from src.train.checkpoint import resolve_checkpoint
from src.train.relocate import relocate_checkpoint
from src.train.run.loop import check_run_dir, make_run_dir, run_train, saved_config
from src.train.state import resume_training


def run_low_res_train(
    device: str | torch.device = "cpu",
    resume: Path | None = None,
    config: Path | None = None,
    run_dir: Path | None = None,
    steps: int | None = None,
    data: Path | None = None,
    path_map: list[tuple[str, str]] | None = None,
) -> Path:
    if path_map and resume is None:
        raise ValueError("path_map requires --resume.")
    device = check_device(device)
    check_run_dir(run_dir)
    payload = None
    if resume:
        if config is not None or data is not None:
            raise ValueError(
                "--resume uses saved settings; only --steps may override them."
            )
        checkpoint = resolve_checkpoint(resume)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("format") != "diffusion-gan3d.lr.train":
            raise ValueError("--resume requires an LR training checkpoint.")
        payload = relocate_checkpoint(payload, path_map)
        cfg = normalize_train_config(payload["config"])
        source = saved_config(checkpoint, payload)
    else:
        path = config or PROJECT_ROOT / "config/train/low_res.yaml"
        source = load_yaml(path)
        if data is not None:
            source["data"] = str(data)
        cfg = load_train_config(path, data=data)
    if steps is not None:
        cfg["train"]["total_steps"] = steps
        source.setdefault("train", {})["total_steps"] = steps
    start = 0 if payload is None else payload["step"]
    if cfg["train"]["total_steps"] <= start:
        raise ValueError("total steps must exceed completed steps.")
    validate_training_settings(cfg)

    build_cfg = copy.deepcopy(cfg)
    if payload:
        build_cfg["train"]["initial_weights"] = None
    trainer = build_trainer(build_cfg, device)
    trainer.cfg = cfg
    if payload:
        resume_training(trainer, payload)
    run_dir = make_run_dir(PROJECT_ROOT / "run", "low_res", run_dir, cfg["nickname"])
    run_train(
        trainer,
        steps=cfg["train"]["total_steps"],
        save_every=cfg["train"]["weights_every_steps"],
        run_dir=run_dir,
        checkpoint_every=cfg["train"]["archive_every_steps"],
        start_step=start,
        config=source,
    )
    return run_dir
