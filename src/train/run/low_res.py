import copy
from pathlib import Path

import torch

from src.build.trainer import build_trainer
from src.config.files import PROJECT_ROOT
from src.config.train import load_train_config, normalize_train_config
from src.train.run.loop import make_run_dir, run_train
from src.train.state import resume_training


def run_low_res_train(
    device: str | torch.device = "cpu",
    resume: Path | None = None,
    config: Path | None = None,
    run_dir: Path | None = None,
    steps: int | None = None,
    data: Path | None = None,
) -> Path:
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    payload = None
    if resume:
        if config is not None or data is not None:
            raise ValueError(
                "--resume uses saved settings; only --steps may override them."
            )
        payload = torch.load(resume, map_location="cpu", weights_only=True)
        if payload.get("format") != "diffusion-gan3d.lr.train":
            raise ValueError("--resume requires an LR training checkpoint.")
        cfg = normalize_train_config(payload["config"])
    else:
        cfg = load_train_config(
            config or PROJECT_ROOT / "config/train/low_res.yaml", data=data
        )
    if steps is not None:
        cfg["train"]["total_steps"] = steps
    start = 0 if payload is None else payload["step"]
    if cfg["train"]["total_steps"] <= start:
        raise ValueError("total steps must exceed completed steps.")

    build_cfg = copy.deepcopy(cfg)
    if payload:
        build_cfg["train"]["initial_weights"] = None
    trainer = build_trainer(build_cfg, device)
    cfg["data"] = trainer.cfg["data"]
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
    )
    return run_dir
