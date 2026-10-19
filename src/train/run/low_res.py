import copy
from pathlib import Path

import torch

from src.build.trainer import build_trainer
from src.config.files import PROJECT_ROOT, load_yaml
from src.config.train import load_train_config
from src.device import check_device
from src.train.run.loop import check_run_dir, make_run_dir, run_train
from src.train.run.resume import check_resume_args, load_resume, set_total_steps
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
    check_resume_args(resume, path_map, config, data)
    device = check_device(device)
    check_run_dir(run_dir)
    payload = None
    if resume:
        payload, cfg, source = load_resume(resume, "low_res", path_map)
    else:
        path = config or PROJECT_ROOT / "config/train/low_res.yaml"
        source = load_yaml(path)
        if data is not None:
            source["data"] = str(data)
        cfg = load_train_config(path, data=data)
    set_total_steps(cfg, source, steps)
    start = 0 if payload is None else payload["step"]
    if cfg["train"]["total_steps"] <= start:
        raise ValueError("total steps must exceed completed steps.")

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
