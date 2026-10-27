import copy
from collections.abc import Mapping
from pathlib import Path

import torch

from src.build.trainer import build_trainer
from src.config.files import PROJECT_ROOT
from src.config.train import load_train_config
from src.device import check_device
from src.train.checkpoint import resolve_checkpoint
from src.train.run.loop import check_run_dir, make_run_dir, run_train
from src.train.run.resume import (
    check_resume_args,
    load_resume,
    load_source,
    set_total_steps,
)
from src.train.state import resume_training


def run_low_res_train(
    device: str | torch.device = "cpu",
    resume: Path | None = None,
    config: Path | Mapping | None = None,
    run_dir: Path | None = None,
    steps: int | None = None,
    path_map: list[tuple[str, str]] | None = None,
) -> Path:
    check_resume_args(resume, path_map, config)
    device = check_device(device)
    check_run_dir(run_dir)
    payload = None
    if resume:
        payload, cfg, source = load_resume(resume, "low_res", path_map)
    else:
        source = load_source(config)
        cfg = load_train_config(source)
        # Prediction reads train.yaml; keep the conditions measured from the
        # images so it needs neither the images nor their current contents.
        for key in ("height_extents", "depth_profiles"):
            if key in cfg["data"]:
                source["data"][key] = cfg["data"][key]
    set_total_steps(cfg, source, steps, payload)
    build_cfg = copy.deepcopy(cfg)
    if payload:
        build_cfg["train"]["initial_weights"] = None
    trainer = build_trainer(build_cfg, device)
    trainer.cfg = cfg
    if payload:
        resume_training(trainer, payload)
    run_dir = make_run_dir(PROJECT_ROOT / "run", "low_res", run_dir, cfg["nickname"])
    if payload and trainer.selection is not None:
        # The best step's weights sit beside the resumed checkpoint's run folder.
        source_run = resolve_checkpoint(resume).parent.parent
        trainer.selection.import_best(source_run, run_dir)
    run_train(
        trainer,
        steps=cfg["train"]["total_steps"],
        save_every=cfg["train"]["weights_every_steps"],
        run_dir=run_dir,
        checkpoint_every=cfg["train"]["archive_every_steps"],
        start_step=trainer.completed_steps,
        config=source,
    )
    return run_dir
