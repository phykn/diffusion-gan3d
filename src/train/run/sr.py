from collections.abc import Mapping
from pathlib import Path

import torch

from src.build.data import build_augmentation
from src.build.predict import resolve_weights
from src.build.sr import build_sr_trainer
from src.config.data import describe_resolution, validate_sr_source
from src.config.files import PROJECT_ROOT, find_saved_config, project_path
from src.config.train import get_sr_resolution, load_train_config, validate_sr_config
from src.console import field, path_field, section
from src.data.bank import load_bank
from src.data.source import fill_height_extents, infer_height_extents
from src.device import check_device
from src.storage import file_hash
from src.train.checkpoint import resolve_checkpoint
from src.train.run.bank import (
    create_bank,
    publish_bank,
    refresh_bank,
)
from src.train.run.loop import check_run_dir, make_run_dir, run_train
from src.train.run.resume import (
    check_resume_args,
    load_resume,
    load_source,
    set_total_steps,
)
from src.train.run.source import load_frozen_source, read_source_config
from src.train.state import resume_training


def run_sr_train(
    device: str | torch.device = "cpu",
    base_weights: Path | None = None,
    resume: Path | None = None,
    config: Path | Mapping | None = None,
    run_dir: Path | None = None,
    steps: int | None = None,
    bank_size: int | None = None,
    path_map: list[tuple[str, str]] | None = None,
) -> Path:
    check_resume_args(resume, path_map, config)
    if base_weights is not None and resume is not None:
        raise ValueError("base_weights cannot change on resume.")
    device = check_device(device)
    check_run_dir(run_dir)
    payload = None
    if resume:
        if bank_size is not None:
            raise ValueError("--bank-size cannot change on resume.")
        payload, cfg, source = load_resume(resume, "sr", path_map)
        bank_path = Path(cfg["source"]["bank"])
        if file_hash(bank_path) != cfg["source"]["bank_sha256"]:
            raise ValueError("LR bank changed since the checkpoint was saved.")
        load_frozen_source(cfg["source"], payload["path_maps"])
        bank_payload = load_bank(bank_path)
    else:
        source = load_source(config)
        cfg = load_train_config(source, "sr")
        prepare_source(cfg, base_weights)
        if base_weights is not None:
            source.setdefault("source", {})["weights"] = str(base_weights)
        if bank_size is not None:
            cfg["lr_bank"]["samples_per_domain"] = bank_size
            source.setdefault("lr_bank", {})["samples_per_domain"] = bank_size
    set_total_steps(cfg, source, steps, payload)
    cfg = validate_sr_config(cfg)
    build_augmentation(cfg)
    run_dir = make_run_dir(PROJECT_ROOT / "run", "sr", run_dir, cfg["nickname"])
    if payload is None:
        bank_payload = create_bank(cfg, device)
        publish_bank(bank_payload, cfg, run_dir, 0)
    trainer = build_sr_trainer(
        cfg,
        bank_payload["volumes"],
        device,
        bank_payload.get("height_origins"),
        bank_payload.get("height_extents"),
    )
    if payload:
        resume_training(trainer, payload)
    cfg = trainer.cfg
    section("SR training")
    field("Resolution", describe_resolution(get_sr_resolution(cfg)))
    path_field("Run", run_dir)
    if trainer.selection is not None:
        if payload:
            source_run = find_saved_config(resolve_checkpoint(resume)).parent
            trainer.selection.load_inputs(source_run)
        trainer.selection.save_inputs(run_dir)

    def before_step(step: int) -> None:
        interval = cfg["lr_bank"]["refresh_every_steps"]
        if interval and step and step % interval == 0:
            refresh_bank(bank_payload, cfg, step, device, trainer.path_maps)
            publish_bank(bank_payload, cfg, run_dir, step)

    run_train(
        trainer,
        steps=cfg["train"]["total_steps"],
        save_every=cfg["train"]["weights_every_steps"],
        run_dir=run_dir,
        checkpoint_every=cfg["train"]["archive_every_steps"],
        start_step=trainer.completed_steps,
        before_step=before_step,
        config=source,
    )

    return run_dir


def prepare_source(cfg: dict, base_weights: Path | None) -> None:
    if base_weights is None:
        configured = cfg.get("source", {}).get("weights")
        if not isinstance(configured, str) or not configured.strip():
            raise ValueError(
                "set source.weights in the SR config or pass base_weights."
            )
        base_weights = project_path(configured)
    base_path = resolve_weights(base_weights)
    base_cfg = read_source_config(find_saved_config(base_path))
    fill_height_extents(base_cfg)
    if cfg["conditioning"]["height_enabled"]:
        cfg["data"]["height_extents"] = infer_height_extents(cfg["data"])
    cfg["data"] = validate_sr_source(cfg["data"], base_cfg["data"])
    if (
        cfg["conditioning"]["height_enabled"]
        != base_cfg["conditioning"]["height_enabled"]
    ):
        raise ValueError("SR height conditioning must match the frozen LR source.")
    cfg["source"] = {
        "weights": str(base_path),
        "weights_sha256": file_hash(base_path),
        "config_sha256": file_hash(find_saved_config(base_path)),
    }
