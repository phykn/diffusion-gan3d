from pathlib import Path

import torch
from tqdm import trange

from src.build.data import build_augmentation, build_datasets
from src.build.predict import load_generator
from src.build.sr import build_sr_trainer
from src.config.data import get_domains, validate_sr_source
from src.config.files import PROJECT_ROOT, find_train_config
from src.config.train import (
    get_sr_sizes,
    load_train_config,
    normalize_train_config,
    validate_sr_config,
)
from src.data.bank import load_bank
from src.data.source import infer_height_extents
from src.train.relocate import relocate_checkpoint
from src.train.run.bank import (
    file_hash,
    model_bank_condition,
    refresh_bank,
    sample_bank_condition,
    save_bank,
)
from src.train.run.loop import make_run_dir, run_train
from src.train.state import resume_sr_training


def run_sr_train(
    device: str | torch.device = "cpu",
    base_weights: Path | None = None,
    resume: Path | None = None,
    config: Path | None = None,
    run_dir: Path | None = None,
    steps: int | None = None,
    bank_size: int | None = None,
    data: Path | None = None,
    path_map: list[tuple[str, str]] | None = None,
) -> Path:
    if path_map and resume is None:
        raise ValueError("path_map requires --resume.")
    if base_weights is not None and resume is not None:
        raise ValueError("base_weights cannot change on resume.")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    payload = None
    if resume:
        if config is not None or data is not None:
            raise ValueError(
                "--resume uses its saved config; only --steps may override training settings."
            )
        if bank_size is not None:
            raise ValueError("--bank-size cannot change on resume.")
        payload = torch.load(resume, map_location="cpu", weights_only=True)
        if payload.get("format") != "diffusion-gan3d.sr.train":
            raise ValueError("resume requires an SR training checkpoint.")
        payload = relocate_checkpoint(payload, path_map)
        cfg = normalize_train_config(payload["config"], "sr")
        bank_path = Path(cfg["source"]["bank"])
        if file_hash(bank_path) != cfg["source"]["bank_sha256"]:
            raise ValueError("LR bank changed since the checkpoint was saved.")
        bank_payload = load_bank(bank_path)
    else:
        cfg = load_train_config(
            config or PROJECT_ROOT / "config/train/sr.yaml",
            "sr",
            data=data,
        )
        if base_weights is None:
            configured = cfg.get("source", {}).get("weights")
            if not isinstance(configured, str) or not configured.strip():
                raise ValueError(
                    "set source.weights in the SR config or provide --base-weights."
                )
            base_path = PROJECT_ROOT / Path(configured).expanduser()
        else:
            base_path = base_weights.expanduser()
        base_path = base_path.resolve()
        if base_path.is_dir():
            base_path = base_path / "generator.pt"
        base_cfg = load_train_config(find_train_config(base_path))
        if cfg["conditioning"]["height_enabled"]:
            cfg["data"]["height_extents"] = infer_height_extents(cfg["data"])
        cfg["data"] = validate_sr_source(cfg["data"], base_cfg["data"])
        if (
            cfg["conditioning"]["height_enabled"]
            != base_cfg["conditioning"]["height_enabled"]
        ):
            raise ValueError("SR height conditioning must match the frozen LR source.")
        if bank_size is not None:
            cfg["lr_bank"]["samples_per_domain"] = bank_size
        cfg["source"] = {
            "weights": str(base_path),
            "weights_sha256": file_hash(base_path),
            "config_sha256": file_hash(find_train_config(base_path)),
        }
    if steps is not None:
        cfg["train"]["total_steps"] = steps
    cfg = validate_sr_config(cfg)
    build_augmentation(cfg)
    if payload and cfg["train"]["total_steps"] <= payload["step"]:
        raise ValueError("--steps must exceed the completed checkpoint step.")
    run_dir = make_run_dir(PROJECT_ROOT / "run", "sr", run_dir, cfg["nickname"])
    if payload is None:
        generator = load_generator(base_path, device)
        bank = {}
        origins, extents, bank_conditions = {}, {}, {}
        datasets = (
            build_datasets(base_cfg) if cfg["conditioning"]["height_enabled"] else None
        )
        for domain in get_domains(cfg["data"]):
            bank_conditions[domain] = [
                sample_bank_condition(base_cfg, domain, datasets)
                for _ in range(cfg["lr_bank"]["samples_per_domain"])
            ]
            origins[domain] = torch.tensor(
                [c.get("height_origin", 0.0) for c in bank_conditions[domain]]
            )
            extents[domain] = torch.tensor(
                [c.get("height_extent", 1.0) for c in bank_conditions[domain]]
            )
            bank[domain] = torch.stack(
                [
                    generator.generate_probs(
                        domain=domain,
                        guidance=cfg["lr_bank"]["guidance"],
                        **model_bank_condition(bank_conditions[domain][i]),
                    )
                    for i in trange(
                        cfg["lr_bank"]["samples_per_domain"],
                        desc=f"LR bank domain {domain}",
                    )
                ]
            )
        del generator
        if device.type == "cuda":
            torch.cuda.empty_cache()
        bank_payload = {
            "format": "diffusion-gan3d.lr-bank",
            "volumes": bank,
            "height_extents": extents,
            "conditions": bank_conditions,
            "height_origins": origins
            if cfg["conditioning"]["height_enabled"]
            else None,
            "data": cfg["data"],
            "source": dict(cfg["source"]),
        }
        cfg["source"].update(save_bank(run_dir, 0, bank_payload))
    trainer = build_sr_trainer(
        cfg,
        bank_payload["volumes"],
        device,
        bank_payload.get("height_origins"),
        bank_payload.get("height_extents"),
    )
    trainer.bank_conditions = bank_payload.get("conditions")
    if payload:
        resume_sr_training(trainer, payload)
    cfg = trainer.cfg
    crop, low, high = get_sr_sizes(cfg)
    print(f"SR: source crop {crop}, LR {low}³ -> HR {high}³; run {run_dir}", flush=True)

    def before_step(step: int) -> None:
        interval = cfg["lr_bank"]["refresh_every_steps"]
        if interval and step and step % interval == 0:
            refresh_bank(trainer, run_dir)

    run_train(
        trainer,
        steps=cfg["train"]["total_steps"],
        save_every=cfg["train"]["weights_every_steps"],
        run_dir=run_dir,
        checkpoint_every=cfg["train"]["archive_every_steps"],
        start_step=trainer.completed_steps,
        before_step=before_step,
    )

    return run_dir
