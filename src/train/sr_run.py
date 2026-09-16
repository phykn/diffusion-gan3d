import hashlib
import json
from datetime import datetime
from pathlib import Path

import torch
from tqdm import trange

from src.build.data import build_augmentation, resolve_height_metadata
from src.build.predict import load_generator
from src.build.sr import build_sr_trainer
from src.config import (
    find_train_config,
    get_domains,
    get_sr_sizes,
    load_train_config,
    normalize_train_config,
    save_yaml,
    validate_sr_config,
    validate_sr_source,
)
from src.data.bank import load_bank


def run_sr_train(
    device: str | torch.device = "cpu",
    base_weights: Path | None = None,
    resume: Path | None = None,
    config: Path | None = None,
    run_dir: Path | None = None,
    steps: int | None = None,
    bank_size: int | None = None,
    data: Path | None = None,
) -> Path:
    if (base_weights is None) == (resume is None):
        raise ValueError("provide exactly one of base_weights or resume.")
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
        if payload.get("format") != "diffusion-gan3d.sr.train.v6":
            raise ValueError("resume requires an SR training checkpoint.")
        cfg = normalize_train_config(payload["config"], "sr")
        bank_path = Path(cfg["source"]["bank"])
        if file_hash(bank_path) != cfg["source"]["bank_sha256"]:
            raise ValueError("LR bank changed since the checkpoint was saved.")
        bank_payload = load_bank(bank_path)
    else:
        cfg = load_train_config(
            config or Path(__file__).resolve().parents[2] / "config/train/sr.yaml",
            "sr",
            data=data,
        )
        base_path = base_weights.resolve()
        if base_path.is_dir():
            base_path = base_path / "generator.pt"
        base_cfg = load_train_config(find_train_config(base_path))
        resolve_height_metadata(cfg)
        validate_sr_source(cfg["data"], base_cfg["data"])
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
    validate_sr_config(cfg)
    build_augmentation(cfg)
    if payload and cfg["train"]["total_steps"] <= payload["step"]:
        raise ValueError("--steps must exceed the completed checkpoint step.")
    run_dir = run_dir or Path("run") / datetime.now().astimezone().strftime(
        "%Y%m%d-%H%M%S-%f-sr"
    )
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "weights").mkdir()
    if payload is None:
        generator = load_generator(base_path, device)
        bank = {}
        origins = {}
        for domain in get_domains(cfg["data"]):
            origins[domain] = sample_origins(
                cfg, domain, cfg["lr_bank"]["samples_per_domain"]
            )
            bank[domain] = torch.stack(
                [
                    generator.generate_probs(
                        domain=domain,
                        guidance=cfg["lr_bank"]["guidance"],
                        **(
                            {"height_origin": float(origins[domain][i])}
                            if cfg["conditioning"]["height_enabled"]
                            else {}
                        ),
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
            "format": "diffusion-gan3d.lr-bank.v2",
            "volumes": bank,
            "height_origins": origins
            if cfg["conditioning"]["height_enabled"]
            else None,
            "data": cfg["data"],
            "source": dict(cfg["source"]),
        }
        bank_path = run_dir / "lr_bank.pt"
        torch.save(bank_payload, bank_path)
        cfg["source"].update(bank=str(bank_path), bank_sha256=file_hash(bank_path))
    trainer = build_sr_trainer(
        cfg, bank_payload["volumes"], device, bank_payload.get("height_origins")
    )
    if payload:
        trainer.resume(payload)
    save_yaml(run_dir / "config.yaml", cfg)
    crop, low, high = get_sr_sizes(cfg)
    print(f"SR: source crop {crop}, LR {low}³ -> HR {high}³; run {run_dir}", flush=True)
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for _ in trange(trainer.step, cfg["train"]["total_steps"], desc="SR train"):
            interval = cfg["lr_bank"]["refresh_every_steps"]
            if interval and trainer.step and trainer.step % interval == 0:
                refresh_bank(trainer, run_dir)
            metrics = trainer.train_step()
            log.write(json.dumps(metrics) + "\n")
            log.flush()
            if (
                trainer.step % cfg["train"]["checkpoint_every_steps"] == 0
                or trainer.step == cfg["train"]["total_steps"]
            ):
                trainer.save(run_dir / "checkpoints/last.pt")
                trainer.export(run_dir / "weights/model.pt")

    return run_dir


def file_hash(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def refresh_bank(trainer, run_dir: Path) -> None:
    """Replace one sample per domain without invalidating banks used by checkpoints."""
    cfg = trainer.cfg
    source = Path(cfg["source"]["weights"])
    if file_hash(source) != cfg["source"]["weights_sha256"]:
        raise ValueError("frozen LR source weights changed before bank refresh.")
    if file_hash(find_train_config(source)) != cfg["source"]["config_sha256"]:
        raise ValueError("frozen LR source configuration changed before bank refresh.")
    generator = load_generator(source, trainer.device)
    interval = cfg["lr_bank"]["refresh_every_steps"]
    for domain, volumes in trainer.bank.items():
        index = (trainer.step // interval - 1) % len(volumes)
        conditions = {}
        if cfg.get("conditioning", {}).get("height_enabled", False):
            origin = sample_origins(cfg, domain, 1)[0]
            trainer.bank_origins[domain][index] = origin
            conditions["height_origin"] = float(origin)
        volumes[index] = generator.generate_probs(
            domain=domain, guidance=cfg["lr_bank"]["guidance"], **conditions
        )
    del generator
    if trainer.device.type == "cuda":
        torch.cuda.empty_cache()
    path = run_dir / f"lr_bank_step_{trainer.step:08d}.pt"
    torch.save(
        {
            "format": "diffusion-gan3d.lr-bank.v2",
            "volumes": trainer.bank,
            "height_origins": getattr(trainer, "bank_origins", None),
            "data": cfg["data"],
            "source": dict(cfg["source"]),
        },
        path,
    )
    cfg["source"].update(bank=str(path), bank_sha256=file_hash(path))
    save_yaml(run_dir / "config.yaml", cfg)


def sample_origins(cfg, domain, count):
    if not cfg["conditioning"]["height_enabled"]:
        return torch.zeros(count)
    maximum = cfg["data"]["height_extents"][domain] - cfg["data"]["crop_size"]
    return torch.rand(count) * maximum
