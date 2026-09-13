import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path

import torch
from tqdm import trange

from src.build.data import build_augmentation
from src.build.predict import load_generator
from src.build.sr import build_sr_trainer
from src.config import (
    find_train_config,
    get_domains,
    get_sr_sizes,
    load_generation_settings,
    load_train_config,
    normalize_train_config,
    save_yaml,
    validate_sr_source,
)
from src.data.sr import load_bank
from src.train.sr import validate_sr_config


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
        if payload.get("format") not in (
            "diffusion-gan3d.sr.train.v1",
            "diffusion-gan3d.sr.train.v2",
            "diffusion-gan3d.sr.train.v3",
            "diffusion-gan3d.sr.train.v4",
        ):
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
        if "data" not in cfg:
            # Legacy SR YAMLs implicitly selected the stage-1 data and augmentation.
            cfg["data"] = copy.deepcopy(base_cfg["data"])
            cfg.setdefault(
                "augmentation", copy.deepcopy(base_cfg.get("augmentation", {}))
            )
            cfg = normalize_train_config(cfg, "sr")
        validate_sr_source(cfg["data"], base_cfg["data"])
        if bank_size is not None:
            cfg["lr_bank"]["samples_per_domain"] = bank_size
        if "guidance" not in cfg["lr_bank"]:
            cfg["lr_bank"]["guidance"] = load_generation_settings().guidance
        cfg["source"] = {
            "weights": str(base_path),
            "weights_sha256": file_hash(base_path),
            "guidance": cfg["lr_bank"]["guidance"],
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
        for domain in get_domains(cfg["data"]):
            bank[domain] = torch.stack(
                [
                    generator.generate(
                        domain=domain, guidance=cfg["source"]["guidance"]
                    )
                    for _ in trange(
                        cfg["lr_bank"]["samples_per_domain"],
                        desc=f"LR bank domain {domain}",
                    )
                ]
            )
        del generator
        if device.type == "cuda":
            torch.cuda.empty_cache()
        bank_payload = {
            "format": "diffusion-gan3d.lr-bank.v1",
            "volumes": bank,
            "data": cfg["data"],
            "source": dict(cfg["source"]),
        }
        bank_path = run_dir / "lr_bank.pt"
        torch.save(bank_payload, bank_path)
        cfg["source"].update(bank=str(bank_path), bank_sha256=file_hash(bank_path))
    trainer = build_sr_trainer(cfg, bank_payload["volumes"], device)
    if payload:
        trainer.resume(payload)
    save_yaml(run_dir / "config.yaml", cfg)
    crop, low, high = get_sr_sizes(cfg)
    print(f"SR: source crop {crop}, LR {low}³ -> HR {high}³; run {run_dir}", flush=True)
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for _ in trange(trainer.step, cfg["train"]["total_steps"], desc="SR train"):
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
