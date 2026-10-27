from pathlib import Path

import torch

from src.config.train import normalize_train_config
from src.storage import SR_FORMAT, torch_save

TRAIN_FORMATS = {
    "low_res": "diffusion-gan3d.lr.train",
    "sr": "diffusion-gan3d.sr.train",
}


def load_training(path: str | Path, stage: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != TRAIN_FORMATS[stage]:
        label = "an SR" if stage == "sr" else "an LR"
        raise ValueError(f"--resume requires {label} training checkpoint.")
    return payload


def save_training(trainer, path: str | Path) -> None:
    payload = _capture_state(trainer)
    payload["format"] = TRAIN_FORMATS[trainer.cfg["stage"]]
    torch_save(payload, path, overwrite=False)


def resume_training(trainer, payload: dict) -> None:
    _check_resume(trainer, payload, trainer.cfg["stage"])
    _restore_state(trainer, payload)


def _check_resume(trainer, payload: dict, stage: str) -> None:
    label = "SR" if stage == "sr" else "LR"
    _require_saved_config(
        normalize_train_config(payload["config"], stage),
        normalize_train_config(trainer.cfg, stage),
        f"{label} resume must use saved settings; only total_steps may change.",
    )
    if payload["data_fingerprint"] != trainer.data_fingerprint:
        raise ValueError(f"{label} source images changed since the checkpoint.")


def _require_saved_config(saved: dict, actual: dict, message: str) -> None:
    """Compare resolved settings, ignoring the total_steps a resume may extend."""

    def without_steps(cfg):
        train = {k: v for k, v in cfg["train"].items() if k != "total_steps"}
        return {**cfg, "train": train}

    if without_steps(saved) != without_steps(actual):
        raise ValueError(message)


def _capture_state(trainer) -> dict:
    return {
        "config": trainer.cfg,
        "step": trainer.completed_steps,
        "data_fingerprint": trainer.data_fingerprint,
        "path_maps": trainer.path_maps,
        "updates": trainer.updates,
        "model": trainer.denoiser.state_dict(),
        "ema": trainer.ema_denoiser.state_dict(),
        "critics": trainer.critics.state_dict(),
        "generator_optim": trainer.denoiser_optim.state_dict(),
        "critic_optims": {
            name: optim.state_dict() for name, optim in trainer.critic_optims.items()
        },
        "scaler": trainer.scaler.state_dict(),
        "selection": None
        if trainer.selection is None
        else trainer.selection.state_dict(),
    }


def _restore_state(trainer, payload: dict) -> None:
    path_maps = payload["path_maps"]
    if set(payload["critic_optims"]) != set(trainer.critic_optims):
        raise ValueError("critic optimizer groups do not match the checkpoint.")
    if set(payload["updates"]) != set(trainer.updates):
        raise ValueError("optimizer update counters do not match the checkpoint.")
    trainer.denoiser.load_state_dict(payload["model"])
    trainer.ema_denoiser.load_state_dict(payload["ema"])
    trainer.critics.load_state_dict(payload["critics"])
    trainer.denoiser_optim.load_state_dict(payload["generator_optim"])
    for group, optimizer in trainer.critic_optims.items():
        optimizer.load_state_dict(payload["critic_optims"][group])
    # Runs without AMP save an empty scaler state; an enabled scaler then starts
    # from its initial scale.
    if payload["scaler"]:
        trainer.scaler.load_state_dict(payload["scaler"])
    if trainer.selection is not None:
        trainer.selection.load_state_dict(payload["selection"])
    trainer.updates = dict(payload["updates"])
    trainer.completed_steps = payload["step"]
    trainer.path_maps = path_maps


def export_generator(cfg: dict, step: int, model: dict, path, overwrite=True) -> None:
    """Write EMA weights for inference; SR exports also carry their config."""
    if cfg["stage"] == "sr":
        model = {"format": SR_FORMAT, "config": cfg, "step": step, "model": model}
    torch_save(model, path, overwrite=overwrite)
