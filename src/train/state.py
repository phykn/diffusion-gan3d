from pathlib import Path

import torch

from src.config.data import get_plane_groups
from src.config.train import normalize_train_config
from src.storage import torch_save

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
    payload.update(
        format=TRAIN_FORMATS["low_res"],
        connectivity=trainer.connectivity_critic.state_dict(),
        connectivity_optim=trainer.connectivity_optim.state_dict(),
        seam=trainer.seam_critic.state_dict(),
        seam_optim=trainer.seam_optim.state_dict(),
        use_multi_anchor_next=trainer.conditions.use_multi_anchor_next,
        anchor_bank=trainer.conditions.anchor_bank.entries,
    )
    torch_save(payload, path, overwrite=False)


def resume_training(trainer, payload: dict) -> None:
    _require_saved_config(
        normalize_train_config(payload["config"]),
        normalize_train_config(trainer.cfg),
        "LR resume must use saved settings; only total_steps may change.",
    )
    if payload["data_fingerprint"] != trainer.data_fingerprint:
        raise ValueError("LR source images changed since the checkpoint.")
    missing = {"connectivity", "connectivity_optim", "seam", "seam_optim"}
    missing -= payload.keys()
    if missing:
        raise ValueError(f"LR training checkpoint is missing {sorted(missing)}.")
    _restore_state(trainer, payload)
    trainer.connectivity_critic.load_state_dict(payload["connectivity"])
    trainer.connectivity_optim.load_state_dict(payload["connectivity_optim"])
    trainer.seam_critic.load_state_dict(payload["seam"])
    trainer.seam_optim.load_state_dict(payload["seam_optim"])
    trainer.conditions.use_multi_anchor_next = payload["use_multi_anchor_next"]
    trainer.conditions.anchor_bank.entries = payload["anchor_bank"]


def save_sr_training(trainer, path: str | Path) -> None:
    payload = _capture_state(trainer)
    payload["format"] = TRAIN_FORMATS["sr"]
    torch_save(payload, path, overwrite=False)


def resume_sr_training(trainer, payload: dict) -> None:
    if payload["data_fingerprint"] != trainer.data_fingerprint:
        raise ValueError("SR source images changed since the checkpoint.")
    saved = normalize_train_config(payload["config"], "sr")
    if get_plane_groups(saved) != get_plane_groups(trainer.cfg):
        raise ValueError("critic plane_groups cannot change on resume.")
    _require_saved_config(
        saved, trainer.cfg, "SR resume must use its saved resolved configuration."
    )
    _restore_state(trainer, payload)


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
    # Runs without CUDA AMP save an empty scaler state; an enabled scaler then
    # starts from its initial scale.
    if payload["scaler"]:
        trainer.scaler.load_state_dict(payload["scaler"])
    trainer.updates = dict(payload["updates"])
    trainer.completed_steps = payload["step"]
    trainer.path_maps = path_maps


def export_sr(trainer, path, *, overwrite=True):
    torch_save(
        {
            "format": "diffusion-gan3d.sr",
            "config": trainer.cfg,
            "step": trainer.completed_steps,
            "model": trainer.ema_denoiser.state_dict(),
        },
        path,
        overwrite=overwrite,
    )
