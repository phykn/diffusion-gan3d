import hashlib
from pathlib import Path

import torch

from src.config import normalize_train_config


def fingerprint_data(streams: dict) -> dict:
    paths = {
        path.resolve()
        for axes in streams.values()
        for stream in axes.values()
        for group in stream.loader.dataset.path_groups
        for path in group
    }
    result = {}
    for path in sorted(paths):
        with path.open("rb") as file:
            result[str(path)] = hashlib.file_digest(file, "sha256").hexdigest()
    return result


def save_training(path: str | Path, trainer) -> None:
    payload = {
        "format": "diffusion-gan3d.lr.train",
        "config": trainer.cfg,
        "step": trainer.completed_steps,
        "model": trainer.denoiser.state_dict(),
        "ema": trainer.ema_denoiser.state_dict(),
        "critics": trainer.critics.state_dict(),
        "connectivity": trainer.connectivity_critic.state_dict(),
        "generator_optim": trainer.denoiser_optim.state_dict(),
        "critic_optims": {
            key: opt.state_dict() for key, opt in trainer.critic_optims.items()
        },
        "connectivity_optim": trainer.connectivity_optim.state_dict(),
        "scaler": trainer.scaler.state_dict(),
        "updates": trainer.updates,
        "use_multi_anchor_next": trainer.use_multi_anchor_next,
        "anchor_bank": trainer.anchor_bank.entries,
        "data_fingerprint": trainer.data_fingerprint,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def resume_training(trainer, payload: dict) -> None:
    if payload.get("format") != "diffusion-gan3d.lr.train":
        raise ValueError(
            "resume requires an LR training checkpoint, not inference weights."
        )
    expected = normalize_train_config(payload["config"])
    actual = normalize_train_config(trainer.cfg)
    expected["train"]["total_steps"] = actual["train"]["total_steps"]
    if expected != actual:
        raise ValueError(
            "LR resume must use saved settings; only total_steps may change."
        )
    if payload["data_fingerprint"] != trainer.data_fingerprint:
        raise ValueError("LR source images changed since the checkpoint.")
    trainer.denoiser.load_state_dict(payload["model"])
    trainer.ema_denoiser.load_state_dict(payload["ema"])
    trainer.critics.load_state_dict(payload["critics"])
    trainer.connectivity_critic.load_state_dict(payload["connectivity"])
    trainer.denoiser_optim.load_state_dict(payload["generator_optim"])
    for key, opt in trainer.critic_optims.items():
        opt.load_state_dict(payload["critic_optims"][key])
    trainer.connectivity_optim.load_state_dict(payload["connectivity_optim"])
    trainer.scaler.load_state_dict(payload["scaler"])
    trainer.updates = dict(payload["updates"])
    trainer.completed_steps = payload["step"]
    trainer.use_multi_anchor_next = payload["use_multi_anchor_next"]
    trainer.anchor_bank.entries = payload["anchor_bank"]
