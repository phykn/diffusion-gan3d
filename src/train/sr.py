import torch
import torch.nn.functional as F

from src.config.data import get_plane_groups
from src.config.train import normalize_train_config
from src.prepare.resize import phase_channels
from src.storage import atomic_torch_save


def corrupt_coarse(low, probability, strength):
    if probability == 0 or strength == 0:
        return low, low.new_zeros(len(low))
    shape = (low.shape[0], 1, *(max(1, n // 4) for n in low.shape[2:]))
    active = torch.rand((len(low), 1, 1, 1, 1), device=low.device) < probability
    level = torch.rand((len(low), 1, 1, 1, 1), device=low.device) * strength * active
    mask = (
        F.interpolate(
            (torch.rand(shape, device=low.device) < level).float(),
            size=low.shape[2:],
            mode="nearest",
        ).bool()
        & active
    )
    labels = torch.randint(low.shape[1], (shape[0], *shape[2:]), device=low.device)
    replacement = phase_channels(labels, low.shape[1])
    replacement = F.interpolate(replacement, size=low.shape[2:], mode="nearest")
    corrupted = torch.where(mask, replacement, low)
    # Matching replacements do not contribute to the realized corruption level.
    level = (corrupted - low).abs().mean(dim=(2, 3, 4)).sum(1) * 0.5
    return corrupted, level


def save_sr_training(trainer, path):
    atomic_torch_save(
        {
            "format": "diffusion-gan3d.sr.train",
            "config": trainer.cfg,
            "step": trainer.completed_steps,
            "data_fingerprint": trainer.data_fingerprint,
            "updates": trainer.updates,
            "model": trainer.denoiser.state_dict(),
            "ema": trainer.ema_denoiser.state_dict(),
            "critics": trainer.critics.state_dict(),
            "generator_optim": trainer.denoiser_optim.state_dict(),
            "critic_optims": {
                name: optim.state_dict()
                for name, optim in trainer.critic_optims.items()
            },
            "scaler": trainer.scaler.state_dict(),
        },
        path,
    )


def resume_sr_training(trainer, payload):
    if payload.get("format") != "diffusion-gan3d.sr.train":
        raise ValueError("unsupported SR training checkpoint format.")
    if payload["data_fingerprint"] != trainer.data_fingerprint:
        raise ValueError("SR source images changed since the checkpoint.")
    saved = normalize_train_config(payload["config"], "sr")
    if get_plane_groups(saved) != get_plane_groups(trainer.cfg):
        raise ValueError("critic plane_groups cannot change on resume.")
    expected, actual = dict(saved), dict(trainer.cfg)
    expected["train"], actual["train"] = dict(expected["train"]), dict(actual["train"])
    expected["train"].pop("total_steps")
    actual["train"].pop("total_steps")
    if expected != actual:
        raise ValueError("SR resume must use its saved resolved configuration.")
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
    trainer.scaler.load_state_dict(payload["scaler"])
    trainer.updates = dict(payload["updates"])
    trainer.completed_steps = payload["step"]


def export_sr(trainer, path):
    atomic_torch_save(
        {
            "format": "diffusion-gan3d.sr",
            "config": trainer.cfg,
            "step": trainer.completed_steps,
            "model": trainer.ema_denoiser.state_dict(),
        },
        path,
    )
