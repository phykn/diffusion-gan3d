from pathlib import Path

from src.config.data import get_plane_groups
from src.config.train import normalize_train_config
from src.storage import torch_save


def save_training(trainer, path: str | Path) -> None:
    payload = _capture_state(trainer)
    payload.update(
        format="diffusion-gan3d.lr.train",
        connectivity=trainer.connectivity_critic.state_dict(),
        connectivity_optim=trainer.connectivity_optim.state_dict(),
        seam=trainer.seam_critic.state_dict(),
        seam_optim=trainer.seam_optim.state_dict(),
        use_multi_anchor_next=trainer.conditions.use_multi_anchor_next,
        anchor_bank=trainer.conditions.anchor_bank.entries,
    )
    torch_save(payload, path, overwrite=False)


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
    if (
        not {"connectivity", "connectivity_optim"} <= payload.keys()
        or "connectivity" not in payload["updates"]
    ):
        raise ValueError("Connectivity training state is incomplete.")
    seam = {"seam", "seam_optim"} & payload.keys()
    if not seam and "seam" not in payload["updates"]:
        if actual["loss"]["anchor_seam_weight"] > 0.0:
            raise ValueError("Seam training state is missing.")
        payload = {**payload, "updates": {**payload["updates"], "seam": 0}}
    elif len(seam) < 2 or "seam" not in payload["updates"]:
        raise ValueError("Seam training state is incomplete.")
    _restore_state(trainer, payload)
    trainer.connectivity_critic.load_state_dict(payload["connectivity"])
    trainer.connectivity_optim.load_state_dict(payload["connectivity_optim"])
    if seam:
        trainer.seam_critic.load_state_dict(payload["seam"])
        trainer.seam_optim.load_state_dict(payload["seam_optim"])
    trainer.conditions.use_multi_anchor_next = payload["use_multi_anchor_next"]
    trainer.conditions.anchor_bank.entries = payload["anchor_bank"]


def save_sr_training(trainer, path: str | Path) -> None:
    payload = _capture_state(trainer)
    payload["format"] = "diffusion-gan3d.sr.train"
    torch_save(payload, path, overwrite=False)


def resume_sr_training(trainer, payload: dict) -> None:
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
    _restore_state(trainer, payload)


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
