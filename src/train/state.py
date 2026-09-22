import hashlib
from pathlib import Path

from PIL import Image

from src.config.data import get_plane_groups
from src.config.train import normalize_train_config
from src.plane import PLANES
from src.storage import atomic_torch_save


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


def describe_data(trainer):
    split = trainer.cfg["data"].get("split", {})
    records = []
    for domain, streams in trainer.streams.items():
        for axis, stream in streams.items():
            dataset = stream.loader.dataset
            for group in dataset.path_groups:
                for path in group:
                    with Image.open(path) as image:
                        shape = [image.height, image.width]
                    records.append(
                        {
                            "image_id": str(path.resolve()),
                            "domain": domain,
                            "plane": PLANES[axis],
                            "source_shape": shape,
                            "sha256": trainer.data_fingerprint[str(path.resolve())],
                            "validation_region": split.get(
                                "validation_regions", {}
                            ).get(str(path.resolve())),
                        }
                    )
    return {
        "has_measured_3d_reference": False,
        "connectivity_reference": "generated_replay"
        if trainer.connectivity_critic is not None
        else None,
        "coordinate_units": "source pixels",
        "height_coordinate": "2 * cell_center / full_source_extent - 1",
        "training_sources": records,
        "split": split,
    }


def save_training(path: str | Path, trainer) -> None:
    payload = _capture_state(trainer)
    payload.update(
        format="diffusion-gan3d.lr.train",
        connectivity=trainer.connectivity_critic.state_dict(),
        connectivity_optim=trainer.connectivity_optim.state_dict(),
        use_multi_anchor_next=trainer.use_multi_anchor_next,
        anchor_bank=trainer.anchor_bank.entries,
    )
    atomic_torch_save(payload, path)


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
    _restore_state(trainer, payload)
    trainer.connectivity_critic.load_state_dict(payload["connectivity"])
    trainer.connectivity_optim.load_state_dict(payload["connectivity_optim"])
    trainer.use_multi_anchor_next = payload["use_multi_anchor_next"]
    trainer.anchor_bank.entries = payload["anchor_bank"]


def save_sr_training(trainer, path: str | Path) -> None:
    payload = _capture_state(trainer)
    payload["format"] = "diffusion-gan3d.sr.train"
    atomic_torch_save(payload, path)


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
        "path_maps": getattr(trainer, "path_maps", []),
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
    trainer.path_maps = payload.get("path_maps", [])


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
