import os
import re
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import torch

from src.config.schema import CONFIG_TYPES, StageConfig, build_config, to_dict
from src.export import STAGE_NAMES, V1_HINT
from src.storage import torch_save

FORMAT = "diffusion-gan3d.train"
VERSION = 2
_NAME = re.compile(r"step_(\d+)\.pt")


def checkpoint_paths(root: Path, step: int) -> tuple[Path, Path]:
    """A new step's training and EMA export files under root/checkpoints."""
    folder = root / "checkpoints"
    path = folder / f"step_{step:08d}.pt"
    export = folder / f"generator_{step:08d}.pt"
    for target in (path, export, path.with_suffix(".complete")):
        if target.exists():
            raise FileExistsError(target)
    return path, export


def complete_checkpoint(path: Path) -> None:
    """Publish completion only after the checkpoint has been flushed and closed."""
    with path.with_suffix(".complete").open("x", encoding="ascii") as file:
        file.write(f"{path.stat().st_size}\n")
        file.flush()
        os.fsync(file.fileno())


def is_complete(path: Path) -> bool:
    try:
        return path.with_suffix(".complete").read_text(encoding="ascii") == (
            f"{path.stat().st_size}\n"
        )
    except (FileNotFoundError, UnicodeError):
        return False


def resolve_checkpoint(value: str | Path) -> Path:
    """The latest completed step of a run folder, or a completed step file."""
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        root = path / "checkpoints" if (path / "checkpoints").is_dir() else path
        candidates = []
        for candidate in root.glob("step_*.pt"):
            match = _NAME.fullmatch(candidate.name)
            if match:
                candidates.append((int(match[1]), candidate))
        for _, candidate in sorted(candidates, reverse=True):
            if is_complete(candidate):
                return candidate
        raise FileNotFoundError(f"no completed training checkpoint found: {value}")
    if not path.is_file():
        raise FileNotFoundError(f"no completed training checkpoint found: {value}")
    if not is_complete(path):
        raise ValueError(f"training checkpoint is incomplete: {path}")
    return path


def capture(trainer) -> dict:
    """Everything a resume needs to continue the run exactly."""
    critics = {group: critic.state_dict() for group, critic in trainer.critics.items()}
    selection = trainer.selection
    return {
        "format": FORMAT,
        "version": VERSION,
        "stage": trainer.config.stage,
        "step": trainer.completed_steps,
        "config": to_dict(trainer.config),
        "models": {
            "generator": trainer.denoiser.state_dict(),
            "ema": trainer.ema.state_dict(),
            "critics": critics,
        },
        "optimizers": {
            "generator": trainer.optimizers["generator"].state_dict(),
            "critics": {
                group: trainer.optimizers[group].state_dict() for group in critics
            },
        },
        "scaler": trainer.scaler.state_dict(),
        "updates": {
            "generator": trainer.updates["generator"],
            "critics": {group: trainer.updates[group] for group in critics},
        },
        "rng": trainer.random.state(),
        "selection": None if selection is None else selection.state_dict(),
        "data_fingerprint": trainer.data_fingerprint,
        "path_maps": list(trainer.path_maps),
        **trainer.stage_state(),
    }


def save_checkpoint(trainer, root: Path, export_payload: dict) -> Path:
    path, export = checkpoint_paths(root, trainer.completed_steps)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch_save(capture(trainer), path, overwrite=False)
    torch_save(export_payload, export, overwrite=False)
    complete_checkpoint(path)
    return path


def load_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or payload.get("format") != FORMAT
        or payload.get("version") != VERSION
        or payload.get("stage") not in STAGE_NAMES
    ):
        raise ValueError(f"not a v2 training checkpoint: {path}. {V1_HINT}")
    return dict(payload)


def _missing_keys(cls, values: Mapping, path: str = "") -> list[str]:
    missing = []
    for item in fields(cls):
        key = f"{path}{item.name}"
        if item.name not in values:
            missing.append(key)
        elif is_dataclass(item.type) and isinstance(values[item.name], Mapping):
            missing.extend(_missing_keys(item.type, values[item.name], f"{key}."))
    return missing


def saved_config(payload: Mapping) -> StageConfig:
    """The checkpoint config under the current schema; no key is filled in."""
    values = payload["config"]
    stage = values.get("stage") if isinstance(values, Mapping) else None
    if stage not in CONFIG_TYPES:
        raise ValueError("checkpoint config has no valid stage.")
    try:
        config = build_config(values)
    except ValueError as error:
        raise ValueError(f"checkpoint config does not match this version: {error}")
    missing = _missing_keys(CONFIG_TYPES[stage], values)
    if missing:
        raise ValueError(
            "checkpoint config does not match this version: missing "
            + ", ".join(missing)
        )
    return config


def fingerprint_changes(saved: Mapping, current: Mapping) -> list[str]:
    """plane/file names whose images were added, removed or changed."""
    changes = []
    for plane in sorted(set(saved) | set(current)):
        before, after = saved.get(plane, {}), current.get(plane, {})
        for name in sorted(set(before) | set(after)):
            if before.get(name) != after.get(name):
                changes.append(f"{plane}/{name}")
    return changes


def check_fingerprint(saved: Mapping, current: Mapping) -> None:
    changes = fingerprint_changes(saved, current)
    if changes:
        shown = ", ".join(changes[:5]) + (" ..." if len(changes) > 5 else "")
        raise ValueError(f"training images changed since the checkpoint: {shown}")


def _pure_path(value):
    text = str(value).replace("\\", "/")
    return PureWindowsPath(text) if PureWindowsPath(text).drive else PurePosixPath(text)


def _remap(value: str, pairs, matched: set) -> str:
    for old, new in pairs:
        try:
            relative = _pure_path(value).relative_to(old)
        except ValueError:
            continue
        matched.add(old)
        return new.joinpath(*relative.parts).as_posix()
    return value


def relocate(payload: dict, path_map) -> dict:
    """Replace saved path prefixes (image folders, source.lr_run) OLD by NEW."""
    if not path_map:
        return payload
    pairs, seen = [], set()
    for old, new in path_map:
        if not str(old).strip() or not str(new).strip():
            raise ValueError("--path-map needs non-empty OLD and NEW prefixes.")
        old = _pure_path(old)
        if old in seen or ".." in old.parts:
            raise ValueError("--path-map OLD prefixes must be distinct and normalized.")
        seen.add(old)
        pairs.append((old, PurePosixPath(Path(new).expanduser().as_posix())))
    # Longest prefix first, so a nested OLD wins over its parent.
    pairs.sort(key=lambda pair: len(pair[0].parts), reverse=True)
    matched: set = set()
    config = dict(payload["config"])
    data = dict(config["data"])
    data["images"] = {
        plane: [_remap(folder, pairs, matched) for folder in folders]
        for plane, folders in data["images"].items()
    }
    config["data"] = data
    if config.get("source", {}).get("lr_run") is not None:
        config["source"] = dict(config["source"])
        config["source"]["lr_run"] = _remap(config["source"]["lr_run"], pairs, matched)
    for old, _ in pairs:
        if old not in matched:
            example = next(iter(data["images"].values()), [None])[0]
            hint = f"; saved paths look like {example}" if example else ""
            raise ValueError(f"--path-map OLD matches no saved path: {old}{hint}")
    result = dict(payload)
    result["config"] = config
    result["path_maps"] = [
        *payload["path_maps"],
        [[old.as_posix(), new.as_posix()] for old, new in pairs],
    ]
    return result


def restore(trainer, payload: Mapping) -> None:
    """Load a checked checkpoint into a trainer built from its config."""
    models, optimizers = payload["models"], payload["optimizers"]
    groups = set(trainer.critics)
    if set(models["critics"]) != groups or set(optimizers["critics"]) != groups:
        raise ValueError("critic groups do not match the checkpoint.")
    trainer.denoiser.load_state_dict(models["generator"])
    trainer.ema.load_state_dict(models["ema"])
    for group, critic in trainer.critics.items():
        critic.load_state_dict(models["critics"][group])
    trainer.optimizers["generator"].load_state_dict(optimizers["generator"])
    for group in groups:
        trainer.optimizers[group].load_state_dict(optimizers["critics"][group])
    # Runs without AMP save an empty scaler state; an enabled scaler then starts
    # from its initial scale.
    if payload["scaler"]:
        trainer.scaler.load_state_dict(payload["scaler"])
    trainer.updates = {
        "generator": payload["updates"]["generator"],
        **dict(payload["updates"]["critics"]),
    }
    trainer.random.load_state(payload["rng"])
    if trainer.selection is not None:
        trainer.selection.load_state_dict(payload["selection"])
    trainer.completed_steps = payload["step"]
    trainer.path_maps = list(payload["path_maps"])
