import os
import re
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import torch

from src.config.schema import CONFIG_TYPES, build_config
from src.export import STAGE_NAMES
from src.storage import torch_save

FORMAT = "diffusion-gan3d.train"
VERSION = 3
_NAME = re.compile(r"step_(\d+)\.pt")


def checkpoint_paths(root, step):
    """A new step's training and EMA export files under root/checkpoints."""
    folder = root / "checkpoints"
    path = folder / f"step_{step:08d}.pt"
    export = folder / f"generator_{step:08d}.pt"
    for target in (path, export, path.with_suffix(".complete")):
        if target.exists():
            raise FileExistsError(target)
    return path, export


def complete_checkpoint(path):
    """Publish completion only after the checkpoint has been flushed and closed."""
    with path.with_suffix(".complete").open("x", encoding="ascii") as file:
        file.write(f"{path.stat().st_size}\n")
        file.flush()
        os.fsync(file.fileno())


def is_complete(path):
    try:
        return path.with_suffix(".complete").read_text(encoding="ascii") == (
            f"{path.stat().st_size}\n"
        )
    except (FileNotFoundError, UnicodeError):
        return False


def resolve_checkpoint(value):
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


def save_checkpoint(payload, root, export_payload):
    path, export = checkpoint_paths(root, payload["step"])
    torch_save(payload, path, overwrite=False)
    torch_save(export_payload, export, overwrite=False)
    complete_checkpoint(path)
    return path


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or payload.get("format") != FORMAT
        or payload.get("version") != VERSION
        or payload.get("stage") not in STAGE_NAMES
    ):
        raise ValueError(f"not a v{VERSION} training checkpoint: {path}.")
    return dict(payload)


def _missing_keys(cls, values, path=""):
    missing = []
    for item in fields(cls):
        key = f"{path}{item.name}"
        if item.name not in values:
            missing.append(key)
        elif is_dataclass(item.type) and isinstance(values[item.name], Mapping):
            missing.extend(_missing_keys(item.type, values[item.name], f"{key}."))
    return missing


def saved_config(payload):
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


def fingerprint_changes(saved, current):
    """plane/file names whose images were added, removed or changed."""
    changes = []
    for plane in sorted(set(saved) | set(current)):
        before, after = saved.get(plane, {}), current.get(plane, {})
        for name in sorted(set(before) | set(after)):
            if before.get(name) != after.get(name):
                changes.append(f"{plane}/{name}")
    return changes


def check_fingerprint(saved, current):
    changes = fingerprint_changes(saved, current)
    if changes:
        shown = ", ".join(changes[:5]) + (" ..." if len(changes) > 5 else "")
        raise ValueError(f"training images changed since the checkpoint: {shown}")


def _pure_path(value):
    text = str(value).replace("\\", "/")
    return PureWindowsPath(text) if PureWindowsPath(text).drive else PurePosixPath(text)


def _remap(value, pairs, matched):
    for old, new in pairs:
        try:
            relative = _pure_path(value).relative_to(old)
        except ValueError:
            continue
        matched.add(old)
        return new.joinpath(*relative.parts).as_posix()
    return value


def relocate(payload, path_map):
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
