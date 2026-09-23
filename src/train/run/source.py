import hashlib
from pathlib import Path

from src.config.files import find_train_config, load_yaml
from src.config.train import (
    normalize_loaded_train_config,
    resolve_external_data_path,
)
from src.train.relocate import PathRemapper


def file_hash(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _validate_source_files(weights: Path, recorded: dict) -> None:
    for key, path, label in (
        ("weights_sha256", weights, "weights"),
        ("config_sha256", find_train_config(weights), "configuration"),
    ):
        if not recorded.get(key):
            raise ValueError(f"frozen LR source has no saved {label} hash.")
        if file_hash(path) != recorded[key]:
            raise ValueError(f"frozen LR source {label} changed since SR training.")


def validate_frozen_source(
    weights: Path, recorded: dict, path_maps: list | None = None
) -> None:
    _validate_source_files(weights, recorded)
    if "data_sha256" in recorded:
        _, data_sha256 = read_source_config(find_train_config(weights), path_maps)
        if data_sha256 != recorded["data_sha256"]:
            raise ValueError(
                "frozen LR source data configuration changed since SR training."
            )


def read_source_config(
    path: Path, path_maps: list | None = None
) -> tuple[dict, str | None]:
    cfg = load_yaml(path)
    external_data = isinstance(cfg.get("data"), str)
    remapper = PathRemapper(path_maps) if path_maps else None
    data_sha256 = None
    if external_data:
        data_path = cfg["data"]
        if remapper:
            mapped_path = remapper(data_path)
            if mapped_path == data_path:
                mapped_path = remapper(str(resolve_external_data_path(data_path)))
            data_path = mapped_path
        data_path = resolve_external_data_path(data_path)
        data_sha256 = file_hash(data_path)
        cfg["data"] = load_yaml(data_path)
    if remapper:
        cfg = remapper.config(cfg)
    return (
        normalize_loaded_train_config(cfg, "low_res", external_data),
        data_sha256,
    )


def load_frozen_source(source: dict, path_maps: list | None = None) -> dict:
    weights = Path(source["weights"])
    _validate_source_files(weights, source)
    cfg, data_sha256 = read_source_config(find_train_config(weights), path_maps)
    if "data_sha256" in source and data_sha256 != source["data_sha256"]:
        raise ValueError(
            "frozen LR source data configuration changed since SR training."
        )
    return cfg
