from pathlib import Path

from src.config.files import find_saved_config, load_yaml, project_path
from src.config.train import normalize_loaded_train_config
from src.storage import file_hash
from src.train.relocate import PathRemapper


def read_source_config(path: Path, path_maps: list | None = None) -> dict:
    cfg = load_yaml(path)
    if path_maps:
        cfg = PathRemapper(path_maps).config(cfg)
    return normalize_loaded_train_config(cfg, "low_res")


def frozen_source_weights(source: dict) -> Path:
    """Return the hash-checked LR generator an SR run was trained from."""
    if not source.get("weights"):
        raise ValueError("SR export has no LR source; pass --lr-weight.")
    weights = project_path(source["weights"])
    load_frozen_source({**source, "weights": str(weights)})
    return weights


def load_frozen_source(source: dict, path_maps: list | None = None) -> dict:
    weights = Path(source["weights"])
    config = find_saved_config(weights)
    for key, path, label in (
        ("weights_sha256", weights, "weights"),
        ("config_sha256", config, "configuration"),
    ):
        if not source.get(key):
            raise ValueError(f"frozen LR source has no saved {label} hash.")
        if file_hash(path) != source[key]:
            raise ValueError(f"frozen LR source {label} changed since SR training.")
    return read_source_config(config, path_maps)
