from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {path}") from exc
    if not isinstance(data, dict):
        raise TypeError("YAML root must be a mapping.")
    return data


def save_yaml(path: str | Path, data: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        yaml.safe_dump(prepare_yaml(data), file, sort_keys=False)


def prepare_yaml(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: prepare_yaml(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [prepare_yaml(item) for item in value]
    return value
