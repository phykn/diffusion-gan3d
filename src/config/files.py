from collections.abc import Mapping
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def find_train_config(weights: str | Path) -> Path:
    path = Path(weights).resolve()
    for parent in path.parents:
        config = parent / "train.yaml"
        if config.is_file():
            return config
    raise FileNotFoundError(f"train.yaml was not found above weights file: {path}")


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)
            if data is None:
                data = {}
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {path}") from exc
    if not isinstance(data, dict):
        raise TypeError("YAML root must be a mapping.")
    return data


def save_yaml(path: str | Path, data: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        blocks = [
            yaml.dump({key: value}, Dumper=ConfigDumper, sort_keys=False)
            for key, value in prepare_yaml(data).items()
        ]
        file.write("\n".join(blocks) if blocks else "{}\n")


class ConfigDumper(yaml.SafeDumper):
    def represent_list(self, values):
        return self.represent_sequence("tag:yaml.org,2002:seq", values, flow_style=True)


def prepare_yaml(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: prepare_yaml(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [prepare_yaml(item) for item in value]
    return value


ConfigDumper.add_representer(list, ConfigDumper.represent_list)
