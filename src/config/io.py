from pathlib import Path

import yaml

from src.config.build import parse, to_dict
from src.config.schema import GenerationSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _FlowListDumper(yaml.SafeDumper):
    def represent_list(self, values):
        return self.represent_sequence("tag:yaml.org,2002:seq", values, flow_style=True)


_FlowListDumper.add_representer(list, _FlowListDumper.represent_list)


def resolve_path(path):
    return (PROJECT_ROOT / Path(path).expanduser()).resolve()


def shorten_path(path):
    resolved = resolve_path(path)
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def load_yaml(path):
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid YAML: {path}") from error
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def dump_yaml(data):
    sections = (
        yaml.dump({key: value}, Dumper=_FlowListDumper, sort_keys=False)
        for key, value in to_dict(data).items()
    )
    return "\n".join(sections)


def save_yaml(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_yaml(data), encoding="utf-8")


def load_generation_settings():
    return parse(GenerationSettings, load_yaml(PROJECT_ROOT / "config" / "gen.yaml"))
