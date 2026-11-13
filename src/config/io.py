from pathlib import Path

import yaml

from src.config.schema import build_config, build_generation_settings, to_dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(path):
    return (PROJECT_ROOT / Path(path).expanduser()).resolve()


def stored_path(path):
    resolved = project_path(path)
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def load_yaml(path):
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {path}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


class _Dumper(yaml.SafeDumper):
    def represent_list(self, values):
        return self.represent_sequence("tag:yaml.org,2002:seq", values, flow_style=True)


_Dumper.add_representer(list, _Dumper.represent_list)


def dump_yaml(data):
    return (
        "\n".join(
            yaml.dump({key: value}, Dumper=_Dumper, sort_keys=False)
            for key, value in to_dict(data).items()
        )
        or "{}\n"
    )


def save_yaml(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_yaml(data), encoding="utf-8")
    return path


def load_config(path):
    return build_config(load_yaml(path))


def load_generation_settings():
    return build_generation_settings(load_yaml(PROJECT_ROOT / "config" / "gen.yaml"))
