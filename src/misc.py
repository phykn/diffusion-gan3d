import math
import os
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from uuid import uuid4

import torch
import yaml


def load_mapping(path: str | Path, *, label: str) -> dict:
    config_path = Path(path)
    try:
        with open(config_path, encoding="utf-8") as file:
            values = yaml.safe_load(file) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"{label} is malformed: {config_path}") from exc
    if not isinstance(values, dict):
        raise TypeError(f"{label} must contain a mapping.")
    return values


def save_mapping(path: str | Path, values: Mapping[object, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = _encode(values)
    with open(output, "w", encoding="utf-8") as file:
        yaml.safe_dump(encoded, file, sort_keys=False)


def atomic_torch_save(values: object, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        torch.save(values, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def require_int(name: str, value: object, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def require_number(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real scalar.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}.")
    return result


def require_finite(name: str, values: torch.Tensor) -> None:
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError(f"{name} must contain only finite values.")


def _encode(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_encode(item) for item in value]
    return value
