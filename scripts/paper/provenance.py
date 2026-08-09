"""Shared, read-only provenance helpers for paper generation artifacts."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.build import SCHEMA_VERSION, require_schema
from src.model.denoiser import validate_guidance_scale
from src.utils import load_yaml


def sha256_file(path: str | Path) -> str:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"provenance input was not found: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_provenance(
    weights: str | Path,
    guidance_scale: float,
    *,
    generation: Mapping[str, object],
    reference: str | Path | None = None,
    additional_inputs: Mapping[str, str | Path] | None = None,
) -> dict[str, object]:
    weight_path = Path(weights).resolve()
    config_path = find_train_config(weight_path)
    require_schema(load_yaml(config_path))
    guidance = validate_guidance_scale(guidance_scale)
    reference_path = None if reference is None else Path(reference).resolve()
    inputs = {
        name: file_record(path)
        for name, path in sorted((additional_inputs or {}).items())
    }
    generation_values = json.loads(
        json.dumps(
            dict(generation),
            sort_keys=True,
            allow_nan=False,
        )
    )
    values: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "weights": str(weight_path),
        "weight_sha256": sha256_file(weight_path),
        "train_config": str(config_path),
        "train_config_sha256": sha256_file(config_path),
        "guidance_scale": guidance,
        "reference": None if reference_path is None else str(reference_path),
        "reference_sha256": (
            None if reference_path is None else sha256_file(reference_path)
        ),
        "additional_inputs": inputs,
        "generation": generation_values,
    }
    values["generation_signature"] = hashlib.sha256(
        json.dumps(
            values,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return values


def find_train_config(weights: str | Path) -> Path:
    path = Path(weights).resolve()
    config = next(
        (
            parent / "train.yaml"
            for parent in path.parents
            if (parent / "train.yaml").is_file()
        ),
        None,
    )
    if config is None:
        raise FileNotFoundError(f"train.yaml was not found above weights file: {path}")
    return config.resolve()


def file_record(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
    }


def describe_files(paths: Sequence[str | Path]) -> list[dict[str, str]]:
    return [file_record(path) for path in paths]


def validate_manifest(
    path: str | Path,
    expected: Mapping[str, object],
    *,
    label: str,
    cached_paths: Sequence[str | Path] | None = None,
) -> dict[str, object]:
    manifest = Path(path)
    if not manifest.is_file():
        raise FileNotFoundError(f"cannot reuse {label} without manifest: {manifest}")
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} manifest is invalid JSON: {manifest}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"{label} manifest must contain a JSON object.")

    for key, expected_value in expected.items():
        if key not in data:
            raise ValueError(f"{label} manifest is missing '{key}'.")
        if data[key] != expected_value:
            raise ValueError(f"{label} manifest '{key}' does not match current inputs.")
    if cached_paths is not None:
        validate_cached_files(data.get("cached_outputs"), cached_paths, label=label)
    return data


def validate_cached_files(
    recorded: object,
    paths: Sequence[str | Path],
    *,
    label: str,
) -> None:
    expected_paths = tuple(Path(path).resolve() for path in paths)
    indexed = _index_file_records(recorded, label)
    if len(indexed) != len(expected_paths):
        raise ValueError(f"{label} cached output count does not match the manifest.")
    if set(indexed) != set(expected_paths):
        raise ValueError(f"{label} cached output paths do not match the manifest.")
    for expected_path in expected_paths:
        if not expected_path.is_file():
            raise FileNotFoundError(f"cached output was not found: {expected_path}")
        if indexed[expected_path] != sha256_file(expected_path):
            raise ValueError(f"cached output SHA-256 does not match: {expected_path}")


def _index_file_records(recorded: object, label: str) -> dict[Path, str]:
    if not isinstance(recorded, list):
        raise TypeError(f"{label} manifest 'cached_outputs' must be a list.")
    indexed: dict[Path, str] = {}
    for entry in recorded:
        if not isinstance(entry, dict):
            raise TypeError(f"{label} cached output records must be JSON objects.")
        raw_path = entry.get("path")
        digest = entry.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(digest, str):
            raise TypeError(f"{label} cached output record has invalid fields.")
        path = Path(raw_path).resolve()
        if path in indexed:
            raise ValueError(f"{label} manifest contains duplicate cached outputs.")
        indexed[path] = digest
    return indexed
