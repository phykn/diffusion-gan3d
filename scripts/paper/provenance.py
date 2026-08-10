import hashlib
import json
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path

from src.config import find_train_config
from src.model.denoiser import validate_guidance_scale


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


def validate_output_paths(
    provenance: Mapping[str, object],
    outputs: Sequence[str | Path],
) -> None:
    input_paths = {
        Path(str(provenance["weights"])).resolve(),
        Path(str(provenance["train_config"])).resolve(),
    }
    reference = provenance.get("reference")
    if reference is not None:
        input_paths.add(Path(str(reference)).resolve())
    records = provenance.get("additional_inputs")
    if not isinstance(records, dict):
        raise TypeError("provenance additional_inputs must be an object.")
    input_paths.update(
        Path(str(record["path"])).resolve() for record in records.values()
    )

    resolved_outputs = tuple(Path(path).resolve() for path in outputs)
    if any(
        _same_file(left, right) for left, right in combinations(resolved_outputs, 2)
    ):
        raise ValueError("paper output paths must be distinct.")
    for output in resolved_outputs:
        if any(_same_file(output, source) for source in input_paths):
            raise ValueError(f"paper output path conflicts with an input: {output}")


def file_record(path: str | Path) -> dict[str, str]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
    }


def describe_files(paths: Sequence[str | Path]) -> list[str]:
    return [str(Path(path).resolve()) for path in paths]


def validate_manifest(
    path: str | Path,
    expected: Mapping[str, object],
    *,
    label: str,
    cached_paths: Sequence[str | Path] | None = None,
    output_paths: Sequence[str | Path] | None = None,
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
    if output_paths is not None:
        validate_cached_files(data.get("outputs"), output_paths, label=label)
    return data


def validate_cached_files(
    recorded: object,
    paths: Sequence[str | Path],
    *,
    label: str,
) -> None:
    expected_paths = tuple(Path(path).resolve() for path in paths)
    if not isinstance(recorded, list):
        raise TypeError(f"{label} manifest output paths must be a list.")
    if not all(isinstance(path, str) for path in recorded):
        raise TypeError(f"{label} manifest output paths must be strings.")
    recorded_paths = tuple(Path(path).resolve() for path in recorded)
    if len(set(recorded_paths)) != len(recorded_paths):
        raise ValueError(f"{label} manifest contains duplicate output paths.")
    if len(recorded_paths) != len(expected_paths):
        raise ValueError(f"{label} cached output count does not match the manifest.")
    if set(recorded_paths) != set(expected_paths):
        raise ValueError(f"{label} cached output paths do not match the manifest.")
    for expected_path in expected_paths:
        if not expected_path.is_file():
            raise FileNotFoundError(f"cached output was not found: {expected_path}")


def _same_file(left: Path, right: Path) -> bool:
    if left == right:
        return True
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False
