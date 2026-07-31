from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeVar

from ..config import (
    load_yaml,
    require_int,
    require_number,
)


@dataclass(frozen=True)
class OutputConfig:
    data_dir: str | Path
    count: int

    def __post_init__(self) -> None:
        require_int("output.count", self.count, minimum=1)


@dataclass(frozen=True)
class GeometryConfig:
    size: int
    big_radius: int
    small_radius: int
    big_fraction: float
    small_fraction: float
    big_elongation: float = 1.0

    def __post_init__(self) -> None:
        require_int("geometry.size", self.size)
        require_int("geometry.big_radius", self.big_radius)
        require_int("geometry.small_radius", self.small_radius)
        for name, value in (
            ("geometry.big_fraction", self.big_fraction),
            ("geometry.small_fraction", self.small_fraction),
            ("geometry.big_elongation", self.big_elongation),
        ):
            require_number(name, value)
        if self.size < 8:
            raise ValueError("geometry.size must be at least 8.")
        if not 1 < self.small_radius < self.big_radius < self.size / 2:
            raise ValueError(
                "geometry radii must satisfy 1 < small_radius < big_radius < size / 2."
            )
        total = float(self.big_fraction) + float(self.small_fraction)
        if self.big_fraction <= 0.0 or self.small_fraction <= 0.0:
            raise ValueError("geometry phase fractions must each be positive.")
        if not 0.0 < total < 1.0:
            raise ValueError(
                "geometry phase fractions must sum to between zero and one."
            )
        if not 1.0 <= float(self.big_elongation) <= 4.0:
            raise ValueError("geometry.big_elongation must be between 1.0 and 4.0.")
        object.__setattr__(self, "big_fraction", float(self.big_fraction))
        object.__setattr__(self, "small_fraction", float(self.small_fraction))
        object.__setattr__(
            self,
            "big_elongation",
            float(self.big_elongation),
        )


@dataclass(frozen=True)
class SimulationConfig:
    output: OutputConfig
    geometry: GeometryConfig


_T = TypeVar("_T")
_SECTIONS = {
    "output": OutputConfig,
    "geometry": GeometryConfig,
}


def load_config(path: str | Path) -> SimulationConfig:
    path = Path(path).resolve()
    values = load_yaml(path, label="simulation config")
    names = set(values)
    expected = set(_SECTIONS)
    if names != expected:
        missing = sorted(expected - names)
        extra = sorted(names - expected)
        parts = []
        if missing:
            parts.append(f"missing sections: {', '.join(missing)}")
        if extra:
            parts.append(f"unknown sections: {', '.join(extra)}")
        raise ValueError(f"simulation config has {'; '.join(parts)}.")

    output = _build_section(OutputConfig, values["output"], "output")
    output = replace(
        output,
        data_dir=_resolve_dir(
            output.data_dir,
            path.parent,
            "output.data_dir",
        ),
    )
    return SimulationConfig(
        output=output,
        geometry=_build_section(GeometryConfig, values["geometry"], "geometry"),
    )


def _build_section(cls: type[_T], value: object, name: str) -> _T:
    if not isinstance(value, dict):
        raise TypeError(f"simulation config section {name} must be a mapping.")
    try:
        return cls(**value)
    except TypeError as exc:
        raise ValueError(f"simulation config section {name} is invalid: {exc}") from exc


def _resolve_dir(value: object, root: Path, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a non-empty path.")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()
