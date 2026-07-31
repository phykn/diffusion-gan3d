from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TypeVar

from ..config import load_yaml, require_int, require_number


@dataclass(frozen=True)
class DataConfig:
    folder: dict[int, str | Path]
    crop_size: int
    patch_size: int
    num_phases: int
    batch_size: int
    num_workers: int = 0

    def __post_init__(self) -> None:
        require_int("data.crop_size", self.crop_size, minimum=1)
        require_int("data.patch_size", self.patch_size, minimum=8)
        require_int("data.num_phases", self.num_phases, minimum=2)
        if self.num_phases > 256:
            raise ValueError("data.num_phases must not exceed 256.")
        require_int("data.batch_size", self.batch_size, minimum=1)
        require_int("data.num_workers", self.num_workers, minimum=0)
        if self.crop_size < self.patch_size:
            raise ValueError("data.crop_size must be at least data.patch_size.")


@dataclass(frozen=True)
class ModelConfig:
    base_channels: int
    channel_multipliers: tuple[int, ...]
    embedding_channels: int
    latent_channels: int
    critic_channels: tuple[int, ...]
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        require_int("model.base_channels", self.base_channels, minimum=4)
        require_int("model.embedding_channels", self.embedding_channels, minimum=4)
        require_int("model.latent_channels", self.latent_channels, minimum=1)
        if not isinstance(self.gradient_checkpointing, bool):
            raise TypeError("model.gradient_checkpointing must be a boolean.")
        _require_ints("model.channel_multipliers", self.channel_multipliers)
        _require_ints("model.critic_channels", self.critic_channels, minimum=2)


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int
    beta_min: float
    beta_max: float

    def __post_init__(self) -> None:
        require_int("diffusion.timesteps", self.timesteps, minimum=1)
        beta_min = require_number("diffusion.beta_min", self.beta_min, minimum=0.0)
        beta_max = require_number("diffusion.beta_max", self.beta_max, minimum=0.0)
        if beta_min <= 0.0 or beta_min >= beta_max:
            raise ValueError(
                "diffusion beta values must satisfy 0 < beta_min < beta_max."
            )


@dataclass(frozen=True)
class AnchorConfig:
    probability: float = 0.0
    loss_weight: float = 0.0

    def __post_init__(self) -> None:
        probability = require_number(
            "anchor.probability",
            self.probability,
            minimum=0.0,
            maximum=1.0,
        )
        weight = require_number(
            "anchor.loss_weight",
            self.loss_weight,
            minimum=0.0,
        )
        if (probability > 0.0) != (weight > 0.0):
            raise ValueError(
                "anchor.probability and anchor.loss_weight must both be "
                "positive or both be zero."
            )

    @property
    def enabled(self) -> bool:
        return self.probability > 0.0


@dataclass(frozen=True)
class OptimConfig:
    generator_lr: float
    critic_lr: float
    beta1: float
    beta2: float
    r1_gamma: float
    r1_interval: int

    def __post_init__(self) -> None:
        for name, value in (
            ("generator_lr", self.generator_lr),
            ("critic_lr", self.critic_lr),
        ):
            if require_number(f"optim.{name}", value, minimum=0.0) <= 0.0:
                raise ValueError(f"optim.{name} must be positive.")
        beta1 = require_number("optim.beta1", self.beta1, minimum=0.0)
        beta2 = require_number("optim.beta2", self.beta2, minimum=0.0)
        if not beta1 < beta2 < 1.0:
            raise ValueError("optim betas must satisfy 0 <= beta1 < beta2 < 1.")
        require_number("optim.r1_gamma", self.r1_gamma, minimum=0.0)
        require_int("optim.r1_interval", self.r1_interval, minimum=1)


@dataclass(frozen=True)
class LoopConfig:
    steps: int
    volume_batch_size: int
    slices_per_axis: int
    mixed_precision: bool
    ema_decay: float
    save_every_steps: int

    def __post_init__(self) -> None:
        require_int("train.steps", self.steps, minimum=1)
        require_int("train.volume_batch_size", self.volume_batch_size, minimum=1)
        require_int("train.slices_per_axis", self.slices_per_axis, minimum=1)
        require_int("train.save_every_steps", self.save_every_steps, minimum=1)
        if not isinstance(self.mixed_precision, bool):
            raise TypeError("train.mixed_precision must be a boolean.")
        ema = require_number("train.ema_decay", self.ema_decay)
        if not 0.0 < ema < 1.0:
            raise ValueError("train.ema_decay must be in (0, 1).")


@dataclass(frozen=True)
class TrainConfig:
    data: DataConfig
    model: ModelConfig
    diffusion: DiffusionConfig
    anchor: AnchorConfig
    optim: OptimConfig
    train: LoopConfig

    def __post_init__(self) -> None:
        generator_factor = 2 ** (len(self.model.channel_multipliers) - 1)
        critic_factor = 2 ** (len(self.model.critic_channels) - 1)
        required_factor = max(generator_factor, critic_factor)
        if self.data.patch_size % required_factor:
            raise ValueError(
                "data.patch_size must be divisible by the model downsample "
                f"factor ({required_factor})."
            )

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


_T = TypeVar("_T")
_SECTIONS = {
    "data": DataConfig,
    "model": ModelConfig,
    "diffusion": DiffusionConfig,
    "anchor": AnchorConfig,
    "optim": OptimConfig,
    "train": LoopConfig,
}


def load_config(path: str | Path) -> TrainConfig:
    config_path = Path(path).resolve()
    values = load_yaml(config_path, label="training config")
    if set(values) != set(_SECTIONS):
        missing = sorted(set(_SECTIONS) - set(values))
        extra = sorted(set(values) - set(_SECTIONS))
        details = []
        if missing:
            details.append(f"missing sections: {', '.join(missing)}")
        if extra:
            details.append(f"unknown sections: {', '.join(extra)}")
        raise ValueError(f"training config has {'; '.join(details)}.")

    data = _build_section(DataConfig, values["data"], "data")
    data = replace(data, folder=_resolve_folders(data.folder, config_path.parent))
    model = _build_model_config(values["model"])
    train = _build_section(LoopConfig, values["train"], "train")
    return TrainConfig(
        data=data,
        model=model,
        diffusion=_build_section(
            DiffusionConfig,
            values["diffusion"],
            "diffusion",
        ),
        anchor=_build_section(AnchorConfig, values["anchor"], "anchor"),
        optim=_build_section(OptimConfig, values["optim"], "optim"),
        train=train,
    )


def _build_section(cls: type[_T], value: object, name: str) -> _T:
    if not isinstance(value, dict):
        raise TypeError(f"training config section {name} must be a mapping.")
    try:
        return cls(**value)
    except TypeError as exc:
        raise ValueError(f"training config section {name} is invalid: {exc}") from exc


def _build_model_config(value: object) -> ModelConfig:
    if not isinstance(value, dict):
        raise TypeError("training config section model must be a mapping.")
    values = dict(value)
    for name in ("channel_multipliers", "critic_channels"):
        item = values.get(name)
        if not isinstance(item, list):
            raise TypeError(f"model.{name} must be a list.")
        values[name] = tuple(item)
    return _build_section(ModelConfig, values, "model")


def _resolve_folders(value: object, root: Path) -> dict[int, Path]:
    if not isinstance(value, dict) or set(value) != {0, 1, 2}:
        raise ValueError("data.folder must contain exactly axes 0, 1, and 2.")
    return {
        axis: _resolve_path(path, root, f"data.folder.{axis}")
        for axis, path in value.items()
    }


def _resolve_path(value: object, root: Path, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a non-empty path.")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _require_ints(
    name: str,
    values: object,
    *,
    minimum: int = 1,
) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty list.")
    for value in values:
        require_int(name, value, minimum=minimum)
