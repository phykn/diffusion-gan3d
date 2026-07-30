from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TypeVar

from ..misc import load_mapping, require_int, require_number


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
        _positive_tuple("model.channel_multipliers", self.channel_multipliers)
        _positive_tuple("model.critic_channels", self.critic_channels)


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
        beta1 = require_number("optim.beta1", self.beta1, minimum=0.0, maximum=1.0)
        beta2 = require_number("optim.beta2", self.beta2, minimum=0.0, maximum=1.0)
        if beta1 >= beta2:
            raise ValueError("optim.beta1 must be smaller than optim.beta2.")
        require_number("optim.r1_gamma", self.r1_gamma, minimum=0.0)
        require_int("optim.r1_interval", self.r1_interval, minimum=1)


@dataclass(frozen=True)
class TrainingConfig:
    steps: int
    volume_batch_size: int
    slices_per_axis: int
    mixed_precision: bool
    ema_decay: float
    save_every_steps: int
    checkpoint: str | Path | None = None

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
class OutputConfig:
    run_root: str | Path


@dataclass(frozen=True)
class TrainConfig:
    data: DataConfig
    model: ModelConfig
    diffusion: DiffusionConfig
    optim: OptimConfig
    train: TrainingConfig
    output: OutputConfig

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

    def resume_signature(self) -> dict[str, object]:
        values = self.as_dict()
        values["train"] = dict(values["train"])
        values["train"].pop("steps")
        values["train"].pop("save_every_steps")
        values["train"].pop("checkpoint")
        values.pop("output")
        return values


_T = TypeVar("_T")
_SECTIONS = {
    "data": DataConfig,
    "model": ModelConfig,
    "diffusion": DiffusionConfig,
    "optim": OptimConfig,
    "train": TrainingConfig,
    "output": OutputConfig,
}


def load_train_config(path: str | Path) -> TrainConfig:
    config_path = Path(path).resolve()
    values = load_mapping(config_path, label="training config")
    if set(values) != set(_SECTIONS):
        missing = sorted(set(_SECTIONS) - set(values))
        extra = sorted(set(values) - set(_SECTIONS))
        details = []
        if missing:
            details.append(f"missing sections: {', '.join(missing)}")
        if extra:
            details.append(f"unknown sections: {', '.join(extra)}")
        raise ValueError(f"training config has {'; '.join(details)}.")

    data = _make(DataConfig, values["data"], "data")
    data = replace(data, folder=_resolve_folders(data.folder, config_path.parent))
    model = _make_model(values["model"])
    train = _make(TrainingConfig, values["train"], "train")
    train = replace(
        train,
        checkpoint=_resolve_optional_path(train.checkpoint, config_path.parent),
    )
    output = _make(OutputConfig, values["output"], "output")
    output = replace(
        output,
        run_root=_resolve_path(output.run_root, config_path.parent, "output.run_root"),
    )
    return TrainConfig(
        data=data,
        model=model,
        diffusion=_make(DiffusionConfig, values["diffusion"], "diffusion"),
        optim=_make(OptimConfig, values["optim"], "optim"),
        train=train,
        output=output,
    )


def _make(cls: type[_T], value: object, name: str) -> _T:
    if not isinstance(value, dict):
        raise TypeError(f"training config section {name} must be a mapping.")
    try:
        return cls(**value)
    except TypeError as exc:
        raise ValueError(f"training config section {name} is invalid: {exc}") from exc


def _make_model(value: object) -> ModelConfig:
    if not isinstance(value, dict):
        raise TypeError("training config section model must be a mapping.")
    values = dict(value)
    for name in ("channel_multipliers", "critic_channels"):
        item = values.get(name)
        if not isinstance(item, list):
            raise TypeError(f"model.{name} must be a list.")
        values[name] = tuple(item)
    return _make(ModelConfig, values, "model")


def _resolve_folders(value: object, root: Path) -> dict[int, Path]:
    if not isinstance(value, dict) or set(value) != {0, 1, 2}:
        raise ValueError("data.folder must contain exactly axes 0, 1, and 2.")
    return {
        axis: _resolve_path(path, root, f"data.folder.{axis}")
        for axis, path in value.items()
    }


def _resolve_optional_path(value: object, root: Path) -> Path | None:
    if value is None:
        return None
    return _resolve_path(value, root, "train.checkpoint")


def _resolve_path(value: object, root: Path, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a non-empty path.")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _positive_tuple(name: str, values: object) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty list.")
    for value in values:
        require_int(name, value, minimum=1)
