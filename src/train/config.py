from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TypeVar

from ..utils import load_yaml, require_int, require_number


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
        if len(self.critic_channels) < 2:
            raise ValueError("model.critic_channels must contain at least two levels.")


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
    dropout: float = 1.0
    loss_weight: float = 0.0
    max_planes: int = 1

    def __post_init__(self) -> None:
        dropout = require_number(
            "anchor.dropout",
            self.dropout,
            minimum=0.0,
            maximum=1.0,
        )
        weight = require_number(
            "anchor.loss_weight",
            self.loss_weight,
            minimum=0.0,
        )
        require_int("anchor.max_planes", self.max_planes, minimum=1)
        if self.max_planes > 3:
            raise ValueError("anchor.max_planes must not exceed 3.")
        if (dropout < 1.0) != (weight > 0.0):
            raise ValueError(
                "anchor.loss_weight must be positive exactly when "
                "anchor.dropout is less than 1."
            )

    @property
    def enabled(self) -> bool:
        return self.dropout < 1.0


@dataclass(frozen=True)
class FractionConfig:
    loss_weight: float
    dropout: float

    def __post_init__(self) -> None:
        require_number(
            "fraction.loss_weight",
            self.loss_weight,
            minimum=0.0,
        )
        require_number(
            "fraction.dropout",
            self.dropout,
            minimum=0.0,
            maximum=1.0,
        )


@dataclass(frozen=True)
class OptimConfig:
    generator_lr: float
    critic_lr: float
    beta1: float
    beta2: float
    r1_gamma: float
    r1_interval: int
    critic_local_weight: float = 0.5

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
        require_number(
            "optim.critic_local_weight",
            self.critic_local_weight,
            minimum=0.0,
        )


@dataclass(frozen=True)
class CheckpointConfig:
    model: Path | None = None
    critic_0: Path | None = None
    critic_1: Path | None = None
    critic_2: Path | None = None


@dataclass(frozen=True)
class LoopConfig:
    steps: int
    volume_batch_size: int
    slices_per_axis: int
    mixed_precision: bool
    ema_decay: float
    save_every_steps: int
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

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
        if not isinstance(self.checkpoint, CheckpointConfig):
            raise TypeError("train.checkpoint must contain checkpoint paths.")


@dataclass(frozen=True)
class TrainConfig:
    data: DataConfig
    model: ModelConfig
    diffusion: DiffusionConfig
    anchor: AnchorConfig
    fraction: FractionConfig
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
    "fraction": FractionConfig,
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
    train = _build_loop_config(values["train"], config_path.parent)
    return TrainConfig(
        data=data,
        model=model,
        diffusion=_build_section(
            DiffusionConfig,
            values["diffusion"],
            "diffusion",
        ),
        anchor=_build_section(AnchorConfig, values["anchor"], "anchor"),
        fraction=_build_section(
            FractionConfig,
            values["fraction"],
            "fraction",
        ),
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


def _build_loop_config(value: object, root: Path) -> LoopConfig:
    if not isinstance(value, dict):
        raise TypeError("training config section train must be a mapping.")
    values = dict(value)
    checkpoint = values.get("checkpoint")
    if checkpoint is None:
        values["checkpoint"] = CheckpointConfig()
    elif isinstance(checkpoint, dict):
        expected = {"model", "critic_0", "critic_1", "critic_2"}
        if set(checkpoint) != expected:
            raise ValueError(
                "train.checkpoint must contain model, critic_0, critic_1, and critic_2."
            )
        paths = {
            name: (
                None
                if path is None
                else _resolve_path(path, root, f"train.checkpoint.{name}")
            )
            for name, path in checkpoint.items()
        }
        values["checkpoint"] = _build_section(
            CheckpointConfig,
            paths,
            "train.checkpoint",
        )
    else:
        raise TypeError("train.checkpoint must be a mapping or null.")
    return _build_section(LoopConfig, values, "train")


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
