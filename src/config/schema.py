import math
import types
import typing
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields, is_dataclass

from src.phase import validate_num_phases
from src.plane import PLANES

STAGES = ("lr", "sr")


def _number(default, *, minimum=0.0, maximum=None, exclusive=False):
    return field(
        default=default,
        metadata={"minimum": minimum, "maximum": maximum, "exclusive": exclusive},
    )


def _integer(default, *, minimum=1):
    return field(default=default, metadata={"minimum": minimum})


def _probability(default):
    return _number(default, maximum=1.0)


def _fraction(default):
    return _number(default, maximum=1.0, exclusive=True)


@dataclass(frozen=True, kw_only=True)
class Data:
    images: Mapping[str, tuple[str, ...]]
    num_phases: int
    crop_size: int = _integer(MISSING)
    lo_res_size: int = _integer(MISSING)
    hi_res_size: int = _integer(MISSING)
    height_extent: int | None = None
    depth_profile: Mapping | None = None


@dataclass(frozen=True, kw_only=True)
class Generator:
    channels: tuple[int, ...] = (16, 32, 64, 64)
    embedding_channels: int = _integer(128)
    latent_channels: int = _integer(64)


@dataclass(frozen=True, kw_only=True)
class Critic:
    channels: tuple[int, ...] = (32, 64, 128, 256)
    plane_groups: tuple[tuple[str, ...], ...] = (("xy", "xz", "yz"),)
    pyramid_min_size: int = _integer(16, minimum=2)


@dataclass(frozen=True, kw_only=True)
class SRCritic(Critic):
    channels: tuple[int, ...] = (32, 64, 128)


@dataclass(frozen=True, kw_only=True)
class DiffusionSchedule:
    num_steps: int = _integer(10)
    beta_min: float = _number(0.1)
    beta_max: float = _number(20.0)


@dataclass(frozen=True, kw_only=True)
class Model:
    generator: Generator = field(default_factory=Generator)
    critic: Critic = field(default_factory=Critic)
    diffusion: DiffusionSchedule = field(default_factory=DiffusionSchedule)
    gradient_checkpointing: bool = False


@dataclass(frozen=True, kw_only=True)
class SRModel(Model):
    critic: SRCritic = field(default_factory=SRCritic)


@dataclass(frozen=True, kw_only=True)
class Augmentation:
    probability: float = _probability(0.5)


@dataclass(frozen=True, kw_only=True)
class RealTransition:
    max_gap: int = _integer(8)
    ramp_steps: int = _integer(5000, minimum=0)
    weight: float = _number(0.1)
    start_step: int = _integer(0, minimum=0)


@dataclass(frozen=True, kw_only=True)
class MirrorBalance:
    weight: float = _number(0.0)
    max_gap: int = _integer(4)
    tolerance: float = _probability(0.02)


@dataclass(frozen=True, kw_only=True)
class GroupStatistics:
    weight: float = _number(0.1)
    max_gap: int = _integer(8)
    tolerance: float = _probability(0.01)
    start_step: int = _integer(0, minimum=0)
    ramp_steps: int = _integer(5000, minimum=0)


@dataclass(frozen=True, kw_only=True)
class SRGroupStatistics(GroupStatistics):
    ramp_steps: int = _integer(2500, minimum=0)


@dataclass(frozen=True, kw_only=True)
class LRLoss:
    r1_weight: float = _number(0.5)
    r2_weight: float = _number(0.1)
    r1_every_steps: int = _integer(8)
    volume_fraction_weight: float = _number(1.0)
    real_transition: RealTransition = field(default_factory=RealTransition)
    mirror_balance: MirrorBalance = field(default_factory=MirrorBalance)
    group_statistics: GroupStatistics = field(default_factory=GroupStatistics)
    spatial_profile_weight: float = _number(1.0)


@dataclass(frozen=True, kw_only=True)
class SRLoss:
    r1_weight: float = _number(0.5)
    r2_weight: float = _number(0.1)
    r1_every_steps: int = _integer(8)
    downsample_consistency_weight: float = _number(10.0)
    group_statistics: SRGroupStatistics = field(default_factory=SRGroupStatistics)
    downsample_mse_tolerance: float = _number(0.005)


@dataclass(frozen=True, kw_only=True)
class Optim:
    generator_lr: float = _number(0.00016)
    critic_lr: float = _number(0.0001)
    ema_decay: float = _fraction(0.999)
    adam_betas: tuple[float, float] = (0.5, 0.9)


@dataclass(frozen=True, kw_only=True)
class BurnIn:
    """R3GAN burn-in: lr, beta2, R1/R2 and EMA reach their end values here."""

    steps: int = _integer(5000, minimum=0)
    regularization_scale: float = _number(0.1)
    lr_scale: float = _number(0.25)
    beta2: float = _fraction(0.99)


@dataclass(frozen=True, kw_only=True)
class SRBurnIn(BurnIn):
    steps: int = _integer(2500, minimum=0)


@dataclass(frozen=True, kw_only=True)
class LRBank:
    samples_per_domain: int = _integer(8)
    guidance: float = 0.9
    refresh_every_steps: int = _integer(500, minimum=0)
    flips: bool = True


@dataclass(frozen=True, kw_only=True)
class Train:
    total_steps: int = _integer(10000)
    real_batch_size: int = _integer(8)
    volume_batch_size: int = _integer(1)
    slice_pairs_per_plane: int = _integer(16)
    weights_every_steps: int = _integer(500)
    archive_every_steps: int = _integer(2500)
    structure_every_steps: int = _integer(250, minimum=0)
    mixed_precision: bool = True
    seed: int = _integer(0, minimum=0)


@dataclass(frozen=True, kw_only=True)
class SRTrain(Train):
    total_steps: int = _integer(5000)
    slice_pairs_per_plane: int = _integer(8)


@dataclass(frozen=True, kw_only=True)
class SpatialProfile:
    enabled: bool = False
    num_bins: int = _integer(16)


@dataclass(frozen=True, kw_only=True)
class LRConditioning:
    height_enabled: bool = False
    spatial_profile: SpatialProfile = field(default_factory=SpatialProfile)
    cfg_dropout_probability: float = _probability(0.1)


@dataclass(frozen=True, kw_only=True)
class SRConditioning:
    height_enabled: bool = False
    coarse_corruption_probability: float = _probability(0.5)
    coarse_corruption_strength: float = _probability(0.2)


@dataclass(frozen=True, kw_only=True)
class Selection:
    enabled: bool = True
    samples: int = _integer(8)


@dataclass(frozen=True, kw_only=True)
class Source:
    """The frozen LR run an SR stage trains from (set when training starts)."""

    lr_run: str | None = None
    lr_step: int | None = None


@dataclass(frozen=True, kw_only=True)
class LRConfig:
    stage: str = "lr"
    nickname: str = ""
    data: Data
    model: Model = field(default_factory=Model)
    augmentation: Augmentation = field(default_factory=Augmentation)
    loss: LRLoss = field(default_factory=LRLoss)
    optim: Optim = field(default_factory=Optim)
    burn_in: BurnIn = field(default_factory=BurnIn)
    train: Train = field(default_factory=Train)
    conditioning: LRConditioning = field(default_factory=LRConditioning)
    selection: Selection = field(default_factory=Selection)

    @property
    def patch_size(self) -> int:
        return self.data.lo_res_size

    @property
    def profile_enabled(self) -> bool:
        return self.conditioning.spatial_profile.enabled


@dataclass(frozen=True, kw_only=True)
class SRConfig:
    stage: str = "sr"
    nickname: str = ""
    data: Data
    model: SRModel = field(default_factory=SRModel)
    augmentation: Augmentation = field(default_factory=Augmentation)
    loss: SRLoss = field(default_factory=SRLoss)
    optim: Optim = field(default_factory=Optim)
    burn_in: SRBurnIn = field(default_factory=SRBurnIn)
    lr_bank: LRBank = field(default_factory=LRBank)
    train: SRTrain = field(default_factory=SRTrain)
    conditioning: SRConditioning = field(default_factory=SRConditioning)
    selection: Selection = field(default_factory=Selection)
    source: Source = field(default_factory=Source)

    @property
    def patch_size(self) -> int:
        return self.data.hi_res_size

    @property
    def profile_enabled(self) -> bool:
        return False

    @property
    def scale_factor(self) -> int:
        return self.data.hi_res_size // self.data.lo_res_size


StageConfig = LRConfig | SRConfig
CONFIG_TYPES = {"lr": LRConfig, "sr": SRConfig}


def build_config(values: Mapping) -> StageConfig:
    """Make a checked stage config; missing keys take the defaults above."""
    if not isinstance(values, Mapping):
        raise ValueError("config must be a mapping.")
    stage = values.get("stage")
    if stage not in CONFIG_TYPES:
        raise ValueError(f"stage must be one of {', '.join(STAGES)}.")
    config = _build(CONFIG_TYPES[stage], values, "")
    validate(config)
    return config


def to_dict(value) -> object:
    """Plain YAML-ready values: dicts, lists, numbers, strings and None."""
    if is_dataclass(value):
        return {item.name: to_dict(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {key: to_dict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_dict(item) for item in value]
    return value


def update(config: StageConfig, path: str, value) -> StageConfig:
    """Return config with one dotted key replaced and checked again."""
    values = to_dict(config)
    target = values
    *sections, key = path.split(".")
    for section in sections:
        target = target[section]
    if key not in target:
        raise ValueError(f"unknown setting: {path}")
    target[key] = value
    return build_config(values)


def _build(cls, values, path):
    if not isinstance(values, Mapping):
        raise ValueError(f"{path or 'config'} must be a mapping.")
    known = {item.name: item for item in fields(cls)}
    for key in values:
        if key not in known:
            raise ValueError(f"unknown setting: {_join(path, key)}")
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for name, item in known.items():
        key = _join(path, name)
        if name in values:
            kwargs[name] = _convert(hints[name], values[name], key)
            _check_range(item, kwargs[name], key)
        elif item.default is MISSING and item.default_factory is MISSING:
            raise ValueError(f"missing setting: {key}")
    return cls(**kwargs)


def _convert(kind, value, path):
    origin = typing.get_origin(kind)
    if origin in (typing.Union, types.UnionType):
        options = typing.get_args(kind)
        if value is None and type(None) in options:
            return None
        (kind,) = (option for option in options if option is not type(None))
        return _convert(kind, value, path)
    if is_dataclass(kind):
        return _build(kind, value, path)
    if kind is bool:
        # YAML reads a quoted "false" as text, which would act as true.
        if type(value) is not bool:
            raise ValueError(f"{path} must be true or false, got {value!r}.")
        return value
    if kind is int:
        if type(value) is not int:
            raise ValueError(f"{path} must be an integer, got {value!r}.")
        return value
    if kind is float:
        if type(value) not in (int, float):
            # PyYAML reads exponents without a decimal point, such as 1e-4, as text.
            hint = " (write 1.0e-4, not 1e-4)" if isinstance(value, str) else ""
            raise ValueError(f"{path} must be a number, got {value!r}{hint}.")
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite.")
        return float(value)
    if kind is str:
        if not isinstance(value, str):
            raise ValueError(f"{path} must be text, got {value!r}.")
        return value
    if origin is tuple:
        args = typing.get_args(kind)
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{path} must be a list.")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                _convert(args[0], item, f"{path}[{index}]")
                for index, item in enumerate(value)
            )
        if len(value) != len(args):
            raise ValueError(f"{path} must list {len(args)} values.")
        return tuple(
            _convert(arg, item, f"{path}[{index}]")
            for index, (arg, item) in enumerate(zip(args, value, strict=True))
        )
    if origin is Mapping or kind is Mapping:
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must be a mapping.")
        args = typing.get_args(kind)
        if not args:
            return types.MappingProxyType(_freeze(value))
        return types.MappingProxyType(
            {
                _convert(args[0], key, path): _convert(args[1], item, _join(path, key))
                for key, item in value.items()
            }
        )
    raise TypeError(f"unsupported setting type for {path}: {kind}")


def _freeze(value):
    if isinstance(value, Mapping):
        return types.MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _check_range(item, value, path):
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    minimum = item.metadata.get("minimum")
    maximum = item.metadata.get("maximum")
    exclusive = item.metadata.get("exclusive", False)
    if minimum is not None and value < minimum:
        if type(value) is int:
            bound = {0: "a non-negative integer", 1: "a positive integer"}.get(
                minimum, f"an integer of at least {minimum}"
            )
            raise ValueError(f"{path} must be {bound}.")
        raise ValueError(f"{path} must be at least {minimum}.")
    if maximum is not None and (value >= maximum if exclusive else value > maximum):
        end = ")" if exclusive else "]"
        raise ValueError(f"{path} must be in [{minimum}, {maximum}{end}.")


def _join(path, key):
    return f"{path}.{key}" if path else str(key)


def validate(config: StageConfig) -> None:
    """Rules that relate several settings of one stage."""
    nickname = config.nickname.strip()
    if nickname != config.nickname or (
        nickname.endswith(".")
        or any(char in '<>:"/\\|?*' or ord(char) < 32 for char in nickname)
    ):
        raise ValueError("nickname must be a valid folder-name component.")
    data = config.data
    validate_num_phases(data.num_phases)
    if set(data.images) - set(PLANES) or not data.images:
        raise ValueError("data.images must map xy, xz or yz to folders.")
    for plane, folders in data.images.items():
        if not folders:
            raise ValueError(f"data.images.{plane} must list at least one folder.")
        if len(set(folders)) != len(folders):
            raise ValueError(f"data.images.{plane} must not repeat a folder.")
    if data.hi_res_size < data.lo_res_size:
        raise ValueError("data.hi_res_size must be at least data.lo_res_size.")
    if isinstance(config, SRConfig) and data.hi_res_size % data.lo_res_size:
        raise ValueError("data.hi_res_size must be a multiple of data.lo_res_size.")
    if data.height_extent is not None and data.height_extent < 1:
        raise ValueError("data.height_extent must be a positive integer or null.")
    model = config.model
    channels = model.generator.channels
    if not channels or any(value < 2 for value in channels):
        raise ValueError("model.generator.channels must list integers of at least 2.")
    if any(value % channels[0] for value in channels):
        raise ValueError(
            "model.generator.channels must be integer multiples of its first value."
        )
    if config.patch_size % 2 ** (len(channels) - 1):
        raise ValueError(
            "the patch size must be divisible by the generator's downsampling."
        )
    critic = model.critic.channels
    if len(critic) < 2 or any(value < 1 for value in critic):
        raise ValueError("model.critic.channels must list at least two levels.")
    _check_plane_groups(model.critic.plane_groups, set(data.images))
    if model.diffusion.beta_min >= model.diffusion.beta_max:
        raise ValueError("model.diffusion.beta_min must be below beta_max.")
    for beta in config.optim.adam_betas:
        if not 0 <= beta < 1:
            raise ValueError("optim.adam_betas must contain two numbers in [0, 1).")
    if (
        config.conditioning.height_enabled
        and config.train.volume_batch_size > config.train.real_batch_size
    ):
        raise ValueError(
            "height conditioning requires train.real_batch_size >= "
            "train.volume_batch_size."
        )
    if config.conditioning.height_enabled and not set(data.images) - {"xy"}:
        raise ValueError("height conditioning requires xz or yz images.")
    if isinstance(config, LRConfig):
        profile = config.conditioning.spatial_profile
        if profile.enabled and not config.conditioning.height_enabled:
            raise ValueError(
                "conditioning.spatial_profile requires conditioning.height_enabled."
            )


def _check_plane_groups(groups, planes):
    seen = []
    for group in groups:
        if not group:
            raise ValueError("each model.critic.plane_groups entry must be non-empty.")
        for plane in group:
            if plane not in PLANES:
                raise ValueError("model.critic.plane_groups must use xy, xz and yz.")
            seen.append(plane)
    if len(seen) != len(set(seen)):
        raise ValueError("each plane must occur in one critic group only.")
    if set(seen) != planes:
        raise ValueError(
            "model.critic.plane_groups must cover exactly the planes in data.images."
        )


def plane_groups(config: StageConfig) -> dict[str, tuple[int, ...]]:
    """Critic name (planes joined by _) to the plane axes it judges."""
    groups = {}
    for group in config.model.critic.plane_groups:
        axes = tuple(sorted(PLANES.index(plane) for plane in group))
        groups["_".join(PLANES[axis] for axis in axes)] = axes
    return dict(sorted(groups.items(), key=lambda item: item[1]))
