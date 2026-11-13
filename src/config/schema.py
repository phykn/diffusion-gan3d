from collections.abc import Mapping
from dataclasses import dataclass, field

from src.axes import PLANES
from src.config.ranges import fraction, integer, number, probability


@dataclass(frozen=True, kw_only=True)
class GenerationSettings:
    guidance: float
    anchor_strength: float = number()
    overlap: int = integer(minimum=0)


@dataclass(frozen=True, kw_only=True)
class Data:
    images: Mapping[str, tuple[str, ...]]
    num_phases: int
    crop_size: int = integer()
    lo_res_size: int = integer()
    hi_res_size: int = integer()
    height_extent: int | None = integer(None)
    depth_profile: Mapping | None = None


@dataclass(frozen=True, kw_only=True)
class Generator:
    channels: tuple[int, ...] = (16, 32, 64, 64)
    embedding_channels: int = integer(128)
    latent_channels: int = integer(64)


@dataclass(frozen=True, kw_only=True)
class Critic:
    channels: tuple[int, ...] = (32, 64, 128, 256)
    plane_groups: tuple[tuple[str, ...], ...] = (("xy", "xz", "yz"),)
    pyramid_min_size: int = integer(16, minimum=2)


@dataclass(frozen=True, kw_only=True)
class SRCritic(Critic):
    channels: tuple[int, ...] = (32, 64, 128)


@dataclass(frozen=True, kw_only=True)
class DiffusionSchedule:
    num_steps: int = integer(10)
    beta_min: float = number(0.1)
    beta_max: float = number(20.0)


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
    probability: float = probability(0.5)


@dataclass(frozen=True, kw_only=True)
class RealTransition:
    max_gap: int = integer(8)
    ramp_steps: int = integer(5000, minimum=0)
    weight: float = number(0.1)


@dataclass(frozen=True, kw_only=True)
class MirrorBalance:
    weight: float = number(0.0)
    max_gap: int = integer(4)
    tolerance: float = probability(0.02)


@dataclass(frozen=True, kw_only=True)
class GroupStatistics:
    weight: float = number(0.1)
    max_gap: int = integer(8)
    tolerance: float = probability(0.01)
    ramp_steps: int = integer(5000, minimum=0)


@dataclass(frozen=True, kw_only=True)
class SRGroupStatistics(GroupStatistics):
    ramp_steps: int = integer(2500, minimum=0)


@dataclass(frozen=True, kw_only=True)
class LRLoss:
    r1_weight: float = number(0.5)
    r2_weight: float = number(0.1)
    r1_every_steps: int = integer(8)
    volume_fraction_weight: float = number(1.0)
    real_transition: RealTransition = field(default_factory=RealTransition)
    mirror_balance: MirrorBalance = field(default_factory=MirrorBalance)
    group_statistics: GroupStatistics = field(default_factory=GroupStatistics)
    spatial_profile_weight: float = number(1.0)


@dataclass(frozen=True, kw_only=True)
class SRLoss:
    r1_weight: float = number(0.5)
    r2_weight: float = number(0.1)
    r1_every_steps: int = integer(8)
    downsample_consistency_weight: float = number(10.0)
    group_statistics: SRGroupStatistics = field(default_factory=SRGroupStatistics)
    downsample_mse_tolerance: float = number(0.005)


@dataclass(frozen=True, kw_only=True)
class Optim:
    generator_lr: float = number(0.00016)
    critic_lr: float = number(0.0001)
    ema_decay: float = fraction(0.999)
    adam_betas: tuple[float, float] = (0.5, 0.9)


@dataclass(frozen=True, kw_only=True)
class BurnIn:
    steps: int = integer(5000, minimum=0)
    regularization_scale: float = number(0.1)
    lr_scale: float = number(0.25)
    beta2: float = fraction(0.99)


@dataclass(frozen=True, kw_only=True)
class SRBurnIn(BurnIn):
    steps: int = integer(2500, minimum=0)


@dataclass(frozen=True, kw_only=True)
class LRBank:
    size: int = integer(8)
    guidance: float = 0.9
    refresh_every_steps: int = integer(500, minimum=0)
    flips: bool = True


@dataclass(frozen=True, kw_only=True)
class Train:
    total_steps: int = integer(10000)
    real_batch_size: int = integer(8)
    volume_batch_size: int = integer(1)
    slice_pairs_per_plane: int = integer(16)
    weights_every_steps: int = integer(500)
    archive_every_steps: int = integer(2500)
    structure_every_steps: int = integer(250, minimum=0)
    mixed_precision: bool = True
    seed: int = integer(0, minimum=0)


@dataclass(frozen=True, kw_only=True)
class SRTrain(Train):
    total_steps: int = integer(5000)
    slice_pairs_per_plane: int = integer(8)


@dataclass(frozen=True, kw_only=True)
class SpatialProfile:
    enabled: bool = False
    num_bins: int = integer(16)


@dataclass(frozen=True, kw_only=True)
class LRConditioning:
    height_enabled: bool = False
    spatial_profile: SpatialProfile = field(default_factory=SpatialProfile)
    cfg_dropout_probability: float = probability(0.1)


@dataclass(frozen=True, kw_only=True)
class SRConditioning:
    height_enabled: bool = False
    coarse_corruption_probability: float = probability(0.5)
    coarse_corruption_strength: float = probability(0.2)


@dataclass(frozen=True, kw_only=True)
class Selection:
    samples: int = integer(8)


@dataclass(frozen=True, kw_only=True)
class Source:
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
    def patch_size(self):
        return self.data.lo_res_size

    @property
    def profile_enabled(self):
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
    def patch_size(self):
        return self.data.hi_res_size

    @property
    def profile_enabled(self):
        return False

    @property
    def scale_factor(self):
        return self.data.hi_res_size // self.data.lo_res_size


CONFIG_TYPES = {"lr": LRConfig, "sr": SRConfig}
STAGES = tuple(CONFIG_TYPES)


def group_planes(config):
    groups = [
        tuple(sorted(PLANES.index(plane) for plane in group))
        for group in config.model.critic.plane_groups
    ]
    return {
        "_".join(PLANES[index] for index in group): group for group in sorted(groups)
    }
