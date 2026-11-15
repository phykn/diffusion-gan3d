from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
)

from src.axes import PLANES

Plane = Literal["xy", "xz", "yz"]
Probability = Annotated[float, Field(ge=0, le=1)]
BelowOne = Annotated[float, Field(ge=0, lt=1)]
Folders = Annotated[tuple[str, ...], Field(min_length=1)]


class Settings(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", strict=True, allow_inf_nan=False
    )


class GenerationSettings(Settings):
    guidance: float
    anchor_strength: NonNegativeFloat
    overlap: NonNegativeInt


class Resolution(Settings):
    crop_size: PositiveInt
    lo_res_size: PositiveInt
    sr_scale: int = Field(ge=2)


class Structure(Settings):
    isotropic: bool
    depth_gradient: bool
    mirror_symmetric: bool


class Tune(Settings):
    lr: dict | None = None
    sr: dict | None = None


class Sample(Settings):
    images: dict[Plane, str | Folders] = Field(min_length=1)
    resolution: Resolution
    structure: Structure
    tune: Tune | None = None
    num_phases: int | None = Field(None, ge=1, le=256)


class Data(Settings):
    images: dict[Plane, Folders] = Field(min_length=1)
    num_phases: int = Field(ge=1, le=256)
    crop_size: PositiveInt
    lo_res_size: PositiveInt
    hi_res_size: PositiveInt
    height_extent: PositiveInt | None = None
    depth_profile: dict | None = None


class Generator(Settings):
    channels: tuple[Annotated[int, Field(ge=2)], ...] = Field(
        (16, 32, 64, 64), min_length=1
    )
    embedding_channels: PositiveInt = 128
    latent_channels: PositiveInt = 64


class Critic(Settings):
    channels: tuple[PositiveInt, ...] = (32, 64, 128, 256)
    plane_groups: tuple[tuple[Plane, ...], ...] = (("xy", "xz", "yz"),)
    pyramid_min_size: int = Field(16, ge=2)


class SRCritic(Critic):
    channels: tuple[PositiveInt, ...] = (32, 64, 128)


class DiffusionSchedule(Settings):
    num_steps: PositiveInt = 10
    beta_min: NonNegativeFloat = 0.1
    beta_max: NonNegativeFloat = 20.0


class Model(Settings):
    generator: Generator = Generator()
    critic: Critic = Critic()
    diffusion: DiffusionSchedule = DiffusionSchedule()
    gradient_checkpointing: bool = False


class SRModel(Model):
    critic: SRCritic = SRCritic()


class Augmentation(Settings):
    probability: Probability = 0.5


class RealTransition(Settings):
    max_gap: PositiveInt = 8
    ramp_steps: NonNegativeInt = 5000
    weight: NonNegativeFloat = 0.1


class MirrorBalance(Settings):
    weight: NonNegativeFloat = 0.0
    max_gap: PositiveInt = 4
    tolerance: Probability = 0.02


class GroupStatistics(Settings):
    weight: NonNegativeFloat = 0.1
    max_gap: PositiveInt = 8
    tolerance: Probability = 0.01
    ramp_steps: NonNegativeInt = 5000


class SRGroupStatistics(GroupStatistics):
    ramp_steps: NonNegativeInt = 2500


class LRLoss(Settings):
    r1_weight: NonNegativeFloat = 0.5
    r2_weight: NonNegativeFloat = 0.1
    r1_every_steps: PositiveInt = 8
    volume_fraction_weight: NonNegativeFloat = 1.0
    real_transition: RealTransition = RealTransition()
    mirror_balance: MirrorBalance = MirrorBalance()
    group_statistics: GroupStatistics = GroupStatistics()
    spatial_profile_weight: NonNegativeFloat = 1.0


class SRLoss(Settings):
    r1_weight: NonNegativeFloat = 0.5
    r2_weight: NonNegativeFloat = 0.1
    r1_every_steps: PositiveInt = 8
    downsample_consistency_weight: NonNegativeFloat = 10.0
    mirror_balance: MirrorBalance = MirrorBalance()
    group_statistics: SRGroupStatistics = SRGroupStatistics()
    downsample_mse_tolerance: NonNegativeFloat = 0.005


class Optim(Settings):
    generator_lr: NonNegativeFloat = 0.00016
    critic_lr: NonNegativeFloat = 0.0001
    ema_decay: BelowOne = 0.999
    adam_betas: tuple[BelowOne, BelowOne] = (0.5, 0.9)


class BurnIn(Settings):
    steps: NonNegativeInt = 5000
    regularization_scale: NonNegativeFloat = 0.1
    lr_scale: NonNegativeFloat = 0.25
    beta2: BelowOne = 0.99


class SRBurnIn(BurnIn):
    steps: NonNegativeInt = 2500


class LRBank(Settings):
    size: PositiveInt = 8
    guidance: float = 0.9
    refresh_every_steps: NonNegativeInt = 500
    flips: bool = True


class Train(Settings):
    total_steps: PositiveInt = 10000
    real_batch_size: PositiveInt = 8
    volume_batch_size: PositiveInt = 1
    slice_pairs_per_plane: PositiveInt = 16
    weights_every_steps: PositiveInt = 500
    archive_every_steps: PositiveInt = 2500
    structure_every_steps: NonNegativeInt = 250
    mixed_precision: bool = True
    seed: NonNegativeInt = 0


class SRTrain(Train):
    total_steps: PositiveInt = 5000
    slice_pairs_per_plane: PositiveInt = 8


class SpatialProfile(Settings):
    enabled: bool = False
    num_bins: PositiveInt = 16


class LRConditioning(Settings):
    height_enabled: bool = False
    spatial_profile: SpatialProfile = SpatialProfile()
    cfg_dropout_probability: Probability = 0.1


class SRConditioning(Settings):
    height_enabled: bool = False
    coarse_corruption_probability: Probability = 0.5
    coarse_corruption_strength: Probability = 0.2


class Selection(Settings):
    samples: PositiveInt = 8


class Source(Settings):
    lr_run: str | None = None
    lr_step: int | None = None


class LRConfig(Settings):
    stage: str = "lr"
    nickname: str = ""
    data: Data
    model: Model = Model()
    augmentation: Augmentation = Augmentation()
    loss: LRLoss = LRLoss()
    optim: Optim = Optim()
    burn_in: BurnIn = BurnIn()
    train: Train = Train()
    conditioning: LRConditioning = LRConditioning()
    selection: Selection = Selection()

    @property
    def patch_size(self):
        return self.data.lo_res_size

    @property
    def profile_enabled(self):
        return self.conditioning.spatial_profile.enabled


class SRConfig(Settings):
    stage: str = "sr"
    nickname: str = ""
    data: Data
    model: SRModel = SRModel()
    augmentation: Augmentation = Augmentation()
    loss: SRLoss = SRLoss()
    optim: Optim = Optim()
    burn_in: SRBurnIn = SRBurnIn()
    lr_bank: LRBank = LRBank()
    train: SRTrain = SRTrain()
    conditioning: SRConditioning = SRConditioning()
    selection: Selection = Selection()
    source: Source = Source()

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


def is_section(setting):
    kind = setting.annotation
    return isinstance(kind, type) and issubclass(kind, Settings)


def group_planes(config):
    groups = [
        tuple(sorted(PLANES.index(plane) for plane in group))
        for group in config.model.critic.plane_groups
    ]
    return {
        "_".join(PLANES[index] for index in group): group for group in sorted(groups)
    }
