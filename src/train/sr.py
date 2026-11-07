"""SR stage: the frozen LR bank, coarse corruption and the SR objective.

SR learns to refine volumes the frozen LR model makes, so it trains on a bank
of them rather than on downsampled real data. Each volume takes its height and
depth profile from a random real side crop; one volume is replaced every
lr_bank.refresh_every_steps. The bank and the LR export travel in the
checkpoint, so a resume needs neither the LR run nor its images.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from tqdm import trange

from src.config.schema import SRConfig
from src.data.crops import CropSampler, phase_channels, resize_phases
from src.data.height import image_profile
from src.data.orientation import augment_volumes, bank_symmetries
from src.export import check_export, check_lr_source, export_config, export_payload
from src.plane import PLANES
from src.rng import rand, randint
from src.stats.labels import phase_profile
from src.train.core import StepInputs, Trainer
from src.train.losses.consistency import consistency_loss
from src.train.losses.group_statistics import group_statistics_loss
from src.train.lr import schedule_ramp


def corrupt_coarse(low, probability: float, strength: float, generator):
    """Replace random blocks of the LR input by random phases.

    Returns the corrupted input and the realized corruption level per volume.
    """
    if probability == 0 or strength == 0:
        return low, low.new_zeros(len(low))
    device = low.device
    count = len(low)
    shape = (count, 1, *(max(1, n // 4) for n in low.shape[2:]))
    active = rand((count, 1, 1, 1, 1), generator, device) < probability
    level = rand((count, 1, 1, 1, 1), generator, device) * strength * active
    blocks = (rand(shape, generator, device) < level).float()
    mask = F.interpolate(blocks, size=low.shape[2:], mode="nearest").bool() & active
    labels = randint(low.shape[1], (count, *shape[2:]), generator, device)
    replacement = F.interpolate(
        phase_channels(labels, low.shape[1]), size=low.shape[2:], mode="nearest"
    )
    corrupted = torch.where(mask, replacement, low)
    # Matching replacements do not count toward the realized level.
    level = (corrupted - low).abs().mean(dim=(2, 3, 4)).sum(1) * 0.5
    return corrupted, level


@dataclass
class Bank:
    """Frozen LR volumes and the real side-crop conditions they came from."""

    volumes: torch.Tensor  # [N, K, l, l, l] LR fractions on the CPU
    height_origins: torch.Tensor | None  # [N] source pixels
    height_extents: torch.Tensor | None  # [N] source pixels
    conditions: list[dict] = field(default_factory=list)
    next_index: int = 0

    def state_dict(self) -> dict:
        return {
            "volumes": self.volumes,
            "height_origins": self.height_origins,
            "height_extents": self.height_extents,
            "conditions": [dict(item) for item in self.conditions],
            "next_index": self.next_index,
        }

    @classmethod
    def from_state(cls, state: Mapping) -> "Bank":
        return cls(
            volumes=state["volumes"],
            height_origins=state["height_origins"],
            height_extents=state["height_extents"],
            conditions=[dict(item) for item in state["conditions"]],
            next_index=int(state["next_index"]),
        )


def crop_profile(image: torch.Tensor, origin, extent, crop_size, bins) -> dict:
    """The depth profile of one side crop as a constant profile over the
    whole source height, flat beyond the crop."""
    values = image_profile(image[None], bins)[0].T.tolist()  # [bins][K]
    start = origin / extent
    length = crop_size / extent
    points = [[0.0, values[0]]] if start > 0 else []
    points.extend(
        [start + index * length / len(values), value]
        for index, value in enumerate(values)
    )
    points.append([1.0, values[-1]])
    return {"axis": "z", "points": points, "interpolation": "constant"}


class SRTrainer(Trainer):
    config: SRConfig

    def __init__(
        self,
        config: SRConfig,
        device: torch.device,
        planes: dict,
        lr: Mapping,
        bank: Bank | None = None,
    ):
        super().__init__(config, device, planes)
        self.lr_export = check_export(lr, "lr")
        self.lr_config = export_config(self.lr_export)
        check_lr_source(self.lr_config, config)
        self.symmetries = bank_symmetries(config)
        self.height = config.conditioning.height_enabled
        # LR-grid side crops: the conditions of each bank volume.
        self.side_samplers = (
            {
                axis: CropSampler(
                    planes[PLANES[axis]],
                    config.data.crop_size,
                    config.data.lo_res_size,
                    self.num_phases,
                )
                for axis in self.axes
                if axis != 0
            }
            if self.height
            else {}
        )
        if self.height and not self.side_samplers:
            raise ValueError("SR height conditioning needs xz or yz images.")
        self._lr_api = None
        self.bank = self.create_bank() if bank is None else bank

    # Bank ------------------------------------------------------------------

    def lr_api(self):
        if self._lr_api is None:
            from src.api import LowResolutionAPI

            self._lr_api = LowResolutionAPI(self.lr_export, self.device)
        return self._lr_api

    def bank_condition(self) -> dict:
        """Height and profile of a random real side crop (none without height)."""
        if not self.height:
            return {}
        generator = self.random["bank"]
        axes = sorted(self.side_samplers)
        axis = axes[int(randint(len(axes), (), generator))]
        crop = self.side_samplers[axis].sample(1, generator)
        origin, extent = float(crop.origins[0]), float(crop.extents[0])
        record = {
            "plane": PLANES[axis],
            "height_origin": origin,
            "height_extent": extent,
        }
        spatial = self.lr_config.conditioning.spatial_profile
        if spatial.enabled:
            record["vf_profile"] = crop_profile(
                crop.images[0],
                origin,
                extent,
                self.config.data.crop_size,
                spatial.num_bins,
            )
        return record

    def bank_volume(self, record: Mapping) -> torch.Tensor:
        seed = int(randint(2**62, (), self.random["bank"]))
        options = {
            key: record[key]
            for key in ("height_origin", "height_extent", "vf_profile")
            if key in record
        }
        return self.lr_api().generate_probs(
            seed=seed, guidance=self.config.lr_bank.guidance, **options
        )

    def create_bank(self) -> Bank:
        count = self.config.lr_bank.samples_per_domain
        records, volumes = [], []
        for _ in trange(count, desc="LR bank", dynamic_ncols=True):
            record = self.bank_condition()
            records.append(record)
            volumes.append(self.bank_volume(record))
        return Bank(
            volumes=torch.stack(volumes),
            height_origins=self._heights(records, "height_origin"),
            height_extents=self._heights(records, "height_extent"),
            conditions=records,
        )

    def _heights(self, records, key):
        if not self.height:
            return None
        return torch.tensor([record[key] for record in records])

    def refresh_bank(self, step: int) -> None:
        """Replace the oldest bank volume every refresh_every_steps."""
        every = self.config.lr_bank.refresh_every_steps
        if not every or not step or step % every:
            return
        bank = self.bank
        index = bank.next_index
        record = self.bank_condition()
        bank.volumes[index] = self.bank_volume(record)
        if self.height:
            bank.height_origins[index] = record["height_origin"]
            bank.height_extents[index] = record["height_extent"]
        bank.conditions[index] = record
        bank.next_index = (index + 1) % len(bank.volumes)

    # Stage hooks -------------------------------------------------------------

    def export_payload(self, step: int, model: Mapping) -> dict:
        return export_payload(self.config, step, model, lr=self.lr_export)

    def stage_state(self) -> dict:
        return {"lr": self.lr_export, "bank": self.bank.state_dict()}

    def load_stage_state(self, payload) -> None:
        self.bank = Bank.from_state(payload["bank"])

    def step(self, step: int, transition: int | None = None) -> dict:
        self.refresh_bank(step)
        return super().step(step, transition)

    def prepare(self, step: int, transition: int | None) -> StepInputs:
        config = self.config
        batch = config.train.volume_batch_size
        generator = self.random["bank"]
        real = self.real_crops(side_heights=self.height)
        indices = randint(len(self.bank.volumes), (batch,), generator)
        low = self.bank.volumes[indices].to(self.device)
        low = augment_volumes(low, self.symmetries, generator)
        conditioning = config.conditioning
        coarse, level = corrupt_coarse(
            low,
            conditioning.coarse_corruption_probability,
            conditioning.coarse_corruption_strength,
            generator,
        )
        conditions = {
            "coarse": resize_phases(coarse, (self.patch_size,) * 3),
            "corruption_level": level,
        }
        if self.height:
            conditions["height"] = self.volume_height(
                self.bank.height_origins[indices], self.bank.height_extents[indices]
            )
        statistics = config.loss.group_statistics
        return StepInputs(
            transition=self.sample_transition() if transition is None else transition,
            real=real,
            conditions=conditions,
            extra={
                "low": low,
                "coarse_mask": (coarse == low).all(1, keepdim=True),
                "group_statistics_ramp": schedule_ramp(
                    step, statistics.start_step, statistics.ramp_steps
                ),
            },
            diagnostics={
                "coarse_corruption_mse": (coarse - low).square().mean(),
                "coarse_corruption_level": level.mean(),
            },
        )

    def objective(self, inputs, clean_probs, fake, real, diagnose):
        loss = self.config.loss
        extra = inputs.extra
        low = extra["low"]
        diagnostics = {}
        total = clean_probs.sum(dtype=torch.float32) * 0.0
        statistics = loss.group_statistics
        ramp = extra["group_statistics_ramp"]
        if statistics.weight > 0 and inputs.transition == 0 and ramp > 0:
            # Differences the LR input already has are allowed on top.
            value, values = group_statistics_loss(
                clean_probs,
                self.groups,
                statistics.max_gap,
                statistics.tolerance,
                self.height,
                low,
            )
            total = total + statistics.weight * ramp * value
            diagnostics.update(values)
            diagnostics["loss/group_statistics"] = value.detach()
            diagnostics["group_statistics/ramp"] = value.new_tensor(ramp)
        consistency, error = consistency_loss(
            clean_probs, low, loss.downsample_mse_tolerance, extra["coarse_mask"]
        )
        total = total + loss.downsample_consistency_weight * consistency
        depth = low.shape[-3]
        detached = clean_probs.detach()
        target = phase_profile(low.detach(), self.num_phases)
        soft = phase_profile(detached, self.num_phases, depth)
        labels = phase_profile(detached.argmax(1), self.num_phases, depth)
        diagnostics.update(
            {
                "consistency": consistency.detach(),
                "lr_mse": error.detach(),
                "profile/sr_coarse_soft_mae": (soft - target).abs().mean(),
                "profile/sr_coarse_label_mae": (labels - target).abs().mean(),
            }
        )
        return total, diagnostics
