from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from tqdm import trange

from src.axes import PLANES
from src.config.io import load_generation_settings
from src.config.schema import SRConfig
from src.data.crops import CropSampler, phase_channels, resize_phases
from src.data.height import image_profile
from src.data.orientation import augment_volumes, bank_symmetries
from src.export import (
    build_diffusion,
    check_export,
    check_lr_source,
    export_config,
    export_payload,
    load_denoiser,
)
from src.rng import rand, randint
from src.sampling.conditions import LRConditions
from src.sampling.memory import check_memory, estimate_memory
from src.sampling.plan import make_plan
from src.sampling.sampler import Predictor, TileSampler
from src.stats.labels import phase_profile
from src.train.core import StepInputs, Trainer
from src.train.losses.consistency import consistency_loss


def corrupt_coarse(low, probability, strength, generator):
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

    def state_dict(self):
        return {
            "volumes": self.volumes,
            "height_origins": self.height_origins,
            "height_extents": self.height_extents,
            "conditions": [dict(item) for item in self.conditions],
            "next_index": self.next_index,
        }

    @classmethod
    def from_state(cls, state):
        return cls(
            volumes=state["volumes"],
            height_origins=state["height_origins"],
            height_extents=state["height_extents"],
            conditions=[dict(item) for item in state["conditions"]],
            next_index=int(state["next_index"]),
        )


def crop_profile(image, origin, extent, crop_size, bins):
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
    """SR learns to refine volumes the frozen LR model makes, so it trains on a bank
    of them rather than on downsampled real data. Each volume takes its height and
    depth profile from a random real side crop; one volume is replaced every
    lr_bank.refresh_every_steps. The bank and the LR export travel in the
    checkpoint, so a resume needs neither the LR run nor its images.
    """

    config: SRConfig

    def __init__(
        self,
        config,
        device,
        planes,
        lr,
        bank=None,
    ):
        super().__init__(config, device, planes)
        self.lr_export = check_export(lr, "lr")
        self.lr_config = export_config(self.lr_export)
        check_lr_source(self.lr_config, config)
        self.symmetries = bank_symmetries(config)
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
        self._lr_sampling = None
        self.bank = self.create_bank() if bank is None else bank

    # Bank ------------------------------------------------------------------

    def bank_sampling(self):
        if self._lr_sampling is None:
            cfg = self.lr_config
            model = load_denoiser(self.lr_export, cfg).to(self.device)
            diffusion = build_diffusion(cfg).to(self.device)
            sampler = TileSampler(
                Predictor.from_config(model, diffusion, cfg, self.device)
            )
            patch = cfg.patch_size
            plan = make_plan(
                (patch,) * 3,
                patch,
                model.downsample_factor,
                load_generation_settings().overlap,
            )
            estimate = estimate_memory(
                plan,
                self.num_phases,
                cfg.model.generator.channels[0],
                self.device,
                probabilities=True,
                profile=cfg.profile_enabled,
            )
            self._lr_sampling = sampler, plan, estimate
        return self._lr_sampling

    def bank_condition(self):
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

    def bank_volume(self, record):
        seed = int(randint(2**62, (), self.random["bank"]))
        options = {
            key: record[key]
            for key in ("height_origin", "height_extent", "vf_profile")
            if key in record
        }
        sampler, plan, estimate = self.bank_sampling()
        conditions = LRConditions(self.lr_config, plan, self.device, **options)
        return sampler.sample(
            plan,
            seed=seed,
            conditions=conditions,
            guidance=self.config.lr_bank.guidance,
            probabilities=True,
            storage=check_memory(estimate, self.device, "auto"),
        )

    def create_bank(self):
        count = self.config.lr_bank.size
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

    def refresh_bank(self, step):
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

    def export_payload(self, step, model):
        return export_payload(self.config, step, model, lr=self.lr_export)

    def stage_state(self):
        return {"lr": self.lr_export, "bank": self.bank.state_dict()}

    def step(self, step, transition=None):
        self.refresh_bank(step)
        return super().step(step, transition)

    def prepare(self, step, transition):
        config = self.config
        batch = config.train.volume_batch_size
        generator = self.random["bank"]
        real = self.real_crops()
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
        return StepInputs(
            step=step,
            transition=self.sample_transition() if transition is None else transition,
            real=real,
            conditions=conditions,
            extra={
                "low": low,
                "coarse_mask": (coarse == low).all(1, keepdim=True),
            },
            diagnostics={
                "coarse_corruption_mse": (coarse - low).square().mean(),
                "coarse_corruption_level": level.mean(),
            },
        )

    def objective(self, inputs, clean_probs):
        loss = self.config.loss
        extra = inputs.extra
        low = extra["low"]
        diagnostics = {}
        total = clean_probs.sum(dtype=torch.float32) * 0.0
        total = self.add_group_statistics(total, inputs, clean_probs, diagnostics, low)
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
