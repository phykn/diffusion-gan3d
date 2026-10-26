import torch

from src.data.augment import augment_volumes
from src.model.layers import NULL_DOMAIN
from src.prepare.height import height_field
from src.prepare.profile import image_profile, rebin_profile
from src.prepare.resize import resize_phases
from src.train.batch import RealBatch, StepPreparation
from src.train.coarse import corrupt_coarse


class ConditionSampler:
    def __init__(
        self,
        settings,
        streams,
        device,
        timesteps,
        active_axes,
        bank=None,
        bank_origins=None,
        bank_extents=None,
    ):
        self.settings = settings
        self.streams = streams
        self.device = device
        self.timesteps = timesteps
        self.active_axes = active_axes
        self.axis_domains = {
            axis: tuple(
                domain for domain, streams in self.streams.items() if axis in streams
            )
            for axis in active_axes
        }
        self.bank = bank
        self.bank_origins = bank_origins
        self.bank_extents = bank_extents

    def prepare_sr_step(
        self,
        domain: int,
        model_domain: int,
        critic_domains: dict[int, int],
        batches: RealBatch,
        transition: int | None,
    ) -> StepPreparation:
        indices = torch.randint(
            len(self.bank[domain]), (self.settings.volume_batch_size,)
        )
        low = self.bank[domain][indices].to(self.device)
        low = augment_volumes(
            low,
            permute_z=self.settings.volume_permute_z,
            flip_z=self.settings.volume_flip_z,
        )
        coarse, level = corrupt_coarse(
            low,
            self.settings.coarse_corruption_probability,
            self.settings.coarse_corruption_strength,
        )
        conditions = {
            "domain": self.make_domain(model_domain, self.settings.volume_batch_size),
            "coarse": resize_phases(coarse, (self.settings.patch_size,) * 3),
            "corruption_level": level,
        }
        if self.settings.height_data is not None:
            conditions["height"] = self.volume_height(
                self.bank_origins[domain][indices],
                domain,
                None
                if self.bank_extents is None
                else self.bank_extents[domain][indices],
            )
        return StepPreparation(
            transition=self.sample_transition() if transition is None else transition,
            domain=domain,
            critic_domains=critic_domains,
            real=batches,
            target_vf=low.new_zeros(
                (self.settings.volume_batch_size, self.settings.num_phases)
            ),
            vf_present=torch.zeros(self.settings.volume_batch_size, dtype=torch.bool),
            model_conditions=conditions,
            real_transition_ramp=0.0,
            coarse_target=low,
            coarse_mask=(coarse == low).all(1, keepdim=True),
            diagnostics={
                "coarse_corruption_mse": (coarse - low).square().mean(),
                "coarse_corruption_level": level.mean(),
            },
        )

    def prepare_lr_step(
        self,
        step: int,
        domain: int,
        model_domain: int,
        critic_domains: dict[int, int],
        batches: RealBatch,
        transition: int | None,
    ) -> StepPreparation:
        owned = tuple(self.streams[domain])
        # Height, profile and vf of a volume share one side crop.
        side = (
            None
            if self.settings.height_data is None
            else next(axis for axis in owned if axis in batches.origins)
        )
        target_vf = self.sample_vf(batches, owned, side)
        vf_present = self.sample_vf_presence()
        model_conditions = {
            "domain": self.make_domain(model_domain, self.settings.volume_batch_size),
            "vf": target_vf,
            "vf_present": vf_present.to(target_vf.device),
        }
        if self.settings.height_data is not None:
            model_conditions["height"] = self.volume_height(
                batches.origins[side][: self.settings.volume_batch_size],
                domain,
                batches.extents[side][: self.settings.volume_batch_size],
            )
        if self.settings.profile_settings["enabled"]:
            profile = batches.profiles[side][: self.settings.volume_batch_size]
            target_vf = profile.mean(-1)
            model_conditions.update(
                profile=profile,
                profile_present=vf_present.to(self.device),
                vf=target_vf,
            )
        return StepPreparation(
            transition=self.sample_transition() if transition is None else transition,
            domain=domain,
            critic_domains=critic_domains,
            real=batches,
            target_vf=target_vf,
            vf_present=vf_present,
            model_conditions=model_conditions,
            real_transition_ramp=self.schedule_ramp(
                step,
                self.settings.real_transition_start_step,
                self.settings.real_transition_ramp_steps,
            ),
        )

    def sample_transition(self) -> int:
        return int(torch.randint(self.timesteps, ()).item())

    def select_batch_domains(self, domain: int) -> dict[int, int]:
        selected = {}
        for axis in self.active_axes:
            if axis in self.streams[domain]:
                selected[axis] = domain
                continue
            candidates = self.axis_domains[axis]
            index = int(torch.randint(len(candidates), ()).item())
            selected[axis] = candidates[index]
        return selected

    def sample_target_domain(self) -> int:
        return int(torch.randint(len(self.streams), ()).item())

    def get_batches(
        self,
        domain: int,
        batch_domains: dict[int, int] | None = None,
    ) -> RealBatch:
        if batch_domains is None:
            batch_domains = self.select_batch_domains(domain)
        batches = RealBatch(images={}, domains=batch_domains)
        for axis in self.active_axes:
            batch = self.streams[batch_domains[axis]][axis].next()
            images = batch["image"]
            if self.settings.height_data is not None:
                origins = batch["height_origin"]
                if axis != 0:
                    batches.origins[axis] = origins
                    batches.extents[axis] = batch["height_extent"]
                    batches.heights[axis] = height_field(
                        images.shape[-2:],
                        0,
                        origins,
                        self.settings.height_data["crop_size"] / images.shape[-1],
                        batch["height_extent"],
                        self.device,
                    )
            batches.images[axis] = images.to(self.device, non_blocking=True)
            if self.settings.profile_settings["enabled"] and axis in batches.origins:
                batches.profiles[axis] = rebin_profile(
                    image_profile(
                        batches.images[axis],
                        bins=self.settings.profile_settings["num_bins"],
                    ),
                    self.settings.patch_size,
                )
        return batches

    def volume_height(self, origins, domain, extents=None):
        data = self.settings.height_data
        return height_field(
            (self.settings.patch_size,) * 3,
            0,
            origins,
            data["crop_size"] / self.settings.patch_size,
            data["height_extents"][domain] if extents is None else extents,
            self.device,
        )

    def sample_domain_condition(self, domain: int) -> int:
        drop = 1.0 - self.settings.domain_keep_probability
        if drop > 0.0 and bool(torch.rand(()) < drop):
            return NULL_DOMAIN
        return domain

    @staticmethod
    def make_critic_domains(
        domain: int,
        model_domain: int,
        batch_domains: dict[int, int],
    ) -> dict[int, int]:
        if model_domain == NULL_DOMAIN:
            return {axis: NULL_DOMAIN for axis in batch_domains}
        return {
            axis: domain if source_domain == domain else NULL_DOMAIN
            for axis, source_domain in batch_domains.items()
        }

    def make_domain(self, domain: int, batch_size: int) -> torch.Tensor:
        return torch.full(
            (batch_size,),
            domain,
            device=self.device,
            dtype=torch.long,
        )

    def sample_vf(
        self,
        batches: RealBatch,
        owned_axes: tuple[int, ...],
        side: int | None = None,
    ) -> torch.Tensor:
        if side is not None:
            axis = side
        else:
            axis = owned_axes[int(torch.randint(len(owned_axes), ()).item())]
        fractions = batches.images[axis].float().mean((2, 3))
        indices = torch.arange(self.settings.volume_batch_size, device=fractions.device)
        return fractions[indices % len(fractions)]

    def sample_vf_presence(self) -> torch.Tensor:
        """Mask of samples that keep vf and profile; the rest train CFG."""
        probability = self.settings.cfg_dropout_probability
        return torch.rand(self.settings.volume_batch_size, device="cpu") >= probability

    @staticmethod
    def schedule_ramp(step: int, start: int, ramp: int) -> float:
        if step < start:
            return 0.0
        if ramp == 0:
            return 1.0
        return min((step - start + 1) / ramp, 1.0)
