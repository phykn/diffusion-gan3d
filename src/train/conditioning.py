"""Sample real data, replay anchors, and LR/SR model conditions."""

import torch

from src.anchor import AnchorCondition, PlaneAnchor, encode_anchors
from src.data.augment import augment_volumes, crop_images
from src.model.layers import NULL_DOMAIN
from src.prepare.height import height_field
from src.prepare.profile import image_profile, rebin_profile
from src.prepare.resize import resize_phases
from src.train.anchor_bank import AnchorBank
from src.train.batch import (
    AnchorSelection,
    ConditionPresence,
    RealBatch,
    StepPreparation,
)
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
        self.anchor_bank = (
            None
            if bank is not None
            else AnchorBank(
                settings.anchor_bank_capacity, settings.anchor_plane_spacing
            )
        )
        self.use_multi_anchor_next = False

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
            low, preserve_height=self.settings.height_data is not None
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
        absent = torch.zeros(self.settings.volume_batch_size, dtype=torch.bool)
        return StepPreparation(
            transition=self.sample_transition(False)
            if transition is None
            else transition,
            domain=domain,
            critic_domains=critic_domains,
            real=batches,
            selection=None,
            target_vf=low.new_zeros(
                (self.settings.volume_batch_size, self.settings.num_phases)
            ),
            presence=ConditionPresence(absent, absent),
            model_conditions=conditions,
            anchor_ramp=0.0,
            connectivity_ramp=0.0,
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
        own_batches = {axis: batches.images[axis] for axis in self.streams[domain]}
        ramp = self.get_anchor_ramp(step)
        selection = (
            None
            if ramp == 0.0
            else self.sample_anchor(
                batches,
                self.settings.patch_size,
                owned_axes=tuple(own_batches),
                domain=domain,
            )
        )
        anchor = None if selection is None else selection.condition
        target_vf = self.sample_vf(batches, selection, tuple(own_batches))
        presence = self.sample_condition_presence(selection is not None)
        model_conditions = self.make_model_conditions(
            anchor,
            target_vf,
            presence,
            self.make_domain(model_domain, self.settings.volume_batch_size),
        )
        if self.settings.height_data is not None:
            height = None if selection is None else selection.height
            if height is None:
                axis = next(a for a in self.streams[domain] if a in batches.origins)
                height = self.volume_height(
                    batches.origins[axis][: self.settings.volume_batch_size],
                    domain,
                    batches.extents[axis][: self.settings.volume_batch_size],
                )
            model_conditions["height"] = height
        if self.settings.profile_settings["enabled"]:
            profile = None if selection is None else selection.profile
            if profile is None:
                axis = next(a for a in own_batches if a in batches.profiles)
                profile = batches.profiles[axis][: self.settings.volume_batch_size]
            target_vf = profile.mean(-1)
            model_conditions.update(
                profile=profile,
                profile_present=presence.vf.to(self.device),
                vf=target_vf,
            )
        if transition is None:
            transition = self.sample_transition(anchor is not None)
        return StepPreparation(
            transition=transition,
            domain=domain,
            critic_domains=critic_domains,
            real=batches,
            selection=selection,
            target_vf=target_vf,
            presence=presence,
            model_conditions=model_conditions,
            anchor_ramp=ramp,
            connectivity_ramp=self.schedule_ramp(
                step,
                self.settings.connectivity_start_step,
                self.settings.connectivity_ramp_steps,
            ),
        )

    def sample_transition(self, anchored: bool) -> int:
        if not anchored:
            return int(torch.randint(self.timesteps, ()).item())
        if self.timesteps == 1 or bool(torch.rand(()) < 0.25):
            return 0
        return int(torch.randint(1, self.timesteps, ()).item())

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
                    batches.geometry[axis] = [
                        {
                            "image_id": image_id,
                            "height_origin": float(origin),
                            "height_extent": float(extent),
                            "crop_origin": crop.tolist(),
                            "source_shape": shape.tolist(),
                        }
                        for image_id, origin, extent, crop, shape in zip(
                            batch["image_id"],
                            origins,
                            batch["height_extent"],
                            batch["crop_origin"],
                            batch["source_shape"],
                            strict=True,
                        )
                    ]
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
        if self.settings.domain_dropout > 0.0 and bool(
            torch.rand(()) < self.settings.domain_dropout
        ):
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
        selection: AnchorSelection | None,
        owned_axes: tuple[int, ...],
    ) -> torch.Tensor:
        if selection is not None and selection.measured is not None:
            measured = selection.measured
            mask = measured.mask[:, 0].float()
            sums = torch.einsum("bcdhw,bdhw->bc", measured.image.float(), mask)
            counts = mask.sum(dim=(1, 2, 3)).unsqueeze(1)
            return (sums / counts + 1.0) * 0.5
        if self.settings.height_data is not None:
            axis = next(axis for axis in owned_axes if axis in batches.origins)
        else:
            axis = owned_axes[int(torch.randint(len(owned_axes), ()).item())]
        fractions = batches.images[axis].float().mean((2, 3))
        indices = torch.arange(self.settings.volume_batch_size, device=fractions.device)
        return fractions[indices % len(fractions)]

    def sample_condition_presence(self, has_anchor: bool) -> ConditionPresence:
        batch = self.settings.volume_batch_size
        anchor = torch.full(
            (batch,),
            has_anchor,
            device="cpu",
            dtype=torch.bool,
        )
        vf = torch.ones(batch, device="cpu", dtype=torch.bool)
        random = torch.rand(batch, device="cpu")
        if has_anchor:
            probability = self.settings.cfg_drop_each_probability
            joint_null = random < probability
            anchor_null = (random >= probability) & (random < 2.0 * probability)
            vf_null = (random >= 2.0 * probability) & (random < 3.0 * probability)
            anchor = ~(joint_null | anchor_null)
            vf = ~(joint_null | vf_null)
        else:
            vf = random >= 2.0 * self.settings.cfg_drop_each_probability
        return ConditionPresence(anchor=anchor, vf=vf)

    @staticmethod
    def make_model_conditions(
        anchor: AnchorCondition | None,
        target_vf: torch.Tensor,
        presence: ConditionPresence,
        domain: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        conditions = {
            "domain": domain,
            "vf": target_vf,
            "vf_present": presence.vf.to(target_vf.device),
        }
        if anchor is not None:
            visible = presence.anchor.to(anchor.mask.device).reshape(-1, 1, 1, 1, 1)
            conditions["anchor_image"] = anchor.image
            conditions["anchor_mask"] = anchor.mask * visible.to(anchor.mask.dtype)
        return conditions

    def get_anchor_ramp(self, step: int) -> float:
        return self.schedule_ramp(
            step, self.settings.anchor_start_step, self.settings.anchor_ramp_steps
        )

    @staticmethod
    def schedule_ramp(step: int, start: int, ramp: int) -> float:
        if step < start:
            return 0.0
        if ramp == 0:
            return 1.0
        return min((step - start + 1) / ramp, 1.0)

    def sample_anchor(
        self,
        batches: RealBatch,
        volume_size: int,
        owned_axes: tuple[int, ...] | None = None,
        domain: int = 0,
    ) -> AnchorSelection | None:
        probability = self.settings.anchor_training_probability
        if probability <= 0.0:
            return None
        if probability < 1.0 and not bool(torch.rand(()) < probability):
            return None
        if self.use_multi_anchor_next:
            replay = self.anchor_bank.sample(
                domain, self.settings.volume_batch_size, self.device
            )
            self.use_multi_anchor_next = False
            if replay is not None:
                return AnchorSelection(
                    condition=replay.condition,
                    observed_mask=replay.measured.mask,
                    observed_axis_masks=replay.measured.axis_masks,
                    source="multi",
                    measured=replay.measured,
                    reference=replay.reference,
                    height=replay.height,
                    profile=replay.profile,
                    geometry=replay.geometry,
                )
        self.use_multi_anchor_next = True
        return self.sample_real_anchor(
            batches,
            volume_size,
            owned_axes=owned_axes,
        )

    def sample_real_anchor(
        self,
        batches: RealBatch,
        volume_size: int,
        owned_axes: tuple[int, ...] | None = None,
    ) -> AnchorSelection:
        if owned_axes is None:
            owned_axes = tuple(batches.images)
        shared_axes = tuple(axis for axis in batches.images if axis not in owned_axes)
        use_shared = (
            bool(shared_axes)
            and self.settings.anchor_shared_axis_probability > 0.0
            and bool(torch.rand(()) < self.settings.anchor_shared_axis_probability)
        )
        axes = shared_axes if use_shared else owned_axes
        if batches.origins:
            # An xy section has no observed absolute height along z.
            axes = tuple(a for a in axes if a in batches.origins)
            if not axes:
                axes = tuple(a for a in owned_axes if a in batches.origins)
                use_shared = False
        axis = axes[int(torch.randint(len(axes), ()).item())]
        images = batches.images[axis]
        batch_indices = torch.randperm(
            images.shape[0],
            device="cpu",
        )[: self.settings.volume_batch_size]
        selected = images.index_select(0, batch_indices.to(images.device))
        shape = tuple(min(volume_size, size) for size in selected.shape[-2:])
        selected = crop_images(selected, shape)
        position = tuple(
            int(torch.randint(volume_size - size + 1, ()).item()) for size in shape
        )
        plane_index = int(torch.randint(volume_size, ()).item())
        plane = PlaneAnchor(
            image=selected,
            axis=axis,
            index=plane_index,
            position=position,
        )
        condition = encode_anchors(
            (plane,),
            batch_size=self.settings.volume_batch_size,
            num_phases=self.settings.num_phases,
            volume_size=volume_size,
            device=self.device,
            dtype=torch.float32,
            reconcile=False,
            validate=False,
        )
        return AnchorSelection(
            condition=condition,
            observed_mask=condition.mask,
            observed_axis_masks=condition.axis_masks,
            source="shared" if use_shared else "real",
            measured=condition,
            height=self.volume_height(
                batches.origins[axis][batch_indices],
                batches.domains[axis],
                batches.extents[axis][batch_indices],
            )
            if axis in batches.origins
            else None,
            profile=batches.profiles[axis][batch_indices.to(self.device)]
            if axis in batches.profiles
            else None,
            geometry=[batches.geometry[axis][int(index)] for index in batch_indices]
            if axis in batches.origins
            else None,
        )
