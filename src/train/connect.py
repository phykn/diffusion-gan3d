from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .. import AXES
from ..anchor import AnchorCondition
from .prior import ConditionalPrior, PriorCondition

TRIPLETS_PER_AXIS = 2
TRIPLETS_PER_STEP = 1 + len(AXES) * TRIPLETS_PER_AXIS


@dataclass(frozen=True)
class TripletBatch:
    values: torch.Tensor
    axes: torch.Tensor
    center_slots: torch.Tensor
    anchor_flags: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 5 or self.values.shape[1] != 3:
            raise ValueError("triplets must have shape [B, 3, C, H, W].")
        if self.axes.shape != (self.values.shape[0],):
            raise ValueError("triplet axes must have shape [B].")
        if self.axes.dtype != torch.long or self.axes.device != self.values.device:
            raise ValueError("triplet axes must use torch.long on the values device.")
        if self.center_slots.shape != (self.values.shape[0],):
            raise ValueError("triplet center slots must have shape [B].")
        if self.center_slots.dtype != torch.long:
            raise ValueError("triplet center slots must use torch.long dtype.")
        if self.center_slots.device != self.values.device:
            raise ValueError("triplet center slots must be on the values device.")
        if self.center_slots.numel() and (
            int(self.center_slots.min()) < 0 or int(self.center_slots.max()) > 2
        ):
            raise ValueError("triplet center slots must be zero, one, or two.")
        if self.anchor_flags.shape != (self.values.shape[0],):
            raise ValueError("triplet anchor flags must have shape [B].")
        if self.anchor_flags.dtype != torch.bool:
            raise ValueError("triplet anchor flags must use boolean dtype.")
        if self.anchor_flags.device != self.values.device:
            raise ValueError("triplet anchor flags must be on the values device.")

    def __len__(self) -> int:
        return self.values.shape[0]

    def index_select(self, indices: torch.Tensor) -> "TripletBatch":
        return TripletBatch(
            values=self.values.index_select(0, indices),
            axes=self.axes.index_select(0, indices),
            center_slots=self.center_slots.index_select(0, indices),
            anchor_flags=self.anchor_flags.index_select(0, indices),
        )


def normal_transition_loss(real: TripletBatch, fake: TripletBatch) -> torch.Tensor:
    """Compare center-to-neighbor phase transitions along the triplet normal."""
    if real.values.shape != fake.values.shape:
        raise ValueError("real and fake triplets must have the same shape.")
    if not torch.equal(real.axes, fake.axes):
        raise ValueError("real and fake triplets must use the same axes.")
    if not torch.equal(real.anchor_flags, fake.anchor_flags):
        raise ValueError("real and fake triplets must use the same groups.")
    _, _, phase_count, height, width = real.values.shape
    if phase_count == 0 or height == 0 or width == 0:
        raise ValueError("triplets must contain phases and spatial values.")
    if len(fake) == 0:
        return fake.values.sum() * 0.0

    batch_indices = torch.arange(len(real), device=real.values.device)

    def transition_matrices(
        values: torch.Tensor,
        center_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        neighbor_slots = torch.stack(
            (
                (center_slots - 1).clamp_min(0),
                (center_slots + 1).clamp_max(2),
            ),
            dim=1,
        )
        valid_neighbors = torch.stack(
            (center_slots > 0, center_slots < 2),
            dim=1,
        )
        probabilities = (values.to(torch.float32) + 1.0) * 0.5
        center = probabilities[batch_indices, center_slots]
        neighbors = probabilities[batch_indices[:, None], neighbor_slots]
        transitions = torch.einsum(
            "bchw,bnkhw->bnck",
            center,
            neighbors,
        ) / (height * width)
        return transitions, valid_neighbors

    real_transitions, real_valid = transition_matrices(
        real.values,
        real.center_slots,
    )
    fake_transitions, fake_valid = transition_matrices(
        fake.values,
        fake.center_slots,
    )
    valid_neighbors = real_valid & fake_valid
    total_variation = 0.5 * (real_transitions - fake_transitions).abs().sum(
        dim=(-2, -1)
    )
    valid_count = valid_neighbors.sum(dim=1)
    per_triplet = (total_variation * valid_neighbors).sum(
        dim=1
    ) / valid_count.clamp_min(1)
    return balanced_group_mean(
        per_triplet,
        fake.anchor_flags,
        valid_count > 0,
    )


def balanced_group_mean(
    values: torch.Tensor,
    anchor_flags: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    if values.shape != anchor_flags.shape:
        raise ValueError("grouped values and anchor flags must have matching shape.")
    if valid is None:
        valid = torch.ones_like(anchor_flags)
    if valid.shape != anchor_flags.shape or valid.dtype != torch.bool:
        raise ValueError("group validity must be boolean with matching shape.")
    means = []
    for group in (False, True):
        selected = valid & (anchor_flags == group)
        if bool(selected.any()):
            means.append(values[selected].mean())
    if not means:
        return values.sum() * 0.0
    return torch.stack(means).mean()


class Connectivity:
    def __init__(
        self,
        *,
        num_phases: int,
        num_domains: int,
        patch_size: int,
        bank_size: int,
        owned_axes: dict[int, tuple[int, ...]],
        plane_stride: int = 1,
    ) -> None:
        for name, value in (
            ("num_phases", num_phases),
            ("num_domains", num_domains),
            ("patch_size", patch_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.num_phases = num_phases
        self.num_domains = num_domains
        self.patch_size = patch_size
        self.prior = ConditionalPrior(
            num_phases=num_phases,
            num_domains=num_domains,
            patch_size=patch_size,
            volume_count=bank_size,
            plane_stride=plane_stride,
            owned_axes=owned_axes,
        )

    @property
    def prior_count(self) -> int:
        return self.prior.count

    @property
    def prior_storage_bytes(self) -> int:
        return self.prior.storage_bytes

    @property
    def prior_ready(self) -> bool:
        return self.prior.ready

    @property
    def prior_updates(self) -> int:
        return self.prior.updates

    def needs_prior(self, domain: int) -> bool:
        return self.prior.needs(domain)

    def record_prior(
        self,
        prediction: torch.Tensor,
        observed: AnchorCondition,
        domain: int,
    ) -> None:
        self.prior.record(prediction, observed, domain)

    def refresh_prior(
        self,
        prediction: torch.Tensor,
        observed: AnchorCondition,
        domain: int,
    ) -> None:
        self.prior.refresh(prediction, observed, domain)

    def sample_prior_condition(
        self,
        domain: int,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> PriorCondition:
        return self.prior.sample_condition(
            domain,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            generator=generator,
        )

    def match_anchor(
        self,
        prediction: torch.Tensor,
        condition: AnchorCondition,
        domain: int,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TripletBatch, TripletBatch]:
        self._check_domain(domain)
        categorical = self._straight_through(prediction)
        anchors = self._sample_anchor_triplets(categorical, condition)
        if len(anchors):
            general = self._sample_general_triplets(categorical, condition)
            reserve = int(general.axes.unique().numel())
            anchors = self._limit_triplets(
                anchors,
                TRIPLETS_PER_STEP - reserve,
            )
            general = self._limit_general_triplets(
                general,
                TRIPLETS_PER_STEP - len(anchors),
            )
            candidates = self._concat(anchors, general)
        else:
            candidates = anchors
        real, matched = self._sample_prior_matches(
            candidates,
            domain,
            generator=generator,
        )
        return real, candidates.index_select(matched)

    def _sample_general_triplets(
        self,
        volume: torch.Tensor,
        condition: AnchorCondition,
    ) -> TripletBatch:
        general = self._empty_triplets(volume)
        for axis in AXES:
            general = self._concat(
                general,
                self._sample_random_triplets(
                    volume,
                    TRIPLETS_PER_AXIS,
                    axis=axis,
                    condition=condition,
                ),
            )
        return general

    def _sample_prior_matches(
        self,
        target: TripletBatch,
        domain: int,
        *,
        generator: torch.Generator | None,
    ) -> tuple[TripletBatch, torch.Tensor]:
        selected = []
        indices = []
        target_fractions = (
            ((target.values.detach().to(torch.float32) + 1.0) * 0.5)
            .mean(dim=(1, 3, 4))
            .to(device="cpu")
        )
        for index, (axis, fraction) in enumerate(
            zip(target.axes.tolist(), target_fractions, strict=True)
        ):
            volumes = self._prior_volumes(domain, axis)
            if not volumes:
                continue
            choices = []
            choice_fractions = []
            for volume in volumes:
                triplet = self._sample_prior_triplet(volume, axis, generator)
                choices.append(triplet)
                choice_fractions.append(
                    ((triplet.to(torch.float32) + 1.0) * 0.5).mean(dim=(0, 2, 3))
                )
            distances = torch.stack(
                [(value - fraction).abs().mean() for value in choice_fractions]
            )
            selected.append(choices[int(distances.argmin())])
            indices.append(index)

        if not selected:
            empty = torch.empty(0, device=target.values.device, dtype=torch.long)
            return target.index_select(empty), empty
        matched = torch.tensor(indices, device=target.values.device, dtype=torch.long)
        metadata = target.index_select(matched)
        return (
            TripletBatch(
                values=torch.stack(selected).to(
                    device=target.values.device,
                    dtype=target.values.dtype,
                ),
                axes=metadata.axes,
                center_slots=torch.ones(
                    len(selected),
                    device=target.values.device,
                    dtype=torch.long,
                ),
                anchor_flags=metadata.anchor_flags,
            ),
            matched,
        )

    def _prior_volumes(self, domain: int, axis: int) -> tuple[torch.Tensor, ...]:
        return self.prior.volumes(domain, axis)

    def _sample_prior_triplet(
        self,
        labels: torch.Tensor,
        axis: int,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        moved = labels.movedim(axis, 0)
        depth, height, width = moved.shape
        if depth < 3 or self.patch_size > min(height, width):
            raise ValueError("prior volume cannot provide the requested triplet.")
        start = self._random_index(depth - 2, generator)
        top = self._random_start(height, self.patch_size, generator)
        left = self._random_start(width, self.patch_size, generator)
        values = moved[
            start : start + 3,
            top : top + self.patch_size,
            left : left + self.patch_size,
        ]
        return (
            F.one_hot(values.to(torch.long), num_classes=self.num_phases)
            .movedim(-1, 1)
            .to(torch.float32)
            .mul_(2.0)
            .sub_(1.0)
        )

    def _sample_anchor_triplets(
        self,
        volume: torch.Tensor,
        condition: AnchorCondition,
    ) -> TripletBatch:
        self._check_volume(volume)
        if condition.axis_masks.shape != (volume.shape[0], 3, *volume.shape[2:]):
            raise ValueError("anchor axis masks must match the generated volume.")
        if condition.mask.shape != (volume.shape[0], 1, *volume.shape[2:]):
            raise ValueError("anchor mask must match the generated volume.")

        triplets = []
        axes = []
        center_slots = []
        occupied: set[tuple[int, int, int, int, int]] = set()
        for batch in range(volume.shape[0]):
            for axis in AXES:
                moved = volume[batch].movedim(axis + 1, 1)
                moved_axis_mask = condition.axis_masks[batch, axis].movedim(axis, 0)
                moved_full_mask = condition.mask[batch].movedim(axis + 1, 1)
                depth, height, width = moved.shape[1:]
                indices = moved_axis_mask.flatten(1).any(dim=1).nonzero().flatten()
                for index_value in indices.tolist():
                    points = moved_axis_mask[index_value].nonzero()
                    if not points.numel():
                        continue
                    row = (int(points[:, 0].min()) + int(points[:, 0].max())) // 2
                    col = (int(points[:, 1].min()) + int(points[:, 1].max())) // 2
                    top = self._centered_start(row, height)
                    left = self._centered_start(col, width)
                    start = self._window_start(index_value, depth)
                    key = (batch, axis, start, top, left)
                    if key in occupied:
                        continue
                    occupied.add(key)
                    values = moved[
                        :,
                        start : start + 3,
                        top : top + self.patch_size,
                        left : left + self.patch_size,
                    ].movedim(0, 1)
                    mask = moved_full_mask[
                        :,
                        start : start + 3,
                        top : top + self.patch_size,
                        left : left + self.patch_size,
                    ].movedim(0, 1)
                    if bool(mask.all().item()):
                        continue
                    triplets.append(values)
                    axes.append(axis)
                    center_slots.append(index_value - start)

        if not triplets:
            return self._empty_triplets(volume)
        return TripletBatch(
            values=torch.stack(triplets),
            axes=torch.tensor(axes, device=volume.device, dtype=torch.long),
            center_slots=torch.tensor(
                center_slots,
                device=volume.device,
                dtype=torch.long,
            ),
            anchor_flags=torch.ones(
                len(triplets),
                device=volume.device,
                dtype=torch.bool,
            ),
        )

    def _sample_random_triplets(
        self,
        volume: torch.Tensor,
        count: int,
        *,
        axis: int,
        condition: AnchorCondition,
    ) -> TripletBatch:
        if count <= 0:
            return self._empty_triplets(volume)
        moved = volume.movedim(axis + 2, 2)
        depth, height, width = moved.shape[2:]
        anchor_planes = condition.axis_masks[:, axis].movedim(axis + 1, 1)
        anchor_planes = anchor_planes.flatten(2).any(dim=2)
        available = [
            (batch, start)
            for batch in range(volume.shape[0])
            for start in range(depth - 2)
            if not bool(anchor_planes[batch, start : start + 3].any())
        ]
        if not available:
            return self._empty_triplets(volume)

        choices = torch.randperm(len(available), device=volume.device)[:count]
        triplets = []
        for choice in choices.tolist():
            batch, start = available[choice]
            top = self._device_random_start(height, volume.device)
            left = self._device_random_start(width, volume.device)
            triplets.append(
                moved[
                    batch,
                    :,
                    start : start + 3,
                    top : top + self.patch_size,
                    left : left + self.patch_size,
                ].movedim(0, 1)
            )
        return TripletBatch(
            values=torch.stack(triplets),
            axes=torch.full(
                (len(triplets),),
                axis,
                device=volume.device,
                dtype=torch.long,
            ),
            center_slots=torch.ones(
                len(triplets),
                device=volume.device,
                dtype=torch.long,
            ),
            anchor_flags=torch.zeros(
                len(triplets),
                device=volume.device,
                dtype=torch.bool,
            ),
        )

    @staticmethod
    def _concat(first: TripletBatch, second: TripletBatch) -> TripletBatch:
        if not len(second):
            return first
        return TripletBatch(
            values=torch.cat((first.values, second.values)),
            axes=torch.cat((first.axes, second.axes)),
            center_slots=torch.cat((first.center_slots, second.center_slots)),
            anchor_flags=torch.cat((first.anchor_flags, second.anchor_flags)),
        )

    @staticmethod
    def _limit_triplets(batch: TripletBatch, limit: int) -> TripletBatch:
        if len(batch) <= limit:
            return batch
        order = torch.randperm(len(batch), device=batch.values.device)[:limit]
        return batch.index_select(order)

    @staticmethod
    def _limit_general_triplets(batch: TripletBatch, limit: int) -> TripletBatch:
        if limit <= 0:
            return Connectivity._limit_triplets(batch, 0)
        if len(batch) <= limit:
            return batch
        selected = []
        for axis in AXES:
            candidates = (batch.axes == axis).nonzero().flatten()
            if candidates.numel() and len(selected) < limit:
                choice = torch.randint(
                    candidates.numel(),
                    (),
                    device=batch.values.device,
                )
                selected.append(candidates[choice])
        remaining = limit - len(selected)
        if remaining:
            chosen = torch.stack(selected)
            available = torch.ones(
                len(batch), device=batch.values.device, dtype=torch.bool
            )
            available[chosen] = False
            candidates = available.nonzero().flatten()
            order = torch.randperm(
                len(candidates),
                device=batch.values.device,
            )[:remaining]
            selected.extend(candidates.index_select(0, order))
        return batch.index_select(torch.stack(selected))

    def _straight_through(self, prediction: torch.Tensor) -> torch.Tensor:
        self._check_volume(prediction)
        hard = (
            F.one_hot(prediction.argmax(dim=1), num_classes=self.num_phases)
            .movedim(-1, 1)
            .to(dtype=prediction.dtype)
        )
        values = hard.mul(2.0).sub(1.0)
        return values + (prediction - prediction.detach())

    def _check_volume(self, volume: torch.Tensor) -> None:
        if volume.ndim != 5 or volume.shape[1] != self.num_phases:
            raise ValueError("volume must have shape [B, C, D, H, W].")

    def _check_domain(self, domain: int) -> None:
        if not isinstance(domain, int) or isinstance(domain, bool):
            raise TypeError("domain must be an integer.")
        if not 0 <= domain < self.num_domains:
            raise ValueError("domain is outside the prior bank.")

    @staticmethod
    def _check_axis(axis: int) -> None:
        if not isinstance(axis, int) or isinstance(axis, bool) or axis not in AXES:
            raise ValueError("axis must be 0, 1, or 2.")

    @staticmethod
    def _window_start(index: int, size: int) -> int:
        if size < 3:
            raise ValueError("triplet axis size must be at least three.")
        if not 0 <= index < size:
            raise ValueError("triplet index is outside the volume.")
        return min(max(index - 1, 0), size - 3)

    def _empty_triplets(self, volume: torch.Tensor) -> TripletBatch:
        return TripletBatch(
            values=volume.new_empty(
                (0, 3, self.num_phases, self.patch_size, self.patch_size)
            ),
            axes=torch.empty(0, device=volume.device, dtype=torch.long),
            center_slots=torch.empty(0, device=volume.device, dtype=torch.long),
            anchor_flags=torch.empty(0, device=volume.device, dtype=torch.bool),
        )

    def _device_random_start(self, size: int, device: torch.device) -> int:
        if size == self.patch_size:
            return 0
        return int(torch.randint(size - self.patch_size + 1, (), device=device).item())

    def _centered_start(self, center: int, size: int) -> int:
        return min(max(center - self.patch_size // 2, 0), size - self.patch_size)

    @staticmethod
    def _random_index(
        size: int,
        generator: torch.Generator | None,
    ) -> int:
        return int(torch.randint(size, (), generator=generator).item())

    @staticmethod
    def _random_start(
        size: int,
        patch_size: int,
        generator: torch.Generator | None,
    ) -> int:
        if size == patch_size:
            return 0
        return int(torch.randint(size - patch_size + 1, (), generator=generator).item())
