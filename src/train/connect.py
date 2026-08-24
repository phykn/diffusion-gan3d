from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .. import AXES
from ..anchor import AnchorCondition


@dataclass(frozen=True)
class TripletBatch:
    values: torch.Tensor
    axes: torch.Tensor
    gaps: torch.Tensor
    center_slots: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 5 or self.values.shape[1] != 3:
            raise ValueError("triplets must have shape [B, 3, C, H, W].")
        if self.axes.shape != (self.values.shape[0],):
            raise ValueError("triplet axes must have shape [B].")
        if self.axes.dtype != torch.long or self.axes.device != self.values.device:
            raise ValueError("triplet axes must use torch.long on the values device.")
        if self.gaps.shape != (self.values.shape[0],):
            raise ValueError("triplet gaps must have shape [B].")
        if self.gaps.dtype != torch.long or self.gaps.device != self.values.device:
            raise ValueError("triplet gaps must use torch.long on the values device.")
        if self.gaps.numel() and bool((self.gaps < 1).any()):
            raise ValueError("triplet gaps must be positive.")
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
    def __len__(self) -> int:
        return self.values.shape[0]

    def index_select(self, indices: torch.Tensor) -> "TripletBatch":
        return TripletBatch(
            values=self.values.index_select(0, indices),
            axes=self.axes.index_select(0, indices),
            gaps=self.gaps.index_select(0, indices),
            center_slots=self.center_slots.index_select(0, indices),
        )


@dataclass(frozen=True)
class _LocatedTriplets:
    triplets: TripletBatch
    locations: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if len(self.locations) != len(self.triplets):
            raise ValueError("triplet locations must match the triplet batch.")
    def index_select(self, indices: torch.Tensor) -> "_LocatedTriplets":
        selected = indices.tolist()
        return _LocatedTriplets(
            self.triplets.index_select(indices),
            tuple(self.locations[index] for index in selected),
        )


def normal_transition_loss(real: TripletBatch, fake: TripletBatch) -> torch.Tensor:
    """Compare center-to-neighbor phase transitions along the triplet normal."""
    if real.values.shape != fake.values.shape:
        raise ValueError("real and fake triplets must have the same shape.")
    if not torch.equal(real.axes, fake.axes):
        raise ValueError("real and fake triplets must use the same axes.")
    if not torch.equal(real.gaps, fake.gaps):
        raise ValueError("real and fake triplets must use the same gaps.")
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
    valid = valid_count > 0
    if not bool(valid.any()):
        return fake.values.sum() * 0.0
    return per_triplet[valid].mean()


class Connectivity:
    def __init__(
        self,
        *,
        num_phases: int,
        max_gap: int = 1,
    ) -> None:
        for name, value in (
            ("num_phases", num_phases),
            ("max_gap", max_gap),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.num_phases = num_phases
        self.max_gap = max_gap

    def match_anchor(
        self,
        prediction: torch.Tensor,
        reference: torch.Tensor,
        condition: AnchorCondition,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TripletBatch, TripletBatch]:
        self._check_volume(prediction)
        self._check_volume(reference)
        if prediction.shape != reference.shape:
            raise ValueError("prediction and reference volumes must have matching shapes.")
        categorical = self._straight_through(prediction)
        reference = self._straight_through(reference).detach()
        located = self._sample_anchor_triplets(
            categorical,
            condition,
            generator=generator,
        )
        return self._extract_triplets(reference, located), located.triplets

    def _extract_triplets(
        self,
        volume: torch.Tensor,
        located: _LocatedTriplets,
    ) -> TripletBatch:
        if not len(located.triplets):
            return self._empty_triplets(volume)
        values = []
        for (batch, axis, index), gap in zip(
            located.locations,
            located.triplets.gaps.tolist(),
            strict=True,
        ):
            moved = volume[batch].movedim(axis + 1, 1)
            slice_indices = _relation_indices(index, moved.shape[1], gap)
            if slice_indices is None:
                raise RuntimeError("located triplet relation no longer fits the volume.")
            values.append(moved[:, list(slice_indices)].movedim(0, 1))
        return TripletBatch(
            values=torch.stack(values),
            axes=located.triplets.axes,
            gaps=located.triplets.gaps,
            center_slots=located.triplets.center_slots,
        )

    def _sample_anchor_triplets(
        self,
        volume: torch.Tensor,
        condition: AnchorCondition,
        generator: torch.Generator | None = None,
    ) -> _LocatedTriplets:
        self._check_volume(volume)
        if condition.axis_masks.shape != (volume.shape[0], 3, *volume.shape[2:]):
            raise ValueError("anchor axis masks must match the generated volume.")
        if condition.mask.shape != (volume.shape[0], 1, *volume.shape[2:]):
            raise ValueError("anchor mask must match the generated volume.")

        triplets = []
        axes = []
        gaps = []
        center_slots = []
        locations = []
        occupied: set[tuple[int, int, tuple[int, int, int]]] = set()
        for batch in range(volume.shape[0]):
            for axis in AXES:
                moved = volume[batch].movedim(axis + 1, 1)
                axis_mask = condition.axis_masks[batch, axis]
                if not bool(axis_mask.any()):
                    axis_mask = condition.mask[batch, 0]
                moved_axis_mask = axis_mask.movedim(axis, 0)
                moved_full_mask = condition.mask[batch].movedim(axis + 1, 1)
                depth = moved.shape[1]
                indices = moved_axis_mask.flatten(1).any(dim=1).nonzero().flatten()
                if not len(indices):
                    continue
                index_value = int(
                    indices[self._random_index(len(indices), generator)].item()
                )
                gap = self._sample_gap(index_value, depth, generator)
                slice_indices = _relation_indices(index_value, depth, gap)
                if slice_indices is None:
                    raise RuntimeError("sampled gap does not fit the volume.")
                key = (batch, axis, slice_indices)
                if key in occupied:
                    continue
                occupied.add(key)
                values = moved[:, list(slice_indices)].movedim(0, 1)
                mask = moved_full_mask[:, list(slice_indices)].movedim(0, 1)
                if bool(mask.all().item()):
                    continue
                triplets.append(values)
                axes.append(axis)
                gaps.append(gap)
                center_slots.append(slice_indices.index(index_value))
                locations.append((batch, axis, index_value))

        if not triplets:
            return _LocatedTriplets(
                self._empty_triplets(volume),
                (),
            )
        return _LocatedTriplets(
            TripletBatch(
                values=torch.stack(triplets),
                axes=torch.tensor(axes, device=volume.device, dtype=torch.long),
                gaps=torch.tensor(gaps, device=volume.device, dtype=torch.long),
                center_slots=torch.tensor(
                    center_slots,
                    device=volume.device,
                    dtype=torch.long,
                ),
            ),
            tuple(locations),
        )

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

    def _sample_gap(
        self,
        index: int,
        size: int,
        generator: torch.Generator | None,
    ) -> int:
        gaps = [
            gap
            for gap in range(1, min(self.max_gap, (size - 1) // 2) + 1)
            if _relation_indices(index, size, gap) is not None
        ]
        if not gaps:
            raise ValueError("no valid connectivity gap is available.")
        return gaps[self._random_index(len(gaps), generator)]

    def _empty_triplets(self, volume: torch.Tensor) -> TripletBatch:
        return TripletBatch(
            values=volume.new_empty(
                (0, 3, self.num_phases, volume.shape[-2], volume.shape[-1])
            ),
            axes=torch.empty(0, device=volume.device, dtype=torch.long),
            gaps=torch.empty(0, device=volume.device, dtype=torch.long),
            center_slots=torch.empty(0, device=volume.device, dtype=torch.long),
        )

    @staticmethod
    def _random_index(
        size: int,
        generator: torch.Generator | None,
    ) -> int:
        return int(torch.randint(size, (), generator=generator).item())


def _relation_indices(
    index: int,
    size: int,
    gap: int,
) -> tuple[int, int, int] | None:
    if gap < 1:
        return None
    if index - gap >= 0 and index + gap < size:
        return index - gap, index, index + gap
    if index + 2 * gap < size:
        return index, index + gap, index + 2 * gap
    if index - 2 * gap >= 0:
        return index - 2 * gap, index - gap, index
    return None
