from dataclasses import dataclass

import torch

from .. import AXES
from ..anchor import AnchorCondition


@dataclass(frozen=True)
class TripletBatch:
    values: torch.Tensor
    axes: torch.Tensor
    gaps: torch.Tensor
    center_slots: torch.Tensor

    def __len__(self) -> int:
        return self.values.shape[0]


@dataclass(frozen=True)
class LocatedTriplets:
    triplets: TripletBatch
    locations: tuple[tuple[int, int, int], ...]


def compute_transition_loss(real: TripletBatch, fake: TripletBatch) -> torch.Tensor:
    if real.values.shape != fake.values.shape:
        raise ValueError("real and fake triplets must have the same shape.")
    if not (
        torch.equal(real.axes, fake.axes)
        and torch.equal(real.gaps, fake.gaps)
        and torch.equal(real.center_slots, fake.center_slots)
    ):
        raise ValueError("real and fake triplets must use matching metadata.")
    if len(fake) == 0:
        return fake.values.sum().mul(0.0)

    triplet_indices = torch.arange(len(real), device=real.values.device)
    center_slots = real.center_slots
    valid_neighbors = torch.stack(
        (center_slots > 0, center_slots < 2),
        dim=1,
    )
    probs = torch.stack(
        (
            real.values.to(torch.float32),
            fake.values.to(torch.float32),
        )
    )
    probs = (probs + 1.0) * 0.5
    centers = probs[:, triplet_indices, center_slots]
    left = probs[:, triplet_indices, (center_slots - 1).clamp_min(0)]
    right = probs[:, triplet_indices, (center_slots + 1).clamp_max(2)]
    changes = torch.stack((left - centers, right - centers), dim=2)
    transition_error = (changes[0] - changes[1]).abs().mean(
        dim=(2, 3, 4)
    )
    valid_count = valid_neighbors.sum(dim=1)
    per_triplet = (transition_error * valid_neighbors).sum(dim=1) / valid_count
    middle = center_slots == 1
    if bool(middle.any()):
        bend = changes[:, :, 1] - changes[:, :, 0]
        bend_error = 0.5 * (bend[0] - bend[1]).abs().mean(dim=(1, 2, 3))
        per_triplet[middle] = 0.5 * (
            per_triplet[middle] + bend_error[middle]
        )
    return per_triplet.mean()


class AnchorTripletSampler:
    def __init__(self, max_gap: int = 1) -> None:
        if not isinstance(max_gap, int) or isinstance(max_gap, bool) or max_gap < 1:
            raise ValueError("max_gap must be a positive integer.")
        self.max_gap = max_gap

    def sample(
        self,
        prediction: torch.Tensor,
        reference: torch.Tensor,
        condition: AnchorCondition,
        generator: torch.Generator | None = None,
    ) -> tuple[TripletBatch, TripletBatch]:
        self._check_volume(prediction)
        self._check_volume(reference)
        if prediction.shape != reference.shape:
            raise ValueError("prediction and reference volumes must have matching shapes.")
        reference = reference.detach()
        located = self._sample_anchor_triplets(
            prediction,
            condition,
            generator=generator,
        )
        return self._extract_triplets(reference, located), located.triplets

    def _extract_triplets(
        self,
        volume: torch.Tensor,
        located: LocatedTriplets,
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
    ) -> LocatedTriplets:
        self._check_volume(volume)
        expected_axis_shape = (volume.shape[0], 3) + tuple(volume.shape[2:])
        expected_mask_shape = (volume.shape[0], 1) + tuple(volume.shape[2:])
        if condition.axis_masks.shape != expected_axis_shape:
            raise ValueError("anchor axis masks must match the generated volume.")
        if condition.mask.shape != expected_mask_shape:
            raise ValueError("anchor mask must match the generated volume.")

        triplets = []
        axes = []
        gaps = []
        center_slots = []
        locations = []
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
            return LocatedTriplets(
                self._empty_triplets(volume),
                (),
            )
        return LocatedTriplets(
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

    def _check_volume(self, volume: torch.Tensor) -> None:
        if volume.ndim != 5:
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
                (0, 3, volume.shape[1], volume.shape[-2], volume.shape[-1])
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
