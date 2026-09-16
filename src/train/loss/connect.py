from dataclasses import dataclass

import torch

from src import AXES
from src.anchor import AnchorCondition


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
    regions: tuple[tuple[int, int, int, int], ...] = ()
    relations: tuple[tuple[int, int, int], ...] = ()


@torch.no_grad()
def anchor_boundary_metrics(
    prediction, reference, condition
) -> dict[str, torch.Tensor]:
    """One-voxel phase agreement across the observed/unobserved boundary; not percolation."""
    total = prediction.new_zeros(())
    excess = prediction.new_zeros(())
    count = prediction.new_zeros(())
    for axis in AXES:
        left = [slice(None)] * 5
        right = [slice(None)] * 5
        left[axis + 2], right[axis + 2] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        boundary = (condition.mask[left] ^ condition.mask[right]).squeeze(1)
        jump = (prediction[left] - prediction[right]).abs().sum(1) * 0.25
        ref_jump = (reference[left] - reference[right]).abs().sum(1) * 0.25
        total += ((1 - jump) * boundary).sum()
        excess += ((jump - ref_jump) * boundary).sum()
        count += boundary.sum()
    return {
        "anchor/boundary_pairs": count,
        "anchor/neighbor_agreement": total / count.clamp_min(1),
        "anchor/neighbor_excess_jump": excess / count.clamp_min(1),
    }


def compute_transition_loss(real: TripletBatch, fake: TripletBatch) -> torch.Tensor:
    if real.values.shape != fake.values.shape:
        raise ValueError("real and fake triplets must have the same shape.")
    if not (
        (real.axes is fake.axes or torch.equal(real.axes, fake.axes))
        and (real.gaps is fake.gaps or torch.equal(real.gaps, fake.gaps))
        and (
            real.center_slots is fake.center_slots
            or torch.equal(real.center_slots, fake.center_slots)
        )
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
    transition_error = (changes[0] - changes[1]).abs().mean(dim=(2, 3, 4))
    valid_count = valid_neighbors.sum(dim=1)
    per_triplet = (transition_error * valid_neighbors).sum(dim=1) / valid_count
    middle = center_slots == 1
    bend = changes[:, :, 1] - changes[:, :, 0]
    bend_error = 0.5 * (bend[0] - bend[1]).abs().mean(dim=(1, 2, 3))
    per_triplet = torch.where(middle, 0.5 * (per_triplet + bend_error), per_triplet)
    return per_triplet.mean()


class AnchorTripletSampler:
    def __init__(self, max_gap: int = 1, windows_per_plane: int = 1) -> None:
        if not isinstance(max_gap, int) or isinstance(max_gap, bool) or max_gap < 1:
            raise ValueError("max_gap must be a positive integer.")
        self.max_gap = max_gap
        if type(windows_per_plane) is not int or windows_per_plane < 1:
            raise ValueError("windows_per_plane must be a positive integer.")
        self.windows_per_plane = windows_per_plane

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
            raise ValueError(
                "prediction and reference volumes must have matching shapes."
            )
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
        for (batch, axis, _), slice_indices, region in zip(
            located.locations,
            located.relations,
            located.regions,
            strict=True,
        ):
            moved = volume[batch].movedim(axis + 1, 1)
            row, col, height, width = region
            values.append(
                moved[
                    :, list(slice_indices), row : row + height, col : col + width
                ].movedim(0, 1)
            )
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

        triplets, axes, gaps, center_slots, locations, regions = [], [], [], [], [], []
        relations = []
        # One mask transfer replaces scalar GPU reads inside the sampling loops.
        masks = torch.cat((condition.axis_masks, condition.mask), dim=1).detach().cpu()
        for batch in range(volume.shape[0]):
            for axis in AXES:
                moved = volume[batch].movedim(axis + 1, 1)
                own_mask = masks[batch, axis].movedim(axis, 0)
                full_mask = masks[batch, 3].movedim(axis, 0)
                depth, height, width = moved.shape[1:]
                if depth < 3:
                    continue
                own_indices = (
                    own_mask.flatten(1).any(dim=1).nonzero().flatten().tolist()
                )
                candidates = (
                    own_indices
                    or full_mask.flatten(1).any(dim=1).nonzero().flatten().tolist()
                )
                if not candidates:
                    continue
                plane_indices = own_indices or [
                    candidates[self._random_index(len(candidates), generator)]
                ]
                for index_value in plane_indices:
                    points = full_mask[index_value].nonzero()
                    for window in range(self.windows_per_plane):
                        gap = (
                            1
                            if window == 0
                            else self._sample_gap(index_value, depth, generator)
                        )
                        slice_indices = _relation_indices(index_value, depth, gap)
                        if slice_indices is None:
                            continue
                        crop_h = (
                            height
                            if self.windows_per_plane == 1
                            else max(1, height // 2)
                        )
                        crop_w = (
                            width if self.windows_per_plane == 1 else max(1, width // 2)
                        )
                        point = points[
                            self._random_index(len(points), generator)
                        ].tolist()
                        row = min(max(point[0] - crop_h // 2, 0), height - crop_h)
                        col = min(max(point[1] - crop_w // 2, 0), width - crop_w)
                        mask = full_mask[
                            list(slice_indices), row : row + crop_h, col : col + crop_w
                        ]
                        if bool(mask.all()):
                            continue
                        values = moved[
                            :,
                            list(slice_indices),
                            row : row + crop_h,
                            col : col + crop_w,
                        ].movedim(0, 1)
                        triplets.append(values)
                        axes.append(axis)
                        gaps.append(gap)
                        center_slots.append(slice_indices.index(index_value))
                        locations.append((batch, axis, index_value))
                        regions.append((row, col, crop_h, crop_w))
                        relations.append(slice_indices)

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
            tuple(regions),
            tuple(relations),
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
