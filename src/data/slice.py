from dataclasses import dataclass

import torch

from src.anchor import AnchorCondition
from src.data.augment import crop_images
from src.plane import AXES


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


def sample_slices(volume: torch.Tensor, axis: int, count: int) -> torch.Tensor:
    planes = volume.movedim(axis + 2, 1)
    planes = planes.flatten(0, 1)
    indices = torch.randint(planes.shape[0], (count,), device=volume.device)
    return planes[indices]


def sample_pairs(
    previous: torch.Tensor,
    current: torch.Tensor,
    axis: int,
    count: int,
    crop_shape: int | tuple[int, int],
    anchor: AnchorCondition | None = None,
    measured: AnchorCondition | None = None,
    height: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    size = previous.shape[axis + 2]
    excluded = (
        set()
        if measured is None
        else {r.index for r in measured.regions if r.axis == axis}
    )
    active = (
        ()
        if measured is None
        else (
            range(previous.shape[0])
            if measured.active_batches is None
            else measured.active_batches
        )
    )
    candidates = [
        (batch, index)
        for batch in range(previous.shape[0])
        for index in range(size)
        if batch not in active or index not in excluded
    ]
    if not candidates:
        shape = previous.movedim(axis + 2, 2).shape
        empty = previous.new_empty((0, shape[1], *shape[-2:]))
        return (empty, empty) if height is None else (empty, empty, empty[:, :1])
    choices = torch.randint(len(candidates), (count,)).tolist()
    selected = [candidates[i] for i in choices]
    centers = []
    if anchor is not None:
        axes = [normal for normal in AXES if normal != axis]
        for slot, (batch, index) in enumerate(selected[: max(1, count // 2)]):
            regions = [
                r
                for r in anchor.regions
                if r.axis != axis
                and (anchor.active_batches is None or batch in anchor.active_batches)
            ]
            if not regions:
                break
            region = regions[int(torch.randint(len(regions), ()))]
            coordinates = {region.axis: region.index}
            other = [normal for normal in AXES if normal != region.axis]
            coordinates[other[0]] = region.row + int(torch.randint(region.height, ()))
            coordinates[other[1]] = region.col + int(torch.randint(region.width, ()))
            start = region.row if other[0] == axis else region.col
            length = region.height if other[0] == axis else region.width
            allowed = [
                i for b, i in candidates if b == batch and start <= i < start + length
            ]
            if not allowed:
                break
            selected[slot] = (batch, allowed[int(torch.randint(len(allowed), ()))])
            centers.append(tuple(coordinates[normal] for normal in axes))
    batch_indices, plane_indices = zip(*selected)
    batch_indices = torch.tensor(batch_indices, device=previous.device)
    plane_indices = torch.tensor(plane_indices, device=previous.device)
    previous = previous.movedim(axis + 2, 2)[batch_indices, :, plane_indices]
    current = current.movedim(axis + 2, 2)[batch_indices, :, plane_indices]
    channels = previous.shape[1]
    values = (previous, current)
    if height is not None:
        values += (height.movedim(axis + 2, 2)[batch_indices, :, plane_indices],)
    pairs = crop_images(
        torch.cat(values, dim=1),
        crop_shape,
        centers,
    )
    result = (pairs[:, :channels], pairs[:, channels : 2 * channels])
    return result if height is None else (*result, pairs[:, 2 * channels :].detach())


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
        active = (
            range(volume.shape[0])
            if condition.active_batches is None
            else condition.active_batches
        )
        for batch in active:
            for axis in AXES:
                moved = volume[batch].movedim(axis + 1, 1)
                depth, height, width = moved.shape[1:]
                if depth < 3:
                    continue
                own = [region for region in condition.regions if region.axis == axis]
                selected = own or list(condition.regions[:1])
                for region in selected:
                    other = [normal for normal in AXES if normal != region.axis]
                    coordinates = {
                        region.axis: region.index,
                        other[0]: region.row
                        + self._random_index(region.height, generator),
                        other[1]: region.col
                        + self._random_index(region.width, generator),
                    }
                    index_value = coordinates[axis]
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
                        coordinates[other[0]] = region.row + self._random_index(
                            region.height, generator
                        )
                        coordinates[other[1]] = region.col + self._random_index(
                            region.width, generator
                        )
                        tangents = [normal for normal in AXES if normal != axis]
                        row = min(
                            max(coordinates[tangents[0]] - crop_h // 2, 0),
                            height - crop_h,
                        )
                        col = min(
                            max(coordinates[tangents[1]] - crop_w // 2, 0),
                            width - crop_w,
                        )
                        if all(
                            any(
                                r.index == i
                                and r.row <= row
                                and r.col <= col
                                and r.row + r.height >= row + crop_h
                                and r.col + r.width >= col + crop_w
                                for r in own
                            )
                            for i in slice_indices
                        ):
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
