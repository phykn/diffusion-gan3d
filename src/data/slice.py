from dataclasses import dataclass, replace

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
    height: torch.Tensor | None = None
    profile: torch.Tensor | None = None

    def __len__(self) -> int:
        return self.values.shape[0]


@dataclass(frozen=True)
class LocatedTriplets:
    triplets: TripletBatch
    locations: tuple[tuple[int, int, int], ...]
    regions: tuple[tuple[int, int, int, int], ...] = ()
    relations: tuple[tuple[int, int, int], ...] = ()


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
    active = () if measured is None else measured.active_batches
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
        focused = []
        for region in anchor.regions:
            if region.axis == axis:
                continue
            other = [normal for normal in AXES if normal != region.axis]
            start = region.row if other[0] == axis else region.col
            length = region.height if other[0] == axis else region.width
            focused.extend(
                (batch, index, region)
                for batch, index in candidates
                if batch in anchor.active_batches and start <= index < start + length
            )
        for slot in range(max(1, count // 2) if focused else 0):
            batch, index, region = focused[int(torch.randint(len(focused), ()))]
            coordinates = {region.axis: region.index}
            other = [normal for normal in AXES if normal != region.axis]
            coordinates[other[0]] = region.row + int(torch.randint(region.height, ()))
            coordinates[other[1]] = region.col + int(torch.randint(region.width, ()))
            selected[slot] = (batch, index)
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
        self.max_gap = max_gap
        self.windows_per_plane = windows_per_plane

    def sample(
        self,
        prediction: torch.Tensor,
        reference: torch.Tensor,
        condition: AnchorCondition,
        generator: torch.Generator | None = None,
        height: torch.Tensor | None = None,
    ) -> tuple[TripletBatch, TripletBatch]:
        reference = reference.detach()
        located = self._sample_anchor_triplets(
            prediction,
            condition,
            generator=generator,
        )
        real, fake = self._extract_triplets(reference, located), located.triplets
        if height is not None:
            heights = self._extract_triplets(height, located).values
            real, fake = replace(real, height=heights), replace(fake, height=heights)
        return real, fake

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
        triplets, axes, gaps, center_slots, locations, regions = [], [], [], [], [], []
        relations = []
        active = condition.active_batches
        for batch in active:
            for axis in AXES:
                moved = volume[batch].movedim(axis + 1, 1)
                depth, height, width = moved.shape[1:]
                if depth < 3:
                    continue
                crop_h, crop_w = (
                    (height, width)
                    if self.windows_per_plane == 1
                    else (max(1, height // 2), max(1, width // 2))
                )
                tangents = [normal for normal in AXES if normal != axis]
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
                        coordinates[other[0]] = region.row + self._random_index(
                            region.height, generator
                        )
                        coordinates[other[1]] = region.col + self._random_index(
                            region.width, generator
                        )
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
    if index - gap >= 0 and index + gap < size:
        return index - gap, index, index + gap
    if index + 2 * gap < size:
        return index, index + gap, index + 2 * gap
    if index - 2 * gap >= 0:
        return index - 2 * gap, index - gap, index
    return None


@dataclass(frozen=True)
class BandBatch:
    values: torch.Tensor
    axes: torch.Tensor

    def __len__(self) -> int:
        return self.values.shape[0]


def sample_seam_bands(
    volume: torch.Tensor,
    condition: AnchorCondition | None,
    rows: int,
    width: int,
    count: int,
    directed_axis: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[BandBatch, BandBatch]:
    for name, value in (("rows", rows), ("width", width), ("count", count)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    near, far = [], []
    active = () if condition is None else condition.active_batches
    for batch in active:
        for axis in AXES:
            regions = [region for region in condition.regions if region.axis == axis]
            fixed = condition.mask[batch, 0].movedim(axis, 0)
            if not regions or fixed.shape[0] <= rows:
                continue
            near_sites, far_sites = _seam_sites(fixed, regions, rows, width)
            if axis == directed_axis:
                # Read height-direction rows upward to keep their physical order.
                far_sites = far_sites[far_sites[:, -1] > 0]
                down = near_sites[:, -1] < 0
                near_sites[down, 1] -= rows - 1
                near_sites[down, -1] = 1
            if not len(near_sites) or not len(far_sites):
                continue
            for sites, found in ((near, near_sites), (far, far_sites)):
                sites.append(
                    torch.cat(
                        (found.new_tensor([batch, axis]).expand(len(found), 2), found),
                        1,
                    )
                )
    near = torch.cat(near) if near else None
    far = torch.cat(far) if far else None
    near_keep, far_keep = _matched_sites(near, far, count, generator)
    return (
        _band_batch(
            volume.detach(), far, far_keep, rows, width, directed_axis, generator
        ),
        _band_batch(volume, near, near_keep, rows, width, directed_axis, generator),
    )


def _band_labels(sites: torch.Tensor) -> torch.Tensor:
    lead = (sites[:, 1] == 0).long()
    return torch.where(sites[:, 2].bool(), 3 - sites[:, 1] - lead, lead)


def _matched_sites(near, far, count, generator):
    """Draw near and far bands with equal counts per anchor axis and section plane.

    The critic sees only the section plane, so near and far must also share
    the orientation of rows within it, which the anchor axis determines.
    """
    empty = torch.empty(0, dtype=torch.long)
    if near is None:
        return empty, empty
    near_keys = (near[:, 1] * 3 + _band_labels(near)).cpu()
    far_keys = (far[:, 1] * 3 + _band_labels(far)).cpu()
    near_keep, far_keep = [], []
    for label in AXES:
        pool, far_pools = [], {}
        for axis in AXES:
            key = 3 * axis + label
            near_index = (near_keys == key).nonzero().flatten()
            far_index = (far_keys == key).nonzero().flatten()
            available = min(len(near_index), len(far_index))
            if available:
                order = torch.randperm(len(near_index), generator=generator)
                pool.append(near_index[order[:available]])
                far_pools[key] = far_index
        if not pool:
            continue
        pool = torch.cat(pool)
        chosen = pool[torch.randperm(len(pool), generator=generator)[:count]]
        near_keep.append(chosen)
        for key, far_index in far_pools.items():
            taken = int((near_keys[chosen] == key).sum())
            order = torch.randperm(len(far_index), generator=generator)
            far_keep.append(far_index[order[:taken]])
    if not near_keep:
        return empty, empty
    return torch.cat(near_keep), torch.cat(far_keep)


def _seam_sites(fixed, regions, rows, width):
    length = fixed.shape[0]
    indices = torch.tensor([region.index for region in regions], device=fixed.device)
    distance = (torch.arange(length, device=fixed.device)[:, None] - indices).abs()
    far_starts = (distance.amin(1) > rows).unfold(0, rows, 1).all(-1)
    near_sites, far_sites = [], []
    for flip in (0, 1):
        planes = fixed.transpose(1, 2) if flip else fixed
        clear = ~planes.unfold(0, rows, 1).any(-1).unfold(2, width, 1).any(-1)
        below = torch.zeros_like(clear)
        above = torch.zeros_like(clear)
        for region in regions:
            positions, (low, high) = (
                (
                    slice(region.col, region.col + region.width),
                    (region.row, region.row + region.height),
                )
                if flip
                else (
                    slice(region.row, region.row + region.height),
                    (region.col, region.col + region.width),
                )
            )
            spans = slice(low, max(low, high - width + 1))
            if region.index >= rows:
                below[region.index - rows, positions, spans] = True
            if region.index + rows < length:
                above[region.index + 1, positions, spans] = True
        far = far_starts[:, None, None] & clear
        for sites, mask, step in (
            (near_sites, below & clear, -1),
            (near_sites, above & clear, 1),
            (far_sites, far, -1),
            (far_sites, far, 1),
        ):
            found = mask.nonzero()
            if step < 0:
                found[:, 0] += rows - 1
            sites.append(
                torch.cat(
                    (
                        torch.full_like(found[:, :1], flip),
                        found,
                        torch.full_like(found[:, :1], step),
                    ),
                    1,
                )
            )
    return torch.cat(near_sites), torch.cat(far_sites)


def _band_batch(volume, sites, keep, rows, width, directed_axis, generator):
    if not len(keep):
        return BandBatch(
            values=volume.new_empty((0, volume.shape[1], rows, width)),
            axes=torch.empty(0, device=volume.device, dtype=torch.long),
        )
    sites = sites[keep.to(sites.device)]
    batch, axis, flip, start, position, span, step = sites.unbind(1)
    lead = (axis == 0).long()
    labels = _band_labels(sites)
    mirror = (torch.rand(len(keep), generator=generator) < 0.5).to(sites.device)
    if directed_axis is not None:
        mirror &= 3 - axis - labels != directed_axis
    lines = (
        start[:, None, None]
        + step[:, None, None] * torch.arange(rows, device=sites.device)[:, None]
    )
    offsets = torch.arange(width, device=sites.device)
    offsets = torch.where(mirror[:, None], width - 1 - offsets, offsets)
    across = (span[:, None] + offsets)[:, None]
    flip = flip[:, None, None].bool()
    first = torch.where(flip, across, position[:, None, None])
    second = torch.where(flip, position[:, None, None], across)
    coordinates = [
        torch.where(
            axis[:, None, None] == dim,
            lines,
            torch.where(lead[:, None, None] == dim, first, second),
        ).expand(-1, rows, width)
        for dim in AXES
    ]
    values = volume[(batch[:, None, None], slice(None), *coordinates)]
    return BandBatch(values=values.permute(0, 3, 1, 2), axes=labels)
