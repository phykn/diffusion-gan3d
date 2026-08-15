import math
from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..anchor import AnchorCondition
from ..model.domain import NULL_DOMAIN


@dataclass(frozen=True)
class RelationCurve:
    values: torch.Tensor
    valid: torch.Tensor
    descriptor: torch.Tensor
    roi_pixels: int


@dataclass(frozen=True)
class RelationTarget:
    lower: torch.Tensor
    upper: torch.Tensor
    weights: torch.Tensor
    valid: torch.Tensor
    ood_weight: float
    source: str


@dataclass(frozen=True)
class RelationLoss:
    loss: torch.Tensor
    queries: int
    matches: int
    domain_matches: int
    shared_matches: int


@dataclass(frozen=True)
class _RelationEntry:
    descriptor: torch.Tensor
    values: torch.Tensor
    valid: torch.Tensor


class _Bucket:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.entries: deque[_RelationEntry] = deque()

    @property
    def ready(self) -> bool:
        return len(self.entries) == self.capacity

    def add(self, entry: _RelationEntry) -> None:
        if not self.ready:
            self.entries.append(entry)


class RelationBank:
    """Frozen EMA relation statistics with domain-to-shared fallback."""

    def __init__(
        self,
        *,
        num_domains: int,
        num_phases: int,
        axes: tuple[int, ...],
        capacity_per_axis: int,
        profiles_per_axis: int,
        neighbors: int,
        quantile_low: float,
        quantile_high: float,
    ) -> None:
        for name, value in (
            ("num_domains", num_domains),
            ("num_phases", num_phases),
            ("capacity_per_axis", capacity_per_axis),
            ("profiles_per_axis", profiles_per_axis),
            ("neighbors", neighbors),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not axes or any(axis not in (0, 1, 2) for axis in axes):
            raise ValueError("relation axes must contain values from 0, 1, and 2.")
        if len(set(axes)) != len(axes):
            raise ValueError("relation axes must be unique.")
        for name, value in (
            ("quantile_low", quantile_low),
            ("quantile_high", quantile_high),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be between zero and one.")
        if quantile_low >= quantile_high:
            raise ValueError("relation quantiles must be strictly increasing.")
        if neighbors < 2:
            raise ValueError("relation neighbors must be at least two.")
        if neighbors > capacity_per_axis:
            raise ValueError("relation neighbors must not exceed bank capacity.")

        self.num_domains = num_domains
        self.num_phases = num_phases
        self.axes = tuple(axes)
        self.capacity_per_axis = capacity_per_axis
        self.profiles_per_axis = profiles_per_axis
        self.neighbors = neighbors
        self.quantile_low = float(quantile_low)
        self.quantile_high = float(quantile_high)
        self._shared: dict[int, dict[int, _Bucket]] = {
            axis: {} for axis in self.axes
        }
        self._domains: tuple[dict[int, _Bucket], ...] = tuple(
            {
                axis: _Bucket(capacity_per_axis)
                for axis in self.axes
            }
            for _ in range(num_domains)
        )

    @property
    def entry_count(self) -> int:
        domain_count = sum(
            len(bucket.entries)
            for buckets in self._domains
            for bucket in buckets.values()
        )
        shared_count = sum(
            len(bucket.entries)
            for sources in self._shared.values()
            for bucket in sources.values()
        )
        return domain_count + shared_count

    @property
    def ready_bucket_count(self) -> int:
        domain_count = sum(
            bucket.ready
            for buckets in self._domains
            for bucket in buckets.values()
        )
        shared_count = sum(
            bucket.ready
            for sources in self._shared.values()
            for bucket in sources.values()
        )
        return int(domain_count + shared_count)

    def needs_data(
        self,
        *,
        condition_domain: int,
        source_domain: int,
        owned_axes: tuple[int, ...],
    ) -> bool:
        self._validate_source(condition_domain, source_domain, owned_axes)
        for axis in self.axes:
            if axis not in owned_axes:
                continue
            shared = self._shared[axis].get(source_domain)
            if shared is None or not shared.ready:
                return True
            if (
                condition_domain != NULL_DOMAIN
                and not self._domains[condition_domain][axis].ready
            ):
                return True
        return False

    @torch.no_grad()
    def add(
        self,
        probs: torch.Tensor,
        *,
        condition_domain: int,
        source_domain: int,
        owned_axes: tuple[int, ...],
        generator: torch.Generator | None = None,
    ) -> None:
        self._validate_probs(probs)
        self._validate_source(condition_domain, source_domain, owned_axes)
        size = probs.shape[2]
        count = min(size, self.profiles_per_axis)
        for axis in self.axes:
            if axis not in owned_axes:
                continue
            centers = torch.randperm(size, generator=generator)[:count].tolist()
            for batch in range(probs.shape[0]):
                for center in centers:
                    roi_shape = tuple(
                        size_value
                        for index, size_value in enumerate(probs.shape[2:])
                        if index != axis
                    )
                    roi = torch.ones(roi_shape, device=probs.device, dtype=torch.bool)
                    curve = relation_curve(
                        probs[batch],
                        axis=axis,
                        index=center,
                        roi=roi,
                    )
                    entry = _RelationEntry(
                        descriptor=curve.descriptor.to(
                            device="cpu",
                            dtype=torch.float16,
                        ),
                        values=curve.values.to(device="cpu", dtype=torch.float16),
                        valid=curve.valid.to(device="cpu"),
                    )
                    sources = self._shared[axis]
                    shared = sources.setdefault(
                        source_domain,
                        _Bucket(self.capacity_per_axis),
                    )
                    shared.add(entry)
                    if condition_domain != NULL_DOMAIN:
                        self._domains[condition_domain][axis].add(entry)

    def loss(
        self,
        probs: torch.Tensor,
        condition: AnchorCondition,
        visible: torch.Tensor,
        *,
        domain: int,
    ) -> RelationLoss:
        self._validate_probs(probs)
        if visible.shape != (probs.shape[0],) or visible.dtype != torch.bool:
            raise ValueError("relation visibility must be boolean with shape [B].")
        if visible.device != probs.device:
            raise ValueError("relation visibility and probabilities must share a device.")
        if domain != NULL_DOMAIN and not 0 <= domain < self.num_domains:
            raise ValueError("relation domain is outside the configured domains.")
        zero = probs.sum() * 0.0
        losses = []
        queries = 0
        domain_matches = 0
        shared_matches = 0
        for batch in visible.nonzero().flatten().tolist():
            frozen = condition.mask[batch, 0]
            for axis in self.axes:
                axis_mask = condition.axis_masks[batch, axis]
                moved_mask = axis_mask.movedim(axis, 0)
                indices = moved_mask.flatten(1).any(dim=1).nonzero().flatten()
                for index_tensor in indices:
                    index = int(index_tensor.item())
                    roi = moved_mask[index]
                    observed = condition.target[batch].select(axis, index)
                    observed_probs = (
                        F.one_hot(observed, num_classes=self.num_phases)
                        .movedim(-1, 0)
                        .to(device=probs.device, dtype=torch.float32)
                    )
                    descriptor = morphology_descriptor(observed_probs, roi)
                    queries += 1
                    target = self._find_target(
                        descriptor,
                        axis=axis,
                        domain=domain,
                        roi_pixels=int(roi.sum().item()),
                        device=probs.device,
                    )
                    if target is None:
                        continue
                    curve = relation_curve(
                        probs[batch],
                        axis=axis,
                        index=index,
                        roi=roi,
                        frozen_mask=frozen,
                    )
                    losses.append(relation_interval_loss(curve, target))
                    if target.source == "domain":
                        domain_matches += 1
                    else:
                        shared_matches += 1
        loss = zero if not losses else torch.stack(losses).mean()
        return RelationLoss(
            loss=loss,
            queries=queries,
            matches=len(losses),
            domain_matches=domain_matches,
            shared_matches=shared_matches,
        )

    def _find_target(
        self,
        descriptor: torch.Tensor,
        *,
        axis: int,
        domain: int,
        roi_pixels: int,
        device: torch.device,
    ) -> RelationTarget | None:
        if domain != NULL_DOMAIN:
            domain_bucket = self._domains[domain][axis]
            if domain_bucket.ready:
                target = self._match(
                    tuple(domain_bucket.entries),
                    descriptor,
                    roi_pixels,
                    device,
                    source="domain",
                )
                if target is not None and target.ood_weight > 0.0:
                    return target
        shared_entries = tuple(
            entry
            for bucket in self._shared[axis].values()
            if bucket.ready
            for entry in bucket.entries
        )
        if len(shared_entries) < self.neighbors:
            return None
        target = self._match(
            shared_entries,
            descriptor,
            roi_pixels,
            device,
            source="shared",
        )
        if target is None or target.ood_weight == 0.0:
            return None
        return target

    def _match(
        self,
        entries: tuple[_RelationEntry, ...],
        query: torch.Tensor,
        roi_pixels: int,
        device: torch.device,
        *,
        source: str,
    ) -> RelationTarget | None:
        if len(entries) < self.neighbors:
            return None
        descriptors = torch.stack(
            [entry.descriptor.to(torch.float32) for entry in entries]
        )
        query = query.detach().to(device="cpu", dtype=torch.float32)
        floor = 1.0 / math.sqrt(max(roi_pixels, 1))
        median = descriptors.median(dim=0).values
        lower_quartile = torch.quantile(descriptors, 0.25, dim=0)
        upper_quartile = torch.quantile(descriptors, 0.75, dim=0)
        scale = (upper_quartile - lower_quartile).clamp_min(floor)
        normalized = (descriptors - median) / scale
        normalized_query = (query - median) / scale
        distances = (
            (normalized - normalized_query).square().mean(dim=1).sqrt()
        )
        ood_weight = self._ood_weight(normalized, float(distances.min()))
        nearest = distances.argsort()[: self.neighbors]
        selected = tuple(entries[int(index)] for index in nearest)
        distance_count = selected[0].values.shape[0]
        lower = torch.zeros(
            distance_count,
            self.num_phases,
            self.num_phases,
            dtype=torch.float32,
        )
        upper = torch.zeros_like(lower)
        midpoint = torch.zeros_like(lower)
        coverage = torch.zeros(distance_count, dtype=torch.float32)
        valid = torch.zeros(distance_count, dtype=torch.bool)
        for distance in range(distance_count):
            values = torch.stack(
                [
                    entry.values[distance].to(torch.float32)
                    for entry in selected
                    if bool(entry.valid[distance])
                ]
            ) if any(bool(entry.valid[distance]) for entry in selected) else None
            if values is None or values.shape[0] < 2:
                continue
            lower[distance] = torch.quantile(
                values,
                self.quantile_low,
                dim=0,
            )
            upper[distance] = torch.quantile(
                values,
                self.quantile_high,
                dim=0,
            )
            midpoint[distance] = torch.quantile(values, 0.5, dim=0)
            coverage[distance] = values.shape[0] / self.neighbors
            valid[distance] = True
        signal = midpoint.abs().mean(dim=(-2, -1))
        spread = (upper - lower).mean(dim=(-2, -1))
        sampling_noise = 0.5 * floor
        raw = (signal - sampling_noise).clamp_min(0.0) / (
            signal + spread + sampling_noise
        ).clamp_min(torch.finfo(signal.dtype).eps)
        raw = raw * coverage * valid
        total = raw.sum()
        if not bool(total > 0.0):
            return None
        weights = raw / total
        return RelationTarget(
            lower=lower.to(device=device),
            upper=upper.to(device=device),
            weights=weights.to(device=device),
            valid=valid.to(device=device),
            ood_weight=ood_weight,
            source=source,
        )

    @staticmethod
    def _ood_weight(normalized: torch.Tensor, query_distance: float) -> float:
        if normalized.shape[0] < 2:
            return 0.0
        pairwise = torch.cdist(normalized, normalized) / math.sqrt(
            normalized.shape[1]
        )
        pairwise.fill_diagonal_(float("inf"))
        nearest = pairwise.min(dim=1).values
        lower = float(torch.quantile(nearest, 0.95))
        upper = float(torch.quantile(nearest, 0.99))
        if query_distance <= lower:
            return 1.0
        if upper <= lower:
            return 0.0
        if query_distance >= upper:
            return 0.0
        return (upper - query_distance) / (upper - lower)

    def _validate_probs(self, probs: torch.Tensor) -> None:
        if (
            not isinstance(probs, torch.Tensor)
            or not probs.is_floating_point()
            or probs.ndim != 5
            or probs.shape[1] != self.num_phases
            or len(set(probs.shape[2:])) != 1
        ):
            raise ValueError("relation probabilities must have shape [B, C, S, S, S].")

    def _validate_source(
        self,
        condition_domain: int,
        source_domain: int,
        owned_axes: tuple[int, ...],
    ) -> None:
        if (
            condition_domain != NULL_DOMAIN
            and not 0 <= condition_domain < self.num_domains
        ):
            raise ValueError("relation condition domain is invalid.")
        if not 0 <= source_domain < self.num_domains:
            raise ValueError("relation source domain is invalid.")
        if any(axis not in self.axes for axis in owned_axes):
            raise ValueError("owned relation axes must be active axes.")


def relation_curve(
    probs: torch.Tensor,
    *,
    axis: int,
    index: int,
    roi: torch.Tensor,
    frozen_mask: torch.Tensor | None = None,
) -> RelationCurve:
    if probs.ndim != 4 or not probs.is_floating_point():
        raise ValueError("relation volume must have shape [C, D, H, W].")
    if axis not in (0, 1, 2):
        raise ValueError("relation axis must be 0, 1, or 2.")
    planes = probs.float().movedim(axis + 1, 1)
    size = planes.shape[1]
    if size < 2:
        raise ValueError("relation volume must contain at least two planes.")
    if not 0 <= index < size:
        raise ValueError("relation center index is outside the volume.")
    if roi.shape != planes.shape[2:] or roi.dtype != torch.bool:
        raise ValueError("relation ROI must be a boolean mask matching one plane.")
    if roi.device != probs.device or not bool(roi.any()):
        raise ValueError("relation ROI must be non-empty and on the volume device.")
    if frozen_mask is not None:
        if frozen_mask.shape != probs.shape[1:] or frozen_mask.dtype != torch.bool:
            raise ValueError("frozen relation mask must match the volume.")
        moved_frozen = frozen_mask.movedim(axis, 0)
        planes = torch.where(moved_frozen.unsqueeze(0), planes.detach(), planes)

    center = planes[:, index].detach()
    neighbor_indices = [value for value in range(size) if value != index]
    distances = torch.tensor(
        [abs(value - index) - 1 for value in neighbor_indices],
        device=probs.device,
        dtype=torch.long,
    )
    neighbors = planes[:, neighbor_indices].movedim(1, 0)
    mask = roi.to(torch.float32)
    area = mask.sum()
    center_masked = center * mask
    neighbors_masked = neighbors * mask
    joint = torch.einsum(
        "chw,nkhw->nck",
        center_masked,
        neighbors_masked,
    ) / area
    center_fraction = center_masked.sum(dim=(-2, -1)) / area
    neighbor_fraction = neighbors_masked.sum(dim=(-2, -1)) / area
    corrected = joint - torch.einsum(
        "c,nk->nck",
        center_fraction,
        neighbor_fraction,
    )
    values = torch.zeros(
        size - 1,
        probs.shape[0],
        probs.shape[0],
        device=probs.device,
        dtype=torch.float32,
    ).index_add(0, distances, corrected)
    counts = torch.zeros(
        size - 1,
        device=probs.device,
        dtype=torch.float32,
    ).index_add(0, distances, torch.ones_like(distances, dtype=torch.float32))
    valid = counts > 0.0
    values = values / counts.clamp_min(1.0).reshape(-1, 1, 1)
    return RelationCurve(
        values=values,
        valid=valid,
        descriptor=morphology_descriptor(center, roi),
        roi_pixels=int(area.item()),
    )


def morphology_descriptor(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if probs.ndim != 3 or not probs.is_floating_point():
        raise ValueError("descriptor probabilities must have shape [C, H, W].")
    if mask.shape != probs.shape[1:] or mask.dtype != torch.bool:
        raise ValueError("descriptor mask must match the probability plane.")
    if mask.device != probs.device or not bool(mask.any()):
        raise ValueError("descriptor mask must be non-empty and share the device.")
    values = probs.float()
    weights = mask.to(torch.float32)
    fractions = (values * weights).sum(dim=(-2, -1)) / weights.sum()
    descriptors = [fractions[:-1]]
    for dimension in (1, 2):
        left = [slice(None), slice(None), slice(None)]
        right = [slice(None), slice(None), slice(None)]
        left[dimension] = slice(None, -1)
        right[dimension] = slice(1, None)
        left_mask = [slice(None), slice(None)]
        right_mask = [slice(None), slice(None)]
        left_mask[dimension - 1] = slice(None, -1)
        right_mask[dimension - 1] = slice(1, None)
        valid = mask[tuple(left_mask)] & mask[tuple(right_mask)]
        if bool(valid.any()):
            change = 0.5 * (
                values[tuple(left)] - values[tuple(right)]
            ).abs().sum(dim=0)
            density = change[valid].mean().reshape(1)
        else:
            density = values.new_zeros(1)
        descriptors.append(density)
    return torch.cat(descriptors).detach()


def relation_interval_loss(
    curve: RelationCurve,
    target: RelationTarget,
) -> torch.Tensor:
    if curve.values.shape != target.lower.shape or target.lower.shape != target.upper.shape:
        raise ValueError("relation curve and interval target shapes must match.")
    if target.weights.shape != curve.valid.shape or target.valid.shape != curve.valid.shape:
        raise ValueError("relation weights and validity must match the curve distances.")
    valid = curve.valid & target.valid
    if not bool(valid.any()) or target.ood_weight <= 0.0:
        return curve.values.sum() * 0.0
    floor = 1.0 / math.sqrt(max(curve.roi_pixels, 1))
    width = (target.upper - target.lower).clamp_min(floor)
    violation = (
        F.relu(target.lower - curve.values)
        + F.relu(curve.values - target.upper)
    ) / width
    per_distance = violation.mean(dim=(-2, -1))
    weights = target.weights * valid
    weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    return float(target.ood_weight) * (weights * per_distance).sum()
