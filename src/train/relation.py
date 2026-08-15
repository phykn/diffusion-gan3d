import math
from collections import deque
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from ..anchor import AnchorCondition
from ..model.domain import NULL_DOMAIN

MINUS_DIRECTION = 0
PLUS_DIRECTION = 1
SUPPORT_FORWARD = 0
SUPPORT_BACKWARD = 1


@dataclass(frozen=True)
class RelationCurve:
    # Direction order is minus, plus. Distance slot zero means one voxel away.
    values: torch.Tensor
    valid: torch.Tensor
    support: torch.Tensor
    support_valid: torch.Tensor
    descriptor: torch.Tensor
    roi_pixels: int


@dataclass(frozen=True)
class RelationTarget:
    lower: torch.Tensor
    upper: torch.Tensor
    weights: torch.Tensor
    valid: torch.Tensor
    support_lower: torch.Tensor
    support_upper: torch.Tensor
    support_weights: torch.Tensor
    support_valid: torch.Tensor
    ood_weight: float
    source: str


@dataclass(frozen=True)
class RelationPenalty:
    loss: torch.Tensor
    phase: torch.Tensor
    support: torch.Tensor
    minus: torch.Tensor
    plus: torch.Tensor
    distance_weights: torch.Tensor


@dataclass(frozen=True)
class RelationLoss:
    loss: torch.Tensor
    phase: torch.Tensor
    support: torch.Tensor
    minus: torch.Tensor
    plus: torch.Tensor
    queries: int
    matches: int
    domain_matches: int
    shared_matches: int
    ood_rejections: int
    missing_references: int
    distance_weights: torch.Tensor


@dataclass(frozen=True)
class _RelationEntry:
    descriptor: torch.Tensor
    values: torch.Tensor
    valid: torch.Tensor
    support: torch.Tensor
    support_valid: torch.Tensor


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
    """Frozen anchor-free relation statistics with shared-domain fallback."""

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
        support_max_radius: int = 2,
        phase_weight: float = 0.75,
        support_weight: float = 0.25,
        direction_reduction: Literal["mean", "max"] = "mean",
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
        if (
            not isinstance(support_max_radius, int)
            or isinstance(support_max_radius, bool)
            or support_max_radius < 0
        ):
            raise ValueError("support_max_radius must be a non-negative integer.")
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
        for name, value in (
            ("phase_weight", phase_weight),
            ("support_weight", support_weight),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and non-negative.")
        if phase_weight + support_weight <= 0.0:
            raise ValueError("at least one relation component weight must be positive.")
        if direction_reduction not in ("mean", "max"):
            raise ValueError("direction_reduction must be 'mean' or 'max'.")
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
        self.support_max_radius = support_max_radius
        self.phase_weight = float(phase_weight)
        self.support_weight = float(support_weight)
        self.direction_reduction = direction_reduction
        self._shared: dict[int, dict[int, _Bucket]] = {
            axis: {} for axis in self.axes
        }
        self._domains: tuple[dict[int, _Bucket], ...] = tuple(
            {axis: _Bucket(capacity_per_axis) for axis in self.axes}
            for _ in range(num_domains)
        )
        self._roi_shapes: dict[int, deque[tuple[float, float]]] = {
            axis: deque(maxlen=capacity_per_axis) for axis in self.axes
        }

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

    def prior_ready(self, owned_axes: dict[int, tuple[int, ...]]) -> bool:
        if set(owned_axes) != set(range(self.num_domains)):
            raise ValueError("owned relation axes must cover every domain.")
        for domain, axes in owned_axes.items():
            for axis in axes:
                if axis not in self.axes:
                    raise ValueError("owned relation axes must be active axes.")
                shared = self._shared[axis].get(domain)
                if shared is None or not shared.ready:
                    return False
                if not self._domains[domain][axis].ready:
                    return False
        return True

    def observe_shape(
        self,
        *,
        axis: int,
        height: int,
        width: int,
        volume_size: int,
    ) -> None:
        if axis not in self.axes:
            return
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            for value in (height, width, volume_size)
        ):
            raise ValueError("relation ROI dimensions must be positive integers.")
        if height > volume_size or width > volume_size:
            raise ValueError("relation ROI dimensions must fit the volume plane.")
        profile = (height / volume_size, width / volume_size)
        profiles = self._roi_shapes[axis]
        if profile not in profiles:
            profiles.append(profile)

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
                    roi = self._sample_roi(
                        axis,
                        size,
                        device=probs.device,
                        generator=generator,
                    )
                    curve = relation_curve(
                        probs[batch],
                        axis=axis,
                        index=center,
                        roi=roi,
                        support_max_radius=self.support_max_radius,
                    )
                    entry = _RelationEntry(
                        descriptor=curve.descriptor.to(
                            device="cpu",
                            dtype=torch.float16,
                        ),
                        values=curve.values.to(device="cpu", dtype=torch.float16),
                        valid=curve.valid.to(device="cpu"),
                        support=curve.support.to(
                            device="cpu",
                            dtype=torch.float16,
                        ),
                        support_valid=curve.support_valid.to(device="cpu"),
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
        penalties = []
        queries = 0
        domain_matches = 0
        shared_matches = 0
        ood_rejections = 0
        missing_references = 0
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
                    descriptor = matching_descriptor(observed_probs, roi)
                    queries += 1
                    target, status = self._find_target(
                        descriptor,
                        axis=axis,
                        domain=domain,
                        roi_pixels=int(roi.sum().item()),
                        device=probs.device,
                    )
                    if target is None:
                        if status == "ood":
                            ood_rejections += 1
                        else:
                            missing_references += 1
                        continue
                    curve = relation_curve(
                        probs[batch],
                        axis=axis,
                        index=index,
                        roi=roi,
                        frozen_mask=frozen,
                        support_max_radius=self.support_max_radius,
                    )
                    penalties.append(
                        relation_penalty(
                            curve,
                            target,
                            phase_weight=self.phase_weight,
                            support_weight=self.support_weight,
                            direction_reduction=self.direction_reduction,
                        )
                    )
                    if target.source == "domain":
                        domain_matches += 1
                    else:
                        shared_matches += 1
        if penalties:
            loss = torch.stack([item.loss for item in penalties]).mean()
            phase = torch.stack([item.phase for item in penalties]).mean()
            support = torch.stack([item.support for item in penalties]).mean()
            minus = torch.stack([item.minus for item in penalties]).mean()
            plus = torch.stack([item.plus for item in penalties]).mean()
            distance_weights = torch.stack(
                [item.distance_weights for item in penalties]
            ).mean(dim=0)
        else:
            loss = phase = support = minus = plus = zero
            distance_weights = probs.new_zeros(probs.shape[2] - 1)
        return RelationLoss(
            loss=loss,
            phase=phase,
            support=support,
            minus=minus,
            plus=plus,
            queries=queries,
            matches=len(penalties),
            domain_matches=domain_matches,
            shared_matches=shared_matches,
            ood_rejections=ood_rejections,
            missing_references=missing_references,
            distance_weights=distance_weights,
        )

    def _find_target(
        self,
        descriptor: torch.Tensor,
        *,
        axis: int,
        domain: int,
        roi_pixels: int,
        device: torch.device,
    ) -> tuple[RelationTarget | None, Literal["matched", "ood", "missing"]]:
        saw_ood = False
        if domain != NULL_DOMAIN:
            domain_bucket = self._domains[domain][axis]
            if domain_bucket.ready:
                target, rejected = self._match(
                    tuple(domain_bucket.entries),
                    descriptor,
                    roi_pixels,
                    device,
                    source="domain",
                )
                saw_ood |= rejected
                if target is not None:
                    return target, "matched"
        shared_entries = tuple(
            entry
            for bucket in self._shared[axis].values()
            if bucket.ready
            for entry in bucket.entries
        )
        if len(shared_entries) < self.neighbors:
            return None, "ood" if saw_ood else "missing"
        target, rejected = self._match(
            shared_entries,
            descriptor,
            roi_pixels,
            device,
            source="shared",
        )
        saw_ood |= rejected
        if target is None:
            return None, "ood" if saw_ood else "missing"
        return target, "matched"

    def _match(
        self,
        entries: tuple[_RelationEntry, ...],
        query: torch.Tensor,
        roi_pixels: int,
        device: torch.device,
        *,
        source: str,
    ) -> tuple[RelationTarget | None, bool]:
        if len(entries) < self.neighbors:
            return None, False
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
        distances = (normalized - normalized_query).square().mean(dim=1).sqrt()
        ood_weight = self._ood_weight(normalized, float(distances.min()))
        if ood_weight == 0.0:
            return None, True
        nearest = distances.argsort()[: self.neighbors]
        selected = tuple(entries[int(index)] for index in nearest)
        phase_stack = torch.stack(
            [entry.values.to(torch.float32) for entry in selected]
        )
        phase_valid_stack = torch.stack([entry.valid for entry in selected])
        support_stack = torch.stack(
            [entry.support.to(torch.float32) for entry in selected]
        )
        support_valid_stack = torch.stack(
            [entry.support_valid for entry in selected]
        )
        distance_count = phase_stack.shape[2]
        lower = torch.zeros(
            distance_count,
            self.num_phases,
            self.num_phases,
            dtype=torch.float32,
        )
        upper = torch.zeros_like(lower)
        midpoint = torch.zeros_like(lower)
        valid = torch.zeros(distance_count, dtype=torch.bool)
        phase_coverage = torch.zeros(distance_count, dtype=torch.float32)
        for distance in range(distance_count):
            selected_mask = phase_valid_stack[:, :, distance]
            values = phase_stack[:, :, distance][selected_mask]
            if values.shape[0] < 2:
                continue
            lower[distance] = torch.quantile(values, self.quantile_low, dim=0)
            upper[distance] = torch.quantile(values, self.quantile_high, dim=0)
            midpoint[distance] = torch.quantile(values, 0.5, dim=0)
            phase_coverage[distance] = values.shape[0] / (2 * self.neighbors)
            valid[distance] = True

        support_shape = support_stack.shape[3:]
        support_lower = torch.zeros(
            distance_count,
            *support_shape,
            dtype=torch.float32,
        )
        support_upper = torch.zeros_like(support_lower)
        support_midpoint = torch.zeros_like(support_lower)
        support_valid = torch.zeros_like(support_lower, dtype=torch.bool)
        support_coverage = torch.zeros_like(support_lower)
        for distance in range(distance_count):
            for radius in range(support_shape[0]):
                for phase in range(self.num_phases):
                    for direction in (SUPPORT_FORWARD, SUPPORT_BACKWARD):
                        mask = support_valid_stack[
                            :,
                            :,
                            distance,
                            radius,
                            phase,
                            direction,
                        ]
                        values = support_stack[
                            :,
                            :,
                            distance,
                            radius,
                            phase,
                            direction,
                        ][mask]
                        if values.shape[0] < 2:
                            continue
                        key = (distance, radius, phase, direction)
                        support_lower[key] = torch.quantile(
                            values,
                            self.quantile_low,
                        )
                        support_upper[key] = torch.quantile(
                            values,
                            self.quantile_high,
                        )
                        support_midpoint[key] = torch.quantile(values, 0.5)
                        support_coverage[key] = values.shape[0] / (
                            2 * self.neighbors
                        )
                        support_valid[key] = True

        phase_weights = _learned_group_weights(
            midpoint,
            upper - lower,
            phase_coverage,
            valid,
            floor,
            feature_dims=(-2, -1),
        )
        support_group_valid = support_valid.any(dim=(-2, -1))
        support_weights = _learned_group_weights(
            support_midpoint,
            support_upper - support_lower,
            support_coverage,
            support_group_valid,
            floor,
            feature_dims=(-2, -1),
            feature_valid=support_valid,
        )
        if not bool(phase_weights.sum() > 0.0) and not bool(
            support_weights.sum() > 0.0
        ):
            return None, False
        return (
            RelationTarget(
                lower=lower.to(device=device),
                upper=upper.to(device=device),
                weights=phase_weights.to(device=device),
                valid=valid.to(device=device),
                support_lower=support_lower.to(device=device),
                support_upper=support_upper.to(device=device),
                support_weights=support_weights.to(device=device),
                support_valid=support_valid.to(device=device),
                ood_weight=ood_weight,
                source=source,
            ),
            False,
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
        if upper <= lower or query_distance >= upper:
            return 0.0
        return (upper - query_distance) / (upper - lower)

    def _sample_roi(
        self,
        axis: int,
        size: int,
        *,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        profiles = self._roi_shapes[axis]
        if not profiles:
            return torch.ones(size, size, device=device, dtype=torch.bool)
        profile_index = int(
            torch.randint(len(profiles), (), generator=generator).item()
        )
        height_fraction, width_fraction = profiles[profile_index]
        height = min(size, max(1, round(height_fraction * size)))
        width = min(size, max(1, round(width_fraction * size)))
        top = int(torch.randint(size - height + 1, (), generator=generator).item())
        left = int(torch.randint(size - width + 1, (), generator=generator).item())
        roi = torch.zeros(size, size, device=device, dtype=torch.bool)
        roi[top : top + height, left : left + width] = True
        return roi

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
    support_max_radius: int = 0,
) -> RelationCurve:
    if probs.ndim != 4 or not probs.is_floating_point():
        raise ValueError("relation volume must have shape [C, D, H, W].")
    if axis not in (0, 1, 2):
        raise ValueError("relation axis must be 0, 1, or 2.")
    if (
        not isinstance(support_max_radius, int)
        or isinstance(support_max_radius, bool)
        or support_max_radius < 0
    ):
        raise ValueError("support_max_radius must be a non-negative integer.")
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
    channels = probs.shape[0]
    values = torch.zeros(
        2,
        size - 1,
        channels,
        channels,
        device=probs.device,
        dtype=torch.float32,
    )
    valid = torch.zeros(2, size - 1, device=probs.device, dtype=torch.bool)
    support = torch.zeros(
        2,
        size - 1,
        support_max_radius,
        channels,
        2,
        device=probs.device,
        dtype=torch.float32,
    )
    support_valid = torch.zeros_like(support, dtype=torch.bool)
    mask = roi.to(torch.float32)
    area = mask.sum()
    center_masked = center * mask
    center_fraction = center_masked.sum(dim=(-2, -1)) / area
    for direction, neighbor_indices in (
        (MINUS_DIRECTION, tuple(range(index - 1, -1, -1))),
        (PLUS_DIRECTION, tuple(range(index + 1, size))),
    ):
        if not neighbor_indices:
            continue
        neighbors = planes[:, neighbor_indices].movedim(1, 0)
        neighbors_masked = neighbors * mask
        joint = torch.einsum(
            "chw,nkhw->nck",
            center_masked,
            neighbors_masked,
        ) / area
        neighbor_fraction = neighbors_masked.sum(dim=(-2, -1)) / area
        corrected = joint - torch.einsum(
            "c,nk->nck",
            center_fraction,
            neighbor_fraction,
        )
        distance_indices = torch.arange(
            len(neighbor_indices),
            device=probs.device,
        )
        values[direction, distance_indices] = corrected
        valid[direction, distance_indices] = True
        if support_max_radius:
            side_support, side_valid = support_relation(
                center,
                neighbors,
                roi,
                max_radius=support_max_radius,
            )
            support[direction, distance_indices] = side_support
            support_valid[direction, distance_indices] = side_valid
    return RelationCurve(
        values=values,
        valid=valid,
        support=support,
        support_valid=support_valid,
        descriptor=matching_descriptor(center, roi),
        roi_pixels=int(area.item()),
    )


def support_relation(
    center: torch.Tensor,
    neighbors: torch.Tensor,
    roi: torch.Tensor,
    *,
    max_radius: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if center.ndim != 3 or neighbors.ndim != 4:
        raise ValueError("support inputs must be [C,H,W] and [N,C,H,W].")
    if neighbors.shape[1:] != center.shape:
        raise ValueError("support center and neighbor planes must match.")
    if roi.shape != center.shape[1:] or roi.dtype != torch.bool:
        raise ValueError("support ROI must match the plane shape.")
    if not isinstance(max_radius, int) or isinstance(max_radius, bool) or max_radius < 1:
        raise ValueError("support max radius must be a positive integer.")
    values = center.new_zeros(
        neighbors.shape[0],
        max_radius,
        center.shape[0],
        2,
        dtype=torch.float32,
    )
    valid = torch.zeros_like(values, dtype=torch.bool)
    roi_float = roi.to(torch.float32)
    masked_center = center.detach() * roi_float
    masked_neighbors = neighbors * roi_float
    epsilon = torch.finfo(torch.float32).eps
    for radius in range(1, max_radius + 1):
        valid_roi = _erode_roi(roi, radius)
        if not bool(valid_roi.any()):
            continue
        weights = valid_roi.to(torch.float32)
        area = weights.sum()
        floor = area.rsqrt()
        kernel = 2 * radius + 1
        dilated_center = F.max_pool2d(
            masked_center,
            kernel_size=kernel,
            stride=1,
            padding=radius,
        )
        dilated_neighbors = F.max_pool2d(
            masked_neighbors,
            kernel_size=kernel,
            stride=1,
            padding=radius,
        )

        neighbor_mass = (neighbors * weights).sum(dim=(-2, -1))
        center_baseline = (dilated_center * weights).sum(dim=(-2, -1)) / area
        forward_raw = (
            neighbors * dilated_center.unsqueeze(0) * weights
        ).sum(dim=(-2, -1)) / neighbor_mass.clamp_min(epsilon)
        forward_scale = 1.0 - center_baseline
        forward = (forward_raw - center_baseline) / forward_scale.clamp_min(
            epsilon
        )
        forward_valid = (neighbor_mass > epsilon) & (forward_scale > floor)

        center_mass = (center.detach() * weights).sum(dim=(-2, -1))
        neighbor_baseline = (
            dilated_neighbors * weights
        ).sum(dim=(-2, -1)) / area
        backward_raw = (
            center.detach().unsqueeze(0) * dilated_neighbors * weights
        ).sum(dim=(-2, -1)) / center_mass.clamp_min(epsilon).unsqueeze(0)
        backward_scale = 1.0 - neighbor_baseline
        backward = (backward_raw - neighbor_baseline) / backward_scale.clamp_min(
            epsilon
        )
        backward_valid = (center_mass.unsqueeze(0) > epsilon) & (
            backward_scale > floor
        )

        values[:, radius - 1, :, SUPPORT_FORWARD] = forward
        values[:, radius - 1, :, SUPPORT_BACKWARD] = backward
        valid[:, radius - 1, :, SUPPORT_FORWARD] = forward_valid
        valid[:, radius - 1, :, SUPPORT_BACKWARD] = backward_valid
    return values, valid


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
        pair_mask = mask[tuple(left_mask)] & mask[tuple(right_mask)]
        if bool(pair_mask.any()):
            change = 0.5 * (
                values[tuple(left)] - values[tuple(right)]
            ).abs().sum(dim=0)
            density = change[pair_mask].mean().reshape(1)
        else:
            density = values.new_zeros(1)
        descriptors.append(density)
    return torch.cat(descriptors).detach()


def matching_descriptor(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Use one categorical representation and include the ROI measurement scale."""
    if probs.ndim != 3 or not probs.is_floating_point():
        raise ValueError("matching probabilities must have shape [C, H, W].")
    labels = probs.argmax(dim=0)
    hard = F.one_hot(labels, num_classes=probs.shape[0]).movedim(-1, 0)
    morphology = morphology_descriptor(hard.to(torch.float32), mask)
    points = mask.nonzero()
    height = points[:, 0].amax() - points[:, 0].amin() + 1
    width = points[:, 1].amax() - points[:, 1].amin() + 1
    geometry = torch.stack(
        (
            mask.to(torch.float32).mean(),
            height.to(torch.float32) / mask.shape[0],
            width.to(torch.float32) / mask.shape[1],
            mask.all().to(torch.float32),
        )
    ).to(device=probs.device)
    return torch.cat((morphology, geometry)).detach()


def relation_penalty(
    curve: RelationCurve,
    target: RelationTarget,
    *,
    phase_weight: float,
    support_weight: float,
    direction_reduction: Literal["mean", "max"],
) -> RelationPenalty:
    _validate_penalty(curve, target, phase_weight, support_weight, direction_reduction)
    zero = curve.values.sum() * 0.0
    floor = 1.0 / math.sqrt(max(curve.roi_pixels, 1))
    phase_values, phase_active = _directional_feature_loss(
        curve.values,
        curve.valid.unsqueeze(-1).unsqueeze(-1),
        target.lower,
        target.upper,
        target.valid.unsqueeze(-1).unsqueeze(-1),
        target.weights,
        floor,
        group_feature_dims=(-2, -1),
    )
    support_values, support_active = _directional_feature_loss(
        curve.support,
        curve.support_valid,
        target.support_lower,
        target.support_upper,
        target.support_valid,
        target.support_weights,
        floor,
        group_feature_dims=(-2, -1),
    )
    phase = _reduce_directions(phase_values, phase_active, direction_reduction, zero)
    support = _reduce_directions(
        support_values,
        support_active,
        direction_reduction,
        zero,
    )
    combined = []
    combined_active = []
    for direction in (MINUS_DIRECTION, PLUS_DIRECTION):
        components = []
        weights = []
        if bool(phase_active[direction]) and phase_weight > 0.0:
            components.append(phase_values[direction])
            weights.append(phase_weight)
        if bool(support_active[direction]) and support_weight > 0.0:
            components.append(support_values[direction])
            weights.append(support_weight)
        if components:
            combined.append(
                sum(
                    weight * value
                    for weight, value in zip(weights, components, strict=True)
                )
                / sum(weights)
            )
            combined_active.append(True)
        else:
            combined.append(zero)
            combined_active.append(False)
    combined_values = torch.stack(combined)
    active = torch.tensor(
        combined_active,
        device=curve.values.device,
        dtype=torch.bool,
    )
    loss = _reduce_directions(combined_values, active, direction_reduction, zero)
    ood = float(target.ood_weight)
    phase_distance = target.weights
    support_distance = target.support_weights.sum(dim=1)
    distance_components = []
    distance_component_weights = []
    if bool(phase_distance.sum() > 0.0) and phase_weight > 0.0:
        distance_components.append(phase_distance)
        distance_component_weights.append(phase_weight)
    if bool(support_distance.sum() > 0.0) and support_weight > 0.0:
        distance_components.append(support_distance)
        distance_component_weights.append(support_weight)
    if distance_components:
        distance_weights = sum(
            weight * value
            for weight, value in zip(
                distance_component_weights,
                distance_components,
                strict=True,
            )
        ) / sum(distance_component_weights)
        distance_weights = distance_weights / distance_weights.sum().clamp_min(
            torch.finfo(distance_weights.dtype).eps
        )
    else:
        distance_weights = target.weights.new_zeros(target.weights.shape)
    return RelationPenalty(
        loss=ood * loss,
        phase=ood * phase,
        support=ood * support,
        minus=ood * combined_values[MINUS_DIRECTION],
        plus=ood * combined_values[PLUS_DIRECTION],
        distance_weights=distance_weights,
    )


def relation_interval_loss(
    curve: RelationCurve,
    target: RelationTarget,
    *,
    phase_weight: float = 0.75,
    support_weight: float = 0.25,
    direction_reduction: Literal["mean", "max"] = "mean",
) -> torch.Tensor:
    return relation_penalty(
        curve,
        target,
        phase_weight=phase_weight,
        support_weight=support_weight,
        direction_reduction=direction_reduction,
    ).loss


def _directional_feature_loss(
    values: torch.Tensor,
    curve_valid: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    target_valid: torch.Tensor,
    group_weights: torch.Tensor,
    floor: float,
    *,
    group_feature_dims: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape[1:] != lower.shape or lower.shape != upper.shape:
        raise ValueError("relation values and interval target shapes must match.")
    valid = curve_valid & target_valid.unsqueeze(0)
    width = (upper - lower).clamp_min(floor)
    violation = (F.relu(lower - values) + F.relu(values - upper)) / width
    valid_float = valid.to(violation.dtype)
    numerator = (violation * valid_float).sum(dim=group_feature_dims)
    denominator = valid_float.sum(dim=group_feature_dims)
    per_group = numerator / denominator.clamp_min(1.0)
    group_valid = denominator > 0.0
    weights = group_weights.unsqueeze(0) * group_valid
    group_dims = tuple(range(1, weights.ndim))
    weights = weights / weights.sum(dim=group_dims, keepdim=True).clamp_min(
        torch.finfo(weights.dtype).eps
    )
    directional = (weights * per_group).sum(dim=group_dims)
    active = group_valid.any(dim=group_dims) & (group_weights.sum() > 0.0)
    return directional, active


def _reduce_directions(
    values: torch.Tensor,
    active: torch.Tensor,
    reduction: Literal["mean", "max"],
    zero: torch.Tensor,
) -> torch.Tensor:
    selected = values[active]
    if not selected.numel():
        return zero
    return selected.mean() if reduction == "mean" else selected.max()


def _learned_group_weights(
    midpoint: torch.Tensor,
    spread: torch.Tensor,
    coverage: torch.Tensor,
    group_valid: torch.Tensor,
    floor: float,
    *,
    feature_dims: tuple[int, ...],
    feature_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    if feature_valid is None:
        signal = midpoint.abs().mean(dim=feature_dims)
        width = spread.mean(dim=feature_dims)
        sample_coverage = coverage
    else:
        mask = feature_valid.to(midpoint.dtype)
        count = mask.sum(dim=feature_dims).clamp_min(1.0)
        signal = (midpoint.abs() * mask).sum(dim=feature_dims) / count
        width = (spread * mask).sum(dim=feature_dims) / count
        sample_coverage = (coverage * mask).sum(dim=feature_dims) / count
    sampling_noise = 0.5 * floor
    raw = (signal - sampling_noise).clamp_min(0.0) / (
        signal + width + sampling_noise
    ).clamp_min(torch.finfo(signal.dtype).eps)
    raw = raw * sample_coverage * group_valid
    total = raw.sum()
    return raw / total if bool(total > 0.0) else torch.zeros_like(raw)


def _erode_roi(roi: torch.Tensor, radius: int) -> torch.Tensor:
    outside = (~roi).to(torch.float32).unsqueeze(0).unsqueeze(0)
    padded = F.pad(outside, (radius, radius, radius, radius), value=1.0)
    near_outside = F.max_pool2d(
        padded,
        kernel_size=2 * radius + 1,
        stride=1,
    )
    return near_outside[0, 0] == 0.0


def _validate_penalty(
    curve: RelationCurve,
    target: RelationTarget,
    phase_weight: float,
    support_weight: float,
    direction_reduction: str,
) -> None:
    if curve.values.ndim != 4 or curve.values.shape[0] != 2:
        raise ValueError("phase relation curve must have two directions.")
    if curve.valid.shape != curve.values.shape[:2]:
        raise ValueError("phase relation validity must match directions and distances.")
    if curve.support.ndim != 5 or curve.support.shape[0] != 2:
        raise ValueError("support relation curve has an invalid shape.")
    if curve.support_valid.shape != curve.support.shape:
        raise ValueError("support relation validity must match its values.")
    if target.weights.shape != curve.valid.shape[1:]:
        raise ValueError("phase relation weights must match distances.")
    if target.valid.shape != curve.valid.shape[1:]:
        raise ValueError("phase target validity must match distances.")
    if target.support_weights.shape != curve.support.shape[1:3]:
        raise ValueError("support weights must match distance and radius.")
    if target.support_valid.shape != curve.support.shape[1:]:
        raise ValueError("support target validity must match its features.")
    for name, value in (
        ("phase_weight", phase_weight),
        ("support_weight", support_weight),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0.0
        ):
            raise ValueError(f"{name} must be finite and non-negative.")
    if phase_weight + support_weight <= 0.0:
        raise ValueError("at least one relation component weight must be positive.")
    if direction_reduction not in ("mean", "max"):
        raise ValueError("direction_reduction must be 'mean' or 'max'.")
