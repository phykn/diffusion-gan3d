import math
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F

from ..anchor import AnchorCondition
from ..model.domain import NULL_DOMAIN
from .relation_math import (
    DEFAULT_MIN_CHANCE_GAP,
    DEFAULT_MIN_PHASE_FRACTION,
    DEFAULT_MIN_SUPPORT_PIXELS,
    SUPPORT_DIRECTIONS,
    RelationTarget,
    _aggregate_direction_weights,
    _collapse_phase_weights,
    _empty_float,
    _matched_directional_mean,
    _summarize_phase_target,
    _summarize_support_target,
    _validate_support_guards,
    _zero_float,
    matching_descriptor,
    relation_curve,
    relation_penalty,
)


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
    phase_distance_weights: torch.Tensor = field(default_factory=_empty_float)
    support_distance_weights: torch.Tensor = field(default_factory=_empty_float)
    correlation_strength: torch.Tensor = field(default_factory=_empty_float)
    uncertainty: torch.Tensor = field(default_factory=_empty_float)
    displacement_quantile: torch.Tensor = field(default_factory=_empty_float)
    dilation_radius: torch.Tensor = field(default_factory=_empty_float)
    valid_ratio_by_phase: torch.Tensor = field(default_factory=_empty_float)
    support_forward: torch.Tensor = field(default_factory=_zero_float)
    support_backward: torch.Tensor = field(default_factory=_zero_float)
    long_range: torch.Tensor = field(default_factory=_zero_float)
    matched_reference_count: torch.Tensor = field(default_factory=_empty_float)

    @classmethod
    def empty(
        cls,
        reference: torch.Tensor,
        *,
        distances: int,
        phases: int,
    ) -> "RelationLoss":
        shape = (distances, phases)
        zero = reference.new_zeros(())
        return cls(
            loss=zero,
            phase=zero,
            support=zero,
            minus=zero,
            plus=zero,
            queries=0,
            matches=0,
            domain_matches=0,
            shared_matches=0,
            ood_rejections=0,
            missing_references=0,
            distance_weights=reference.new_zeros(distances),
            phase_distance_weights=reference.new_zeros(shape),
            support_distance_weights=reference.new_zeros(shape),
            correlation_strength=reference.new_zeros(shape),
            uncertainty=reference.new_zeros(shape),
            displacement_quantile=reference.new_full(
                (*shape, SUPPORT_DIRECTIONS),
                -1.0,
            ),
            dilation_radius=reference.new_full(shape, -1.0),
            valid_ratio_by_phase=reference.new_zeros(phases),
            support_forward=zero,
            support_backward=zero,
            long_range=zero,
            matched_reference_count=reference.new_zeros(shape),
        )


@dataclass(frozen=True)
class _RelationEntry:
    descriptor: torch.Tensor
    values: torch.Tensor
    valid: torch.Tensor
    support: torch.Tensor
    support_valid: torch.Tensor
    independent: torch.Tensor
    phase_valid: torch.Tensor
    support_raw: torch.Tensor
    displacement_hist: torch.Tensor
    displacement_valid: torch.Tensor


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


def _sample_profile_centers(
    size: int,
    count: int,
    *,
    generator: torch.Generator | None,
) -> list[int]:
    if count == 1:
        return torch.randperm(size, generator=generator)[:1].tolist()
    interior = (
        (torch.randperm(size - 2, generator=generator)[: count - 2] + 1).tolist()
        if count > 2
        else []
    )
    return [0, size - 1, *interior]


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
        min_support_pixels: int = DEFAULT_MIN_SUPPORT_PIXELS,
        min_phase_fraction: float = DEFAULT_MIN_PHASE_FRACTION,
        min_chance_gap: float = DEFAULT_MIN_CHANCE_GAP,
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
        _validate_support_guards(
            min_support_pixels,
            min_phase_fraction,
            min_chance_gap,
        )
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
        self.min_support_pixels = min_support_pixels
        self.min_phase_fraction = float(min_phase_fraction)
        self.min_chance_gap = float(min_chance_gap)
        self._shared: dict[int, dict[int, _Bucket]] = {axis: {} for axis in self.axes}
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
            bucket.ready for buckets in self._domains for bucket in buckets.values()
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
            not isinstance(value, int) or isinstance(value, bool) or value < 1
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
            centers = _sample_profile_centers(
                size,
                count,
                generator=generator,
            )
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
                        min_support_pixels=self.min_support_pixels,
                        min_phase_fraction=self.min_phase_fraction,
                        min_chance_gap=self.min_chance_gap,
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
                        independent=curve.independent.to(
                            device="cpu",
                            dtype=torch.float16,
                        ),
                        phase_valid=curve.phase_valid.to(device="cpu"),
                        support_raw=curve.support_raw.to(
                            device="cpu",
                            dtype=torch.float16,
                        ),
                        displacement_hist=curve.displacement_hist.to(
                            device="cpu",
                            dtype=torch.float16,
                        ),
                        displacement_valid=curve.displacement_valid.to(device="cpu"),
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
            raise ValueError(
                "relation visibility and probabilities must share a device."
            )
        if domain != NULL_DOMAIN and not 0 <= domain < self.num_domains:
            raise ValueError("relation domain is outside the configured domains.")
        zero = probs.sum() * 0.0
        penalties = []
        matched_targets = []
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
                        min_support_pixels=self.min_support_pixels,
                        min_phase_fraction=self.min_phase_fraction,
                        min_chance_gap=self.min_chance_gap,
                        collect_displacement=False,
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
                    matched_targets.append(target)
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
            phase_distance_weights = torch.stack(
                [item.phase_distance_weights for item in penalties]
            ).mean(dim=0)
            support_distance_weights = torch.stack(
                [item.support_distance_weights for item in penalties]
            ).mean(dim=0)
            support_forward = torch.stack(
                [item.support_forward for item in penalties]
            ).mean()
            support_backward = torch.stack(
                [item.support_backward for item in penalties]
            ).mean()
            long_range = torch.stack([item.long_range for item in penalties]).mean()
            phase_target_valid = torch.stack(
                [item.phase_valid.any(dim=-1) for item in matched_targets]
            )
            correlation_strength = _matched_directional_mean(
                torch.stack([item.correlation_strength for item in matched_targets]),
                phase_target_valid,
            )
            uncertainty = _matched_directional_mean(
                torch.stack([item.uncertainty for item in matched_targets]),
                phase_target_valid,
            )
            support_target_valid = torch.stack(
                [item.support_valid for item in matched_targets]
            )
            displacement_quantile = _matched_directional_mean(
                torch.stack([item.displacement_quantile for item in matched_targets]),
                support_target_valid,
                invalid_value=-1.0,
            )
            dilation_valid = support_target_valid.any(dim=-1)
            dilation_radius = _matched_directional_mean(
                torch.stack(
                    [item.dilation_radius.to(torch.float32) for item in matched_targets]
                ),
                dilation_valid,
                invalid_value=-1.0,
            )
            sample_count = torch.stack(
                [item.sample_count.to(torch.float32) for item in matched_targets]
            )
            reference_count = torch.stack(
                [item.valid_count.to(torch.float32) for item in matched_targets]
            )
            valid_ratio_by_phase = reference_count.sum(dim=(0, 1, 2)) / (
                sample_count.sum(dim=(0, 1, 2)).clamp_min(1.0)
            )
            matched_reference_count = _matched_directional_mean(
                reference_count,
                phase_target_valid,
            )
        else:
            loss = phase = support = minus = plus = zero
            distance_weights = probs.new_zeros(probs.shape[2] - 1)
            shape = (probs.shape[2] - 1, self.num_phases)
            phase_distance_weights = probs.new_zeros(shape)
            support_distance_weights = probs.new_zeros(shape)
            correlation_strength = probs.new_zeros(shape)
            uncertainty = probs.new_zeros(shape)
            displacement_quantile = probs.new_full((*shape, SUPPORT_DIRECTIONS), -1.0)
            dilation_radius = probs.new_full(shape, -1.0)
            valid_ratio_by_phase = probs.new_zeros(self.num_phases)
            support_forward = support_backward = long_range = zero
            matched_reference_count = probs.new_zeros(shape)
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
            phase_distance_weights=phase_distance_weights,
            support_distance_weights=support_distance_weights,
            correlation_strength=correlation_strength,
            uncertainty=uncertainty,
            displacement_quantile=displacement_quantile,
            dilation_radius=dilation_radius,
            valid_ratio_by_phase=valid_ratio_by_phase,
            support_forward=support_forward,
            support_backward=support_backward,
            long_range=long_range,
            matched_reference_count=matched_reference_count,
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
        ood_lower, ood_upper = self._ood_bounds(normalized)
        ood_weight = self._ood_weight_from_bounds(
            float(distances.min()),
            lower=ood_lower,
            upper=ood_upper,
        )
        if ood_weight == 0.0:
            return None, True
        # Keep the full morphology-nearest ordering.  A global top-k cut creates
        # an artificial long-range cutoff because random centers provide fewer
        # valid samples for either signed direction at large distances.  The
        # target summarizers instead take the nearest valid k references for
        # each individual direction/distance/phase feature.
        eligibility_limit = max(ood_lower, ood_upper)
        nearest = distances.argsort(stable=True)
        nearest = nearest[distances[nearest] <= eligibility_limit]
        selected = tuple(entries[int(index)] for index in nearest)
        phase_stack = torch.stack(
            [entry.values.to(torch.float32) for entry in selected]
        )
        independent_stack = torch.stack(
            [entry.independent.to(torch.float32) for entry in selected]
        )
        phase_valid_stack = torch.stack([entry.phase_valid for entry in selected])
        support_stack = torch.stack(
            [entry.support.to(torch.float32) for entry in selected]
        )
        support_valid_stack = torch.stack([entry.support_valid for entry in selected])
        displacement_stack = torch.stack(
            [entry.displacement_hist.to(torch.float32) for entry in selected]
        )
        displacement_valid_stack = torch.stack(
            [entry.displacement_valid for entry in selected]
        )
        phase = _summarize_phase_target(
            phase_stack,
            independent_stack,
            phase_valid_stack,
            quantile_low=self.quantile_low,
            quantile_high=self.quantile_high,
            uncertainty_floor=floor,
            max_samples=self.neighbors,
        )
        support = _summarize_support_target(
            support_stack,
            support_valid_stack,
            displacement_stack,
            displacement_valid_stack,
            quantile_low=self.quantile_low,
            quantile_high=self.quantile_high,
            uncertainty_floor=floor,
            max_samples=self.neighbors,
        )
        if not bool(phase.weights.sum() > 0.0) and not bool(
            support.weights.sum() > 0.0
        ):
            return None, False
        legacy_phase_weights = _collapse_phase_weights(
            _aggregate_direction_weights(phase.weights)
        )
        return (
            RelationTarget(
                lower=phase.lower.to(device=device),
                upper=phase.upper.to(device=device),
                weights=legacy_phase_weights.to(device=device),
                valid=phase.valid.any(dim=(0, 2, 3)).to(device=device),
                support_lower=support.lower.to(device=device),
                support_upper=support.upper.to(device=device),
                support_weights=support.weights.to(device=device),
                support_valid=support.valid.to(device=device),
                ood_weight=ood_weight,
                source=source,
                mean_relation=phase.mean_relation.to(device=device),
                independent_baseline=phase.independent.to(device=device),
                correlation_strength=phase.strength.to(device=device),
                uncertainty=phase.uncertainty.to(device=device),
                phase_distance_weights=phase.weights.to(device=device),
                phase_valid=phase.valid.to(device=device),
                displacement_quantile=support.displacement_quantile.to(device=device),
                dilation_radius=support.dilation_radius.to(device=device),
                sample_count=phase.sample_count.to(device=device),
                valid_count=phase.valid_count.to(device=device),
                support_strength=support.strength.to(device=device),
                support_uncertainty=support.uncertainty.to(device=device),
                support_sample_count=support.sample_count.to(device=device),
                support_valid_count=support.valid_count.to(device=device),
            ),
            False,
        )

    @staticmethod
    def _ood_weight(normalized: torch.Tensor, query_distance: float) -> float:
        lower, upper = RelationBank._ood_bounds(normalized)
        return RelationBank._ood_weight_from_bounds(
            query_distance,
            lower=lower,
            upper=upper,
        )

    @staticmethod
    def _ood_bounds(normalized: torch.Tensor) -> tuple[float, float]:
        if normalized.shape[0] < 2:
            return 0.0, 0.0
        pairwise = torch.cdist(normalized, normalized) / math.sqrt(normalized.shape[1])
        pairwise.fill_diagonal_(float("inf"))
        nearest = pairwise.min(dim=1).values
        lower = float(torch.quantile(nearest, 0.95))
        upper = float(torch.quantile(nearest, 0.99))
        return lower, upper

    @staticmethod
    def _ood_weight_from_bounds(
        query_distance: float,
        *,
        lower: float,
        upper: float,
    ) -> float:
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
