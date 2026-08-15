import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F

MINUS_DIRECTION = 0
PLUS_DIRECTION = 1
SUPPORT_FORWARD = 0
SUPPORT_BACKWARD = 1
SUPPORT_DIRECTIONS = 2
DEFAULT_MIN_SUPPORT_PIXELS = 8
DEFAULT_MIN_PHASE_FRACTION = 0.001
DEFAULT_MIN_CHANCE_GAP = 0.02
DISPLACEMENT_QUANTILE = 0.90


def _empty_float() -> torch.Tensor:
    return torch.empty(0, dtype=torch.float32)


def _empty_bool() -> torch.Tensor:
    return torch.empty(0, dtype=torch.bool)


def _empty_long() -> torch.Tensor:
    return torch.empty(0, dtype=torch.long)


def _zero_float() -> torch.Tensor:
    return torch.tensor(0.0, dtype=torch.float32)


@dataclass(frozen=True)
class RelationCurve:
    # Direction order is minus, plus. Distance slot zero means one voxel away.
    values: torch.Tensor
    valid: torch.Tensor
    support: torch.Tensor
    support_valid: torch.Tensor
    descriptor: torch.Tensor
    roi_pixels: int
    # ``values`` is already the excess relation (joint - independent). Keeping
    # the analytic baseline explicit avoids accidentally subtracting it twice.
    independent: torch.Tensor = field(default_factory=_empty_float)
    phase_valid: torch.Tensor = field(default_factory=_empty_bool)
    support_raw: torch.Tensor = field(default_factory=_empty_float)
    displacement_hist: torch.Tensor = field(default_factory=_empty_float)
    displacement_valid: torch.Tensor = field(default_factory=_empty_bool)


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
    mean_relation: torch.Tensor = field(default_factory=_empty_float)
    independent_baseline: torch.Tensor = field(default_factory=_empty_float)
    correlation_strength: torch.Tensor = field(default_factory=_empty_float)
    uncertainty: torch.Tensor = field(default_factory=_empty_float)
    phase_distance_weights: torch.Tensor = field(default_factory=_empty_float)
    phase_valid: torch.Tensor = field(default_factory=_empty_bool)
    displacement_quantile: torch.Tensor = field(default_factory=_empty_float)
    dilation_radius: torch.Tensor = field(default_factory=_empty_long)
    sample_count: torch.Tensor = field(default_factory=_empty_long)
    valid_count: torch.Tensor = field(default_factory=_empty_long)
    support_strength: torch.Tensor = field(default_factory=_empty_float)
    support_uncertainty: torch.Tensor = field(default_factory=_empty_float)
    support_sample_count: torch.Tensor = field(default_factory=_empty_long)
    support_valid_count: torch.Tensor = field(default_factory=_empty_long)


@dataclass(frozen=True)
class RelationPenalty:
    loss: torch.Tensor
    phase: torch.Tensor
    support: torch.Tensor
    minus: torch.Tensor
    plus: torch.Tensor
    distance_weights: torch.Tensor
    phase_distance_weights: torch.Tensor = field(default_factory=_empty_float)
    support_distance_weights: torch.Tensor = field(default_factory=_empty_float)
    support_forward: torch.Tensor = field(default_factory=_zero_float)
    support_backward: torch.Tensor = field(default_factory=_zero_float)
    long_range: torch.Tensor = field(default_factory=_zero_float)


@dataclass(frozen=True)
class _PhaseTargetStatistics:
    lower: torch.Tensor
    upper: torch.Tensor
    mean_relation: torch.Tensor
    independent: torch.Tensor
    strength: torch.Tensor
    uncertainty: torch.Tensor
    weights: torch.Tensor
    valid: torch.Tensor
    sample_count: torch.Tensor
    valid_count: torch.Tensor


@dataclass(frozen=True)
class _SupportTargetStatistics:
    lower: torch.Tensor
    upper: torch.Tensor
    strength: torch.Tensor
    uncertainty: torch.Tensor
    weights: torch.Tensor
    valid: torch.Tensor
    displacement_quantile: torch.Tensor
    dilation_radius: torch.Tensor
    sample_count: torch.Tensor
    valid_count: torch.Tensor


def _validate_max_samples(max_samples: int | None) -> None:
    if max_samples is None:
        return
    if (
        not isinstance(max_samples, int)
        or isinstance(max_samples, bool)
        or max_samples < 2
    ):
        raise ValueError("maximum relation samples must be at least two.")


def _nearest_valid_indices(
    valid: torch.Tensor,
    max_samples: int | None,
) -> torch.Tensor:
    if valid.ndim != 1 or valid.dtype != torch.bool:
        raise ValueError("relation sample validity must be a boolean vector.")
    indices = valid.nonzero().flatten()
    return indices if max_samples is None else indices[:max_samples]


def _summarize_phase_target(
    excess: torch.Tensor,
    independent: torch.Tensor,
    valid: torch.Tensor,
    *,
    quantile_low: float,
    quantile_high: float,
    uncertainty_floor: float,
    max_samples: int | None = None,
) -> _PhaseTargetStatistics:
    if excess.ndim != 5 or valid.shape != excess.shape:
        raise ValueError("phase samples must have shape [N,2,D,C,C].")
    if independent.shape != excess.shape:
        raise ValueError("phase sample baselines and validity must match.")
    sample_size, directions, distances, phases, relation_dims = excess.shape
    if directions != 2 or phases != relation_dims:
        raise ValueError("phase samples must contain two directions and square rows.")
    _validate_max_samples(max_samples)
    minimum_samples = max_samples or 2
    lower = excess.new_zeros(directions, distances, phases, relation_dims)
    upper = torch.zeros_like(lower)
    mean_relation = torch.zeros_like(lower)
    mean_independent = torch.zeros_like(lower)
    mean_excess = torch.zeros_like(lower)
    feature_valid = torch.zeros_like(lower, dtype=torch.bool)
    strength = excess.new_zeros(directions, distances, phases)
    uncertainty = torch.zeros_like(strength)
    sample_count = torch.full(
        (directions, distances, phases),
        min(sample_size, max_samples or sample_size),
        dtype=torch.long,
        device=excess.device,
    )
    valid_count = torch.zeros_like(sample_count)
    for direction in range(directions):
        for distance in range(distances):
            for phase_from in range(phases):
                row_valid = valid[:, direction, distance, phase_from]
                valid_count[direction, distance, phase_from] = len(
                    _nearest_valid_indices(row_valid.any(dim=-1), max_samples)
                )
                for phase_to in range(relation_dims):
                    selected = _nearest_valid_indices(
                        row_valid[:, phase_to],
                        max_samples,
                    )
                    if selected.numel() < minimum_samples:
                        continue
                    values = excess[
                        selected,
                        direction,
                        distance,
                        phase_from,
                        phase_to,
                    ]
                    baselines = independent[
                        selected,
                        direction,
                        distance,
                        phase_from,
                        phase_to,
                    ]
                    key = (direction, distance, phase_from, phase_to)
                    lower[key] = torch.quantile(values, quantile_low)
                    upper[key] = torch.quantile(values, quantile_high)
                    mean_excess[key] = values.mean()
                    mean_independent[key] = baselines.mean()
                    mean_relation[key] = mean_excess[key] + mean_independent[key]
                    feature_valid[key] = True
    feature_mask = feature_valid.to(excess.dtype)
    feature_count = feature_mask.sum(dim=-1).clamp_min(1.0)
    strength = (
        (mean_excess.square() * feature_mask).sum(dim=-1).div(feature_count).sqrt()
    )
    width = upper - lower
    uncertainty = (width.square() * feature_mask).sum(dim=-1).div(feature_count).sqrt()
    group_valid = feature_valid.any(dim=-1)
    weights = _data_driven_distance_weights(
        strength,
        uncertainty,
        group_valid,
        uncertainty_floor=uncertainty_floor,
    )
    return _PhaseTargetStatistics(
        lower=lower,
        upper=upper,
        mean_relation=mean_relation,
        independent=mean_independent,
        strength=strength,
        uncertainty=uncertainty,
        weights=weights,
        valid=feature_valid,
        sample_count=sample_count,
        valid_count=valid_count,
    )


def _summarize_support_target(
    support: torch.Tensor,
    support_valid: torch.Tensor,
    displacement_hist: torch.Tensor,
    displacement_valid: torch.Tensor,
    *,
    quantile_low: float,
    quantile_high: float,
    uncertainty_floor: float,
    max_samples: int | None = None,
) -> _SupportTargetStatistics:
    if support.ndim != 6 or support.shape != support_valid.shape:
        raise ValueError("support samples must have shape [N,2,D,R,C,2].")
    if displacement_hist.ndim != 6 or displacement_hist.shape[:4] != (
        *support.shape[:3],
        support.shape[4],
    ):
        raise ValueError("displacement histograms must match support samples.")
    if displacement_hist.shape[4] != SUPPORT_DIRECTIONS:
        raise ValueError("displacement histograms need forward and backward values.")
    if displacement_valid.shape != displacement_hist.shape[:-1]:
        raise ValueError("displacement validity must match its histograms.")
    samples, directions, distances, radii, phases, support_directions = support.shape
    if directions != 2 or support_directions != SUPPORT_DIRECTIONS:
        raise ValueError("support samples contain invalid direction dimensions.")
    if displacement_hist.shape[-1] != radii + 1:
        raise ValueError("displacement histograms need one overflow bin.")
    _validate_max_samples(max_samples)
    minimum_samples = max_samples or 2

    lower = support.new_zeros(
        directions,
        distances,
        phases,
        support_directions,
    )
    upper = torch.zeros_like(lower)
    mean = torch.zeros_like(lower)
    target_valid = torch.zeros_like(lower, dtype=torch.bool)
    displacement_quantile = support.new_full(
        (directions, distances, phases, support_directions),
        -1.0,
    )
    quantile_valid = torch.zeros_like(displacement_quantile, dtype=torch.bool)
    dilation_radius = torch.full(
        (directions, distances, phases),
        -1,
        dtype=torch.long,
        device=support.device,
    )
    sample_count = torch.full(
        (directions, distances, phases),
        min(samples, max_samples or samples),
        dtype=torch.long,
        device=support.device,
    )
    valid_count = torch.zeros_like(sample_count)

    for direction in range(directions):
        for distance in range(distances):
            for phase in range(phases):
                for support_direction in range(support_directions):
                    selected = _nearest_valid_indices(
                        displacement_valid[
                            :,
                            direction,
                            distance,
                            phase,
                            support_direction,
                        ],
                        max_samples,
                    )
                    if selected.numel() < minimum_samples:
                        continue
                    histogram = displacement_hist[
                        selected,
                        direction,
                        distance,
                        phase,
                        support_direction,
                    ].sum(dim=0)
                    histogram = histogram / histogram.sum().clamp_min(
                        torch.finfo(histogram.dtype).eps
                    )
                    reached = (
                        histogram.cumsum(dim=0) >= DISPLACEMENT_QUANTILE
                    ).nonzero()
                    if not reached.numel():
                        continue
                    radius = int(reached[0].item())
                    if radius >= radii:
                        continue
                    key = (direction, distance, phase, support_direction)
                    displacement_quantile[key] = float(radius)
                    quantile_valid[key] = True
                if not bool(quantile_valid[direction, distance, phase].all()):
                    continue
                radius = int(
                    displacement_quantile[direction, distance, phase].max().item()
                )
                dilation_radius[direction, distance, phase] = radius
                support_counts = []
                for support_direction in range(support_directions):
                    selected = _nearest_valid_indices(
                        support_valid[
                            :,
                            direction,
                            distance,
                            radius,
                            phase,
                            support_direction,
                        ],
                        max_samples,
                    )
                    values = support[
                        selected,
                        direction,
                        distance,
                        radius,
                        phase,
                        support_direction,
                    ]
                    support_counts.append(values.shape[0])
                    if values.shape[0] < minimum_samples:
                        continue
                    key = (direction, distance, phase, support_direction)
                    lower[key] = torch.quantile(values, quantile_low)
                    upper[key] = torch.quantile(values, quantile_high)
                    mean[key] = values.mean()
                    target_valid[key] = True
                valid_count[direction, distance, phase] = min(support_counts)

    feature_mask = target_valid.to(mean.dtype)
    feature_count = feature_mask.sum(dim=-1).clamp_min(1.0)
    strength = (mean.square() * feature_mask).sum(dim=-1).div(feature_count).sqrt()
    width = upper - lower
    uncertainty = (width.square() * feature_mask).sum(dim=-1).div(feature_count).sqrt()
    group_valid = target_valid.any(dim=-1)
    weights = _data_driven_distance_weights(
        strength,
        uncertainty,
        group_valid,
        uncertainty_floor=uncertainty_floor,
    )
    return _SupportTargetStatistics(
        lower=lower,
        upper=upper,
        strength=strength,
        uncertainty=uncertainty,
        weights=weights,
        valid=target_valid,
        displacement_quantile=displacement_quantile,
        dilation_radius=dilation_radius,
        sample_count=sample_count,
        valid_count=valid_count,
    )


def _data_driven_distance_weights(
    strength: torch.Tensor,
    uncertainty: torch.Tensor,
    valid: torch.Tensor,
    *,
    uncertainty_floor: float,
) -> torch.Tensor:
    if strength.ndim not in (2, 3) or strength.shape != uncertainty.shape:
        raise ValueError(
            "relation strength and uncertainty must have shape [D,C] or [2,D,C]."
        )
    if valid.shape != strength.shape or valid.dtype != torch.bool:
        raise ValueError("relation weight validity must match [D,C].")
    if not math.isfinite(uncertainty_floor) or uncertainty_floor <= 0.0:
        raise ValueError("relation uncertainty floor must be positive and finite.")
    regularized = (uncertainty.square() + float(uncertainty_floor) ** 2).sqrt()
    raw = torch.where(valid, strength / regularized, torch.zeros_like(strength))
    denominator = raw.sum(dim=-2, keepdim=True)
    return torch.where(
        denominator > 0.0,
        raw / denominator.clamp_min(torch.finfo(raw.dtype).eps),
        torch.zeros_like(raw),
    )


def _collapse_phase_weights(weights: torch.Tensor) -> torch.Tensor:
    if weights.ndim != 2:
        raise ValueError("phase distance weights must have shape [D,C].")
    active = weights > 0.0
    phase_count = active.sum(dim=1).clamp_min(1)
    collapsed = weights.sum(dim=1) / phase_count
    total = collapsed.sum()
    return collapsed / total if bool(total > 0.0) else torch.zeros_like(collapsed)


def _aggregate_direction_weights(weights: torch.Tensor) -> torch.Tensor:
    if weights.ndim == 2:
        return weights
    if weights.ndim != 3 or weights.shape[0] != 2:
        raise ValueError("directional weights must have shape [2,D,C].")
    active = weights.sum(dim=1) > 0.0
    return (weights * active.unsqueeze(1)).sum(dim=0) / active.sum(dim=0).clamp_min(
        1
    ).unsqueeze(0)


def _matched_directional_mean(
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    invalid_value: float = 0.0,
) -> torch.Tensor:
    if values.shape != valid.shape or values.ndim < 3 or values.shape[1] != 2:
        raise ValueError(
            "matched directional values and validity must have shape [N,2,...]."
        )
    mask = valid.to(values.dtype)
    count = mask.sum(dim=(0, 1))
    mean = (values * mask).sum(dim=(0, 1)) / count.clamp_min(1.0)
    return torch.where(
        count > 0.0,
        mean,
        torch.full_like(mean, float(invalid_value)),
    )


def relation_curve(
    probs: torch.Tensor,
    *,
    axis: int,
    index: int,
    roi: torch.Tensor,
    frozen_mask: torch.Tensor | None = None,
    support_max_radius: int = 0,
    min_support_pixels: int = DEFAULT_MIN_SUPPORT_PIXELS,
    min_phase_fraction: float = DEFAULT_MIN_PHASE_FRACTION,
    min_chance_gap: float = DEFAULT_MIN_CHANCE_GAP,
    collect_displacement: bool = True,
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
    _validate_support_guards(
        min_support_pixels,
        min_phase_fraction,
        min_chance_gap,
    )
    if not isinstance(collect_displacement, bool):
        raise TypeError("collect_displacement must be a boolean.")
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
    independent = torch.zeros_like(values)
    valid = torch.zeros(2, size - 1, device=probs.device, dtype=torch.bool)
    phase_valid = torch.zeros(
        2,
        size - 1,
        channels,
        channels,
        device=probs.device,
        dtype=torch.bool,
    )
    radius_count = support_max_radius + 1
    support = torch.zeros(
        2,
        size - 1,
        radius_count,
        channels,
        SUPPORT_DIRECTIONS,
        device=probs.device,
        dtype=torch.float32,
    )
    support_valid = torch.zeros_like(support, dtype=torch.bool)
    support_raw = torch.zeros_like(support)
    displacement_hist = torch.zeros(
        2,
        size - 1,
        channels,
        SUPPORT_DIRECTIONS,
        radius_count + 1 if collect_displacement else 0,
        device=probs.device,
        dtype=torch.float32,
    )
    displacement_valid = torch.zeros(
        2,
        size - 1,
        channels,
        SUPPORT_DIRECTIONS,
        device=probs.device,
        dtype=torch.bool,
    )
    mask = roi.to(torch.float32)
    area = mask.sum()
    center_masked = center * mask
    center_fraction = center_masked.sum(dim=(-2, -1)) / area
    minimum_mass = _minimum_phase_mass(
        area,
        min_support_pixels=min_support_pixels,
        min_phase_fraction=min_phase_fraction,
    )
    center_mass = center_masked.sum(dim=(-2, -1))
    center_valid = center_mass >= minimum_mass
    for direction, neighbor_indices in (
        (MINUS_DIRECTION, tuple(range(index - 1, -1, -1))),
        (PLUS_DIRECTION, tuple(range(index + 1, size))),
    ):
        if not neighbor_indices:
            continue
        neighbors = planes[:, neighbor_indices].movedim(1, 0)
        neighbors_masked = neighbors * mask
        joint = (
            torch.einsum(
                "chw,nkhw->nck",
                center_masked,
                neighbors_masked,
            )
            / area
        )
        neighbor_fraction = neighbors_masked.sum(dim=(-2, -1)) / area
        side_independent = torch.einsum(
            "c,nk->nck",
            center_fraction,
            neighbor_fraction,
        )
        corrected = joint - side_independent
        distance_indices = torch.arange(
            len(neighbor_indices),
            device=probs.device,
        )
        values[direction, distance_indices] = corrected
        independent[direction, distance_indices] = side_independent
        valid[direction, distance_indices] = True
        neighbor_mass = neighbors_masked.sum(dim=(-2, -1))
        phase_valid[direction, distance_indices] = center_valid.view(1, channels, 1) & (
            neighbor_mass >= minimum_mass
        ).unsqueeze(1)
        side = _support_statistics(
            center,
            neighbors,
            roi,
            max_radius=support_max_radius,
            min_support_pixels=min_support_pixels,
            min_phase_fraction=min_phase_fraction,
            min_chance_gap=min_chance_gap,
            collect_displacement=collect_displacement,
        )
        support[direction, distance_indices] = side.values
        support_valid[direction, distance_indices] = side.valid
        support_raw[direction, distance_indices] = side.raw
        if collect_displacement:
            displacement_hist[direction, distance_indices] = side.displacement_hist
            displacement_valid[direction, distance_indices] = side.displacement_valid
    return RelationCurve(
        values=values,
        valid=valid,
        support=support,
        support_valid=support_valid,
        descriptor=matching_descriptor(center, roi),
        roi_pixels=int(area.item()),
        independent=independent,
        phase_valid=phase_valid,
        support_raw=support_raw,
        displacement_hist=displacement_hist,
        displacement_valid=displacement_valid,
    )


def support_relation(
    center: torch.Tensor,
    neighbors: torch.Tensor,
    roi: torch.Tensor,
    *,
    max_radius: int,
    min_support_pixels: int = DEFAULT_MIN_SUPPORT_PIXELS,
    min_phase_fraction: float = DEFAULT_MIN_PHASE_FRACTION,
    min_chance_gap: float = DEFAULT_MIN_CHANCE_GAP,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = _support_statistics(
        center,
        neighbors,
        roi,
        max_radius=max_radius,
        min_support_pixels=min_support_pixels,
        min_phase_fraction=min_phase_fraction,
        min_chance_gap=min_chance_gap,
        collect_displacement=False,
    )
    return result.values, result.valid


@dataclass(frozen=True)
class _SupportStatistics:
    values: torch.Tensor
    valid: torch.Tensor
    raw: torch.Tensor
    baseline: torch.Tensor
    displacement_hist: torch.Tensor
    displacement_valid: torch.Tensor


def _support_statistics(
    center: torch.Tensor,
    neighbors: torch.Tensor,
    roi: torch.Tensor,
    *,
    max_radius: int,
    min_support_pixels: int,
    min_phase_fraction: float,
    min_chance_gap: float,
    collect_displacement: bool = True,
) -> _SupportStatistics:
    if center.ndim != 3 or neighbors.ndim != 4:
        raise ValueError("support inputs must be [C,H,W] and [N,C,H,W].")
    if neighbors.shape[1:] != center.shape:
        raise ValueError("support center and neighbor planes must match.")
    if roi.shape != center.shape[1:] or roi.dtype != torch.bool:
        raise ValueError("support ROI must match the plane shape.")
    if (
        not isinstance(max_radius, int)
        or isinstance(max_radius, bool)
        or max_radius < 0
    ):
        raise ValueError("support max radius must be a non-negative integer.")
    _validate_support_guards(
        min_support_pixels,
        min_phase_fraction,
        min_chance_gap,
    )
    if not isinstance(collect_displacement, bool):
        raise TypeError("collect_displacement must be a boolean.")
    radius_count = max_radius + 1
    values = center.new_zeros(
        neighbors.shape[0],
        radius_count,
        center.shape[0],
        SUPPORT_DIRECTIONS,
        dtype=torch.float32,
    )
    valid = torch.zeros_like(values, dtype=torch.bool)
    raw = torch.zeros_like(values)
    baseline = torch.zeros_like(values)
    roi_float = roi.to(torch.float32)
    masked_center = center.detach() * roi_float
    masked_neighbors = neighbors * roi_float
    epsilon = torch.finfo(torch.float32).eps
    for radius in range(radius_count):
        valid_roi = _erode_roi(roi, radius)
        if not bool(valid_roi.any()):
            continue
        weights = valid_roi.to(torch.float32)
        area = weights.sum()
        minimum_mass = _minimum_phase_mass(
            area,
            min_support_pixels=min_support_pixels,
            min_phase_fraction=min_phase_fraction,
        )
        dilated_center = _dilate(masked_center, radius)
        dilated_neighbors = _dilate(masked_neighbors, radius)

        neighbor_mass = (neighbors * weights).sum(dim=(-2, -1))
        center_mass = (center.detach() * weights).sum(dim=(-2, -1))
        pair_valid = (neighbor_mass >= minimum_mass) & (
            center_mass.unsqueeze(0) >= minimum_mass
        )
        center_baseline = (dilated_center * weights).sum(dim=(-2, -1)) / area
        forward_raw = (neighbors * dilated_center.unsqueeze(0) * weights).sum(
            dim=(-2, -1)
        ) / neighbor_mass.clamp_min(epsilon)
        forward_scale = (1.0 - center_baseline).clamp_min(min_chance_gap)
        forward = (forward_raw - center_baseline) / forward_scale.clamp_min(epsilon)
        forward_valid = pair_valid & (
            center_baseline.unsqueeze(0) <= 1.0 - min_chance_gap
        )

        neighbor_baseline = (dilated_neighbors * weights).sum(dim=(-2, -1)) / area
        backward_raw = (center.detach().unsqueeze(0) * dilated_neighbors * weights).sum(
            dim=(-2, -1)
        ) / center_mass.clamp_min(epsilon).unsqueeze(0)
        backward_scale = (1.0 - neighbor_baseline).clamp_min(min_chance_gap)
        backward = (backward_raw - neighbor_baseline) / backward_scale.clamp_min(
            epsilon
        )
        backward_valid = pair_valid & (neighbor_baseline <= 1.0 - min_chance_gap)

        values[:, radius, :, SUPPORT_FORWARD] = forward
        values[:, radius, :, SUPPORT_BACKWARD] = backward
        valid[:, radius, :, SUPPORT_FORWARD] = forward_valid
        valid[:, radius, :, SUPPORT_BACKWARD] = backward_valid
        raw[:, radius, :, SUPPORT_FORWARD] = forward_raw
        raw[:, radius, :, SUPPORT_BACKWARD] = backward_raw
        baseline[:, radius, :, SUPPORT_FORWARD] = center_baseline.unsqueeze(0)
        baseline[:, radius, :, SUPPORT_BACKWARD] = neighbor_baseline

    if collect_displacement:
        displacement_hist, displacement_valid = _displacement_histogram(
            center,
            neighbors,
            roi,
            max_radius=max_radius,
            min_support_pixels=min_support_pixels,
            min_phase_fraction=min_phase_fraction,
        )
    else:
        displacement_hist = center.new_empty(
            neighbors.shape[0],
            center.shape[0],
            SUPPORT_DIRECTIONS,
            0,
            dtype=torch.float32,
        )
        displacement_valid = torch.zeros(
            neighbors.shape[0],
            center.shape[0],
            SUPPORT_DIRECTIONS,
            device=center.device,
            dtype=torch.bool,
        )
    return _SupportStatistics(
        values=values,
        valid=valid,
        raw=raw,
        baseline=baseline,
        displacement_hist=displacement_hist,
        displacement_valid=displacement_valid,
    )


def _displacement_histogram(
    center: torch.Tensor,
    neighbors: torch.Tensor,
    roi: torch.Tensor,
    *,
    max_radius: int,
    min_support_pixels: int,
    min_phase_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    common_roi = _erode_roi(roi, max_radius)
    radius_count = max_radius + 1
    hist = center.new_zeros(
        neighbors.shape[0],
        center.shape[0],
        SUPPORT_DIRECTIONS,
        radius_count + 1,
        dtype=torch.float32,
    )
    valid = torch.zeros(
        neighbors.shape[0],
        center.shape[0],
        SUPPORT_DIRECTIONS,
        device=center.device,
        dtype=torch.bool,
    )
    if not bool(common_roi.any()):
        return hist, valid

    weights = common_roi.to(torch.float32)
    area = weights.sum()
    minimum_mass = _minimum_phase_mass(
        area,
        min_support_pixels=min_support_pixels,
        min_phase_fraction=min_phase_fraction,
    )
    center_labels = center.detach().argmax(dim=0)
    hard_center = (
        F.one_hot(
            center_labels,
            num_classes=center.shape[0],
        )
        .movedim(-1, 0)
        .to(torch.float32)
    )
    neighbor_labels = neighbors.detach().argmax(dim=1)
    hard_neighbors = (
        F.one_hot(
            neighbor_labels,
            num_classes=center.shape[0],
        )
        .movedim(-1, 1)
        .to(torch.float32)
    )
    hard_center = hard_center * weights
    hard_neighbors = hard_neighbors * weights
    center_mass = (hard_center * weights).sum(dim=(-2, -1))
    neighbor_mass = (hard_neighbors * weights).sum(dim=(-2, -1))
    pair_valid = (center_mass.unsqueeze(0) >= minimum_mass) & (
        neighbor_mass >= minimum_mass
    )
    epsilon = torch.finfo(torch.float32).eps
    cdf = hist.new_zeros(
        neighbors.shape[0],
        radius_count,
        center.shape[0],
        SUPPORT_DIRECTIONS,
    )
    for radius in range(radius_count):
        dilated_center = _dilate(hard_center, radius)
        dilated_neighbors = _dilate(hard_neighbors, radius)
        cdf[:, radius, :, SUPPORT_FORWARD] = (
            hard_neighbors * dilated_center.unsqueeze(0) * weights
        ).sum(dim=(-2, -1)) / neighbor_mass.clamp_min(epsilon)
        cdf[:, radius, :, SUPPORT_BACKWARD] = (
            hard_center.unsqueeze(0) * dilated_neighbors * weights
        ).sum(dim=(-2, -1)) / center_mass.clamp_min(epsilon).unsqueeze(0)
    cdf = cdf.clamp(0.0, 1.0).cummax(dim=1).values
    cdf = cdf.permute(0, 2, 3, 1)
    hist[..., 0] = cdf[..., 0]
    if radius_count > 1:
        hist[..., 1:radius_count] = (cdf[..., 1:] - cdf[..., :-1]).clamp_min(0.0)
    hist[..., radius_count] = (1.0 - cdf[..., -1]).clamp_min(0.0)
    valid.copy_(pair_valid.unsqueeze(-1).expand_as(valid))
    hist *= valid.unsqueeze(-1)
    return hist, valid


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
            change = 0.5 * (values[tuple(left)] - values[tuple(right)]).abs().sum(dim=0)
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
    directional_phase_weights = _target_phase_weights(target, curve.values.shape[2])
    phase_values, phase_active = _directional_phase_loss(
        curve,
        target,
        directional_phase_weights,
        floor,
    )
    (
        support_values,
        support_active,
        support_direction_values,
        support_direction_active,
        support_distance_weights,
    ) = _directional_support_loss(
        curve,
        target,
        floor,
        zero,
    )
    phase_distance_weights = _aggregate_direction_weights(directional_phase_weights)
    phase = _reduce_directions(phase_values, phase_active, direction_reduction, zero)
    support = _reduce_directions(
        support_values,
        support_active,
        direction_reduction,
        zero,
    )
    support_forward = _reduce_directions(
        support_direction_values[:, SUPPORT_FORWARD],
        support_direction_active[:, SUPPORT_FORWARD],
        direction_reduction,
        zero,
    )
    support_backward = _reduce_directions(
        support_direction_values[:, SUPPORT_BACKWARD],
        support_direction_active[:, SUPPORT_BACKWARD],
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
    phase_distance = _collapse_phase_weights(phase_distance_weights)
    support_distance = _collapse_phase_weights(support_distance_weights)
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
        phase_distance_weights=phase_distance_weights,
        support_distance_weights=support_distance_weights,
        support_forward=ood * support_forward,
        support_backward=ood * support_backward,
        long_range=ood * phase,
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


def _target_phase_weights(target: RelationTarget, phases: int) -> torch.Tensor:
    distances = target.weights.shape[0]
    if target.phase_distance_weights.numel():
        if target.phase_distance_weights.shape == (distances, phases):
            return target.phase_distance_weights.unsqueeze(0).expand(2, -1, -1)
        if target.phase_distance_weights.shape == (2, distances, phases):
            return target.phase_distance_weights
        raise ValueError("phase distance weights must have shape [D,C] or [2,D,C].")
    if target.weights.shape != (distances,):
        raise ValueError("legacy phase relation weights must match distances.")
    return target.weights.view(1, distances, 1).expand(2, -1, phases)


def _directional_phase_loss(
    curve: RelationCurve,
    target: RelationTarget,
    weights: torch.Tensor,
    floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    directions, distances, phases, relation_dims = curve.values.shape
    if target.lower.shape == (distances, phases, relation_dims):
        lower = target.lower.unsqueeze(0).expand(directions, -1, -1, -1)
        upper = target.upper.unsqueeze(0).expand_as(lower)
    elif target.lower.shape == curve.values.shape:
        lower = target.lower
        upper = target.upper
    else:
        raise ValueError("phase targets must have shape [D,C,C] or [2,D,C,C].")

    if curve.phase_valid.shape == curve.values.shape:
        curve_valid = curve.phase_valid
    elif curve.phase_valid.shape == curve.values.shape[:3]:
        curve_valid = curve.phase_valid.unsqueeze(-1).expand_as(curve.values)
    else:
        curve_valid = curve.valid.view(directions, distances, 1, 1).expand_as(
            curve.values
        )

    if target.phase_valid.shape == lower.shape:
        target_valid = target.phase_valid
    elif target.phase_valid.shape == (distances, phases, relation_dims):
        target_valid = target.phase_valid.unsqueeze(0).expand_as(lower)
    elif target.phase_valid.shape == (directions, distances, phases):
        target_valid = target.phase_valid.unsqueeze(-1).expand_as(lower)
    elif target.phase_valid.shape == (distances, phases):
        target_valid = target.phase_valid.view(1, distances, phases, 1).expand_as(lower)
    else:
        target_valid = target.valid.view(1, distances, 1, 1).expand_as(lower)
    feature_valid = curve_valid & target_valid

    interval_width = upper - lower
    if target.uncertainty.shape == (directions, distances, phases):
        uncertainty = target.uncertainty
    elif target.uncertainty.shape == (distances, phases):
        uncertainty = target.uncertainty.unsqueeze(0).expand(directions, -1, -1)
    else:
        mask = target_valid.to(interval_width.dtype)
        uncertainty = (
            (interval_width.square() * mask)
            .sum(dim=-1)
            .div(mask.sum(dim=-1).clamp_min(1.0))
            .sqrt()
        )
    scale = (uncertainty.square() + floor**2).sqrt()
    violation = F.relu(lower - curve.values) + F.relu(curve.values - upper)
    violation = violation / scale.unsqueeze(-1)
    valid_float = feature_valid.to(violation.dtype)
    per_phase_distance = (violation * valid_float).sum(dim=-1) / valid_float.sum(
        dim=-1
    ).clamp_min(1.0)
    return _weighted_phase_distance(
        per_phase_distance,
        feature_valid.any(dim=-1),
        weights,
    )


def _directional_support_loss(
    curve: RelationCurve,
    target: RelationTarget,
    floor: float,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    directions, distances, _, phases, support_directions = curve.support.shape
    empty_direction_values = zero.expand(directions, support_directions)
    empty_direction_active = torch.zeros(
        directions,
        support_directions,
        device=curve.support.device,
        dtype=torch.bool,
    )
    if not target.support_lower.numel():
        return (
            zero.expand(directions),
            torch.zeros(directions, device=curve.support.device, dtype=torch.bool),
            empty_direction_values,
            empty_direction_active,
            curve.support.new_zeros(distances, phases),
        )
    if target.support_lower.ndim == 4 and not target.dilation_radius.numel():
        values, active = _directional_feature_loss(
            curve.support,
            curve.support_valid,
            target.support_lower,
            target.support_upper,
            target.support_valid,
            target.support_weights,
            floor,
            group_feature_dims=(-2, -1),
        )
        legacy = target.support_weights.sum(dim=1, keepdim=True).expand(-1, phases)
        return values, active, empty_direction_values, empty_direction_active, legacy
    selected_shape = (distances, phases, support_directions)
    directional_shape = (directions, *selected_shape)
    if target.support_lower.shape == selected_shape:
        lower = target.support_lower.unsqueeze(0).expand(directions, -1, -1, -1)
        upper = target.support_upper.unsqueeze(0).expand_as(lower)
        target_valid = target.support_valid.unsqueeze(0).expand_as(lower)
    elif target.support_lower.shape == directional_shape:
        lower = target.support_lower
        upper = target.support_upper
        target_valid = target.support_valid
    else:
        raise ValueError(
            "selected support targets must have shape [D,C,2] or [2,D,C,2]."
        )
    if target.support_weights.shape == (distances, phases):
        weights = target.support_weights.unsqueeze(0).expand(directions, -1, -1)
    elif target.support_weights.shape == (directions, distances, phases):
        weights = target.support_weights
    else:
        raise ValueError("support weights must have shape [D,C] or [2,D,C].")
    if target.dilation_radius.shape == (distances, phases):
        target_radii = target.dilation_radius.unsqueeze(0).expand(
            directions,
            -1,
            -1,
        )
    elif target.dilation_radius.shape == (directions, distances, phases):
        target_radii = target.dilation_radius
    else:
        raise ValueError("learned dilation radii must have shape [D,C] or [2,D,C].")

    radii = target_radii.clamp(0, curve.support.shape[2] - 1)
    gather_index = radii.view(directions, distances, 1, phases, 1).expand(
        directions,
        distances,
        1,
        phases,
        support_directions,
    )
    selected = curve.support.gather(2, gather_index).squeeze(2)
    selected_valid = curve.support_valid.gather(2, gather_index).squeeze(2)
    selected_valid &= target_valid
    selected_valid &= (target_radii >= 0).unsqueeze(-1)
    width = upper - lower
    if target.support_uncertainty.shape == (directions, distances, phases):
        uncertainty = target.support_uncertainty
    elif target.support_uncertainty.shape == (distances, phases):
        uncertainty = target.support_uncertainty.unsqueeze(0).expand(
            directions,
            -1,
            -1,
        )
    else:
        mask = target_valid.to(width.dtype)
        uncertainty = (
            (width.square() * mask)
            .sum(dim=-1)
            .div(mask.sum(dim=-1).clamp_min(1.0))
            .sqrt()
        )
    scale = (uncertainty.square() + floor**2).sqrt()
    violation = F.relu(lower - selected) + F.relu(selected - upper)
    violation = violation / scale.unsqueeze(-1)
    support_values = []
    support_active = []
    for support_direction in range(support_directions):
        values, active = _weighted_phase_distance(
            violation[..., support_direction],
            selected_valid[..., support_direction],
            weights,
        )
        support_values.append(values)
        support_active.append(active)
    per_support_direction = torch.stack(support_values, dim=1)
    per_support_active = torch.stack(support_active, dim=1)
    combined = []
    combined_active = []
    for direction in range(directions):
        active = per_support_active[direction]
        combined.append(
            per_support_direction[direction][active].mean()
            if bool(active.any())
            else zero
        )
        combined_active.append(bool(active.any()))
    return (
        torch.stack(combined),
        torch.tensor(combined_active, device=curve.support.device, dtype=torch.bool),
        per_support_direction,
        per_support_active,
        _aggregate_direction_weights(weights),
    )


def _weighted_phase_distance(
    values: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 3 or valid.shape != values.shape:
        raise ValueError("directional phase values must have shape [2,D,C].")
    if weights.shape == values.shape[1:]:
        directional_weights = weights.unsqueeze(0).expand(values.shape[0], -1, -1)
    elif weights.shape == values.shape:
        directional_weights = weights
    else:
        raise ValueError("phase distance weights must match [D,C] or [2,D,C].")
    effective = directional_weights * valid
    effective = effective / effective.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(effective.dtype).eps
    )
    per_phase = (effective * values).sum(dim=1)
    phase_active = valid.any(dim=1) & (directional_weights.sum(dim=1) > 0.0)
    count = phase_active.sum(dim=1)
    directional = (per_phase * phase_active).sum(dim=1) / count.clamp_min(1)
    return directional, count > 0


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


def _dilate(values: torch.Tensor, radius: int) -> torch.Tensor:
    if radius == 0:
        return values
    return F.max_pool2d(
        values,
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )


def _minimum_phase_mass(
    area: torch.Tensor,
    *,
    min_support_pixels: int,
    min_phase_fraction: float,
) -> torch.Tensor:
    return torch.maximum(
        area * float(min_phase_fraction),
        area.new_tensor(float(min_support_pixels)),
    )


def _validate_support_guards(
    min_support_pixels: int,
    min_phase_fraction: float,
    min_chance_gap: float,
) -> None:
    if (
        not isinstance(min_support_pixels, int)
        or isinstance(min_support_pixels, bool)
        or min_support_pixels < 1
    ):
        raise ValueError("min_support_pixels must be a positive integer.")
    for name, value in (
        ("min_phase_fraction", min_phase_fraction),
        ("min_chance_gap", min_chance_gap),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0.0 < value < 1.0
        ):
            raise ValueError(
                f"{name} must be finite and strictly between zero and one."
            )


def _erode_roi(roi: torch.Tensor, radius: int) -> torch.Tensor:
    if radius == 0:
        return roi
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
    if curve.independent.numel() and curve.independent.shape != curve.values.shape:
        raise ValueError("independent relation baseline must match phase values.")
    if curve.phase_valid.numel() and curve.phase_valid.shape not in (
        curve.values.shape[:3],
        curve.values.shape,
    ):
        raise ValueError("phase validity must match relation rows or features.")
    if target.weights.shape != curve.valid.shape[1:]:
        raise ValueError("phase relation weights must match distances.")
    if target.valid.shape != curve.valid.shape[1:]:
        raise ValueError("phase target validity must match distances.")
    phase_weight_shapes = (
        (curve.values.shape[1], curve.values.shape[2]),
        (2, curve.values.shape[1], curve.values.shape[2]),
    )
    if (
        target.phase_distance_weights.numel()
        and target.phase_distance_weights.shape not in phase_weight_shapes
    ):
        raise ValueError("phase distance weights must be [D,C] or [2,D,C].")
    if target.support_lower.ndim == 4 and not target.dilation_radius.numel():
        if target.support_weights.shape != curve.support.shape[1:3]:
            raise ValueError("legacy support weights must match distance and radius.")
        if target.support_valid.shape != curve.support.shape[1:]:
            raise ValueError("legacy support target validity must match its features.")
    elif target.support_lower.ndim in (3, 4):
        expected = (
            curve.support.shape[1],
            curve.support.shape[3],
            curve.support.shape[4],
        )
        directional_expected = (2, *expected)
        if target.support_lower.shape not in (expected, directional_expected):
            raise ValueError("selected support intervals must be [D,C,2] or [2,D,C,2].")
        if target.support_upper.shape != target.support_lower.shape:
            raise ValueError("selected support interval bounds must match.")
        if target.support_valid.shape != target.support_lower.shape:
            raise ValueError("selected support validity must match its intervals.")
        if target.support_weights.shape not in (expected[:2], (2, *expected[:2])):
            raise ValueError("selected support weights must be [D,C] or [2,D,C].")
    else:
        raise ValueError("support targets must be selected or radius profiles.")
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
