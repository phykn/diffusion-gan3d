import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


def _spatial_tuple(name: str, value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        values = (value, value, value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(value)
    else:
        raise TypeError(f"{name} must be an integer or a sequence of three integers.")
    if len(values) != 3 or any(
        not isinstance(item, int) or isinstance(item, bool) for item in values
    ):
        raise ValueError(f"{name} must contain exactly three integers.")
    return values


def periodic_crop_3d(
    values: torch.Tensor,
    start: Sequence[int],
    size: int | Sequence[int],
) -> torch.Tensor:
    """Crop the last three dimensions, wrapping every spatial index periodically."""
    if not isinstance(values, torch.Tensor):
        raise TypeError("values must be a torch.Tensor.")
    if values.ndim < 3:
        raise ValueError("values must have at least three spatial dimensions.")
    if any(length < 1 for length in values.shape[-3:]):
        raise ValueError("values must have non-empty spatial dimensions.")
    starts = _spatial_tuple("start", start)
    sizes = _spatial_tuple("size", size)
    if any(length < 1 for length in sizes):
        raise ValueError("size values must be positive.")

    cropped = values
    for dim, (offset, length) in enumerate(zip(starts, sizes, strict=True), -3):
        indices = torch.arange(length, device=values.device)
        indices = indices.add(offset).remainder(values.shape[dim])
        cropped = cropped.index_select(dim, indices)
    return cropped


@dataclass(frozen=True)
class AdjacentViewPlan:
    """Two tiled inputs whose cores are adjacent along one spatial axis."""

    axis: int
    core_size: int
    overlap: int
    tile_size: int
    first_start: tuple[int, int, int]
    second_start: tuple[int, int, int]

    @property
    def first_band(self) -> tuple[slice, slice, slice]:
        band = [slice(None), slice(None), slice(None)]
        band[self.axis] = slice(self.core_size, self.core_size + 2 * self.overlap)
        return tuple(band)  # type: ignore[return-value]

    @property
    def second_band(self) -> tuple[slice, slice, slice]:
        band = [slice(None), slice(None), slice(None)]
        band[self.axis] = slice(0, 2 * self.overlap)
        return tuple(band)  # type: ignore[return-value]

    def crop_views(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        size = (self.tile_size,) * 3
        return (
            periodic_crop_3d(values, self.first_start, size),
            periodic_crop_3d(values, self.second_start, size),
        )

    def overlap_bands(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.tile_size,) * 3
        for name, values in (("first", first), ("second", second)):
            if not isinstance(values, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if values.ndim < 3 or values.shape[-3:] != expected:
                raise ValueError(f"{name} must end with spatial shape {expected}.")
        if first.shape[:-3] != second.shape[:-3]:
            raise ValueError("first and second must have matching leading dimensions.")
        return (
            first[(..., *self.first_band)],
            second[(..., *self.second_band)],
        )


def make_adjacent_view_plan(
    origin: Sequence[int],
    axis: int,
    core_size: int,
    overlap: int,
) -> AdjacentViewPlan:
    """Plan two periodic tile inputs around adjacent cores starting at ``origin``."""
    origins = _spatial_tuple("origin", origin)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    if not isinstance(core_size, int) or isinstance(core_size, bool) or core_size < 1:
        raise ValueError("core_size must be a positive integer.")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 2:
        raise ValueError("overlap must be an integer of at least two.")

    first_start = tuple(position - overlap for position in origins)
    second_start = tuple(
        position + (core_size if dim == axis else 0)
        for dim, position in enumerate(first_start)
    )
    return AdjacentViewPlan(
        axis=axis,
        core_size=core_size,
        overlap=overlap,
        tile_size=core_size + 2 * overlap,
        first_start=first_start,
        second_start=second_start,
    )


def cosine_overlap_weights(
    overlap: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the sin-squared fusion importance over a ``2 * overlap`` band."""
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 2:
        raise ValueError("overlap must be an integer of at least two.")
    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating point.")
    positions = torch.arange(overlap, device=device, dtype=dtype)
    ramp = positions.mul(math.pi / (2 * overlap)).sin().square()
    return torch.cat((ramp, ramp.flip(0)))


def weighted_probability_mse(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    axis: int,
    overlap: int,
    known_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compare aligned probability bands while excluding known anchor voxels."""
    compute_dtype = _validate_probability_pair(student, teacher, axis, overlap)
    student_values = student.to(dtype=compute_dtype)
    teacher_values = teacher.detach().to(dtype=compute_dtype)
    error = (student_values - teacher_values).square().mean(dim=1, keepdim=True)
    effective = _overlap_weights(student, axis, overlap, compute_dtype)

    known = _validate_known_mask(known_mask, student)
    if known is not None:
        effective = effective * (~known).to(dtype=compute_dtype)

    numerator = (error * effective).sum()
    denominator = effective.expand_as(error).sum()
    return numerator / denominator.clamp_min(torch.finfo(compute_dtype).eps)


def _validate_probability_pair(
    student: torch.Tensor,
    teacher: torch.Tensor,
    axis: int,
    overlap: int,
) -> torch.dtype:
    if not isinstance(student, torch.Tensor) or not isinstance(teacher, torch.Tensor):
        raise TypeError("student and teacher must be torch.Tensor instances.")
    if student.ndim != 5 or teacher.ndim != 5:
        raise ValueError("student and teacher must have shape [B, C, D, H, W].")
    if student.shape != teacher.shape:
        raise ValueError("student and teacher must have the same shape.")
    if student.device != teacher.device:
        raise ValueError("student and teacher must use the same device.")
    if not student.is_floating_point() or not teacher.is_floating_point():
        raise TypeError("student and teacher must use floating-point dtypes.")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 2:
        raise ValueError("overlap must be an integer of at least two.")
    if student.shape[axis + 2] != 2 * overlap:
        raise ValueError("the selected spatial axis must have length 2 * overlap.")

    compute_dtype = torch.promote_types(student.dtype, teacher.dtype)
    if compute_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    return compute_dtype


def _overlap_weights(
    values: torch.Tensor,
    axis: int,
    overlap: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    weights = cosine_overlap_weights(
        overlap,
        device=values.device,
        dtype=dtype,
    )
    weight_shape = [1, 1, 1, 1, 1]
    weight_shape[axis + 2] = 2 * overlap
    return weights.view(weight_shape)


def _validate_known_mask(
    known_mask: torch.Tensor | None,
    probabilities: torch.Tensor,
) -> torch.Tensor | None:
    if known_mask is None:
        return None
    if not isinstance(known_mask, torch.Tensor):
        raise TypeError("known_mask must be a torch.Tensor or None.")
    expected_spatial = probabilities.shape[2:]
    if (
        known_mask.ndim != 5
        or known_mask.shape[0] != probabilities.shape[0]
        or known_mask.shape[1] not in (1, probabilities.shape[1])
        or known_mask.shape[2:] != expected_spatial
    ):
        raise ValueError(
            "known_mask must match [B, 1, D, H, W] or the probability shape."
        )
    if known_mask.device != probabilities.device:
        raise ValueError("known_mask and probabilities must use the same device.")
    if known_mask.dtype != torch.bool:
        raise TypeError("known_mask must use torch.bool.")
    return known_mask.any(dim=1, keepdim=True)
