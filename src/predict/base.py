import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import torch

from src.prepare.resize import phase_channels


def volume_shape(volume):
    if volume is None:
        return None
    if not isinstance(volume, torch.Tensor):
        raise TypeError("base must be a torch.Tensor.")
    if volume.ndim not in (3, 4) or any(n < 1 for n in volume.shape):
        raise ValueError("base shape must be D,H,W labels or C,D,H,W fractions.")
    return tuple(volume.shape[-3:])


def resolve_offset(offset, shape, base_shape):
    maximum = tuple(n - b for n, b in zip(shape, base_shape, strict=True))
    if any(n < 0 for n in maximum):
        raise ValueError("base shape must fit inside the output shape.")
    if offset is None:
        values = (None,) * 3
    elif isinstance(offset, Sequence) and not isinstance(offset, (str, bytes)):
        values = tuple(offset)
    else:
        raise TypeError("base_offset must be a sequence of three values.")
    if len(values) != 3:
        raise ValueError("base_offset must contain exactly three values.")
    result = []
    for axis, (value, limit) in enumerate(zip(values, maximum, strict=True)):
        if value is None:
            value = limit // 2
        elif type(value) is not int:
            raise TypeError("base_offset values must be integers or None.")
        if not 0 <= value <= limit:
            raise ValueError(f"base_offset axis {axis} must be between 0 and {limit}.")
        result.append(value)
    return tuple(result), tuple(value is not None for value in values)


def phase_fractions(volume, num_phases):
    volume_shape(volume)
    if volume.ndim == 3:
        return phase_channels(volume.cpu().unsqueeze(0), num_phases)
    values = volume.to(device="cpu", dtype=torch.float32)
    if (
        not volume.dtype.is_floating_point
        or values.shape[0] != num_phases
        or not torch.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not torch.allclose(values.sum(0), torch.ones_like(values[0]), atol=1e-5)
    ):
        raise ValueError("base fractions must be finite, non-negative and sum to one.")
    return values.unsqueeze(0).clone()


@dataclass(frozen=True)
class Base:
    clean: torch.Tensor
    noise: torch.Tensor
    region: tuple[slice, slice, slice]
    weight: torch.Tensor
    preserve: bool = False

    def intersection(self, region):
        common = tuple(
            slice(max(a.start, b.start), min(a.stop, b.stop))
            for a, b in zip(region, self.region, strict=True)
        )
        if any(p.start >= p.stop for p in common):
            return None
        source = tuple(
            slice(p.start - b.start, p.stop - b.start)
            for p, b in zip(common, self.region, strict=True)
        )
        target = tuple(
            slice(p.start - r.start, p.stop - r.start)
            for p, r in zip(common, region, strict=True)
        )
        return source, target

    def prediction(self, region, device):
        if not self.preserve or any(
            r.start < b.start or r.stop > b.stop
            for r, b in zip(region, self.region, strict=True)
        ):
            return None
        source, _ = self.intersection(region)
        return self.clean[(slice(None), slice(None), *source)].to(device)

    def constrain(self, values, region, diffusion=None, time=0):
        if not self.preserve or (parts := self.intersection(region)) is None:
            return
        source, target = parts
        index = (slice(None), slice(None), *source)
        known = self.clean[index]
        if time:
            known = diffusion.add_noise(known, time, noise=self.noise[index])
        values[(slice(None), slice(None), *target)].copy_(known)

    def condition(self, state, diffusion, time, block_size):
        shape = self.clean.shape[-3:]
        for start in product(*(range(0, n, block_size) for n in shape)):
            source = tuple(
                slice(s, min(s + block_size, n)) for s, n in zip(start, shape)
            )
            target = tuple(
                slice(r.start + p.start, r.start + p.stop)
                for r, p in zip(self.region, source)
            )
            index = (slice(None), slice(None), *source)
            noisy = diffusion.add_noise(
                self.clean[index], time, noise=self.noise[index]
            )
            values = state.read(target).to(device="cpu", dtype=torch.float32)
            values.lerp_(noisy, self.weight[(slice(None), slice(None), *source)])
            state.write(target, values)


def prepare_base(volume, num_phases, shape, margin, offset, shell, preserve):
    if type(preserve) is not bool:
        raise ValueError("preserve_base must be a boolean.")
    if volume is None:
        if offset is not None or preserve:
            raise ValueError("setting base_offset or preserve_base requires base.")
        return None
    base_shape = volume_shape(volume)
    position, explicit = resolve_offset(offset, shape, base_shape)
    clean = phase_fractions(volume, num_phases).mul_(2).sub_(1)
    region = tuple(
        slice(margin + p, margin + p + n) for p, n in zip(position, base_shape)
    )
    axes = []
    for axis, n in enumerate(base_shape):
        weights = torch.ones(n)
        width = min(shell, (n - 1) // 2)
        if not preserve and width and shape[axis] + 2 * margin > n:
            ramp = (
                (torch.arange(1, width + 1).div(width + 1) * (math.pi / 2))
                .sin()
                .square()
            )
            if not explicit[axis] or position[axis] > 0:
                weights[:width] = ramp
            if not explicit[axis] or position[axis] + n < shape[axis]:
                weights[-width:] = ramp.flip(0)
        axes.append(weights)
    weight = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return Base(clean, torch.randn_like(clean), region, weight[None, None], preserve)


def validate_base_anchors(base, anchors, shape, margin) -> Base | None:
    if base is None or not base.preserve:
        return base
    for anchor in anchors:
        axis = anchor.axis
        index = anchor.index + margin
        if not base.region[axis].start <= index < base.region[axis].stop:
            continue
        axes = tuple(a for a in range(3) if a != axis)
        lengths = anchor.image.shape[-2:]
        position = anchor.position or tuple(
            (shape[a] - n) // 2 for a, n in zip(axes, lengths)
        )
        ranges = tuple(
            slice(p + margin, p + margin + n) for p, n in zip(position, lengths)
        )
        region = list(ranges)
        region.insert(axis, slice(index, index + 1))
        parts = base.intersection(region)
        if parts is None:
            continue
        source, target = parts
        image = anchor.image.cpu()
        expected = (
            phase_channels(image[None], base.clean.shape[1])[0]
            if image.ndim == 2
            else image.float()
        )
        expected = expected[(slice(None), *(target[a] for a in axes))] * 2 - 1
        known = base.clean[(0, slice(None), *source)].squeeze(axis + 1)
        if not torch.allclose(known, expected, atol=1e-5, rtol=0):
            raise ValueError("anchor conflicts with the preserved base region.")
    return base
