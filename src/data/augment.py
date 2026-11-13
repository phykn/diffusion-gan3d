from dataclasses import dataclass
from itertools import permutations

import torch

from src.axes import PLANE_DIRECTIONS, PLANES

# Transform n turns a section n % 4 quarter turns, mirrored first when n >= 4.
IDENTITY, HALF_TURN, FLIP_LEFT_RIGHT, FLIP_UP_DOWN = 0, 2, 4, 6


@dataclass(frozen=True)
class PlaneTransforms:
    flippable_axes: tuple
    rotatable: bool


@dataclass(frozen=True)
class VolumeTransforms:
    axis_orders: tuple
    flippable_axes: tuple


class SectionAugment:
    def __init__(self, plane_transforms, probability):
        self.probability = probability
        self.allowed_transforms = {
            PLANES.index(plane): self._list_transforms(plane, allowed)
            for plane, allowed in plane_transforms.items()
        }
        self._pixel_maps = {}

    def __call__(self, sections, axis, generator):
        sections = tuple(sections)
        if self.probability <= 0.0:
            return sections
        count, *_, height, width = sections[0].shape
        candidates = [
            transform
            for transform in self.allowed_transforms[axis]
            if transform != IDENTITY
            and (height == width or not self._is_quarter_turn(transform))
        ]
        candidates = torch.tensor(candidates or [IDENTITY])
        chosen = candidates[
            torch.randint(len(candidates), (count,), generator=generator)
        ]
        untouched = torch.rand(count, generator=generator) >= self.probability
        chosen.masked_fill_(untouched, IDENTITY)
        return tuple(self._transform(section, chosen) for section in sections)

    @property
    def rotatable_axes(self):
        if self.probability <= 0.0:
            return ()
        return tuple(
            axis
            for axis, transforms in self.allowed_transforms.items()
            if any(map(self._is_quarter_turn, transforms))
        )

    @staticmethod
    def _list_transforms(plane, allowed):
        if allowed.rotatable:
            return tuple(range(8 if allowed.flippable_axes else 4))
        row_axis, column_axis = PLANE_DIRECTIONS[plane]
        flip_rows = row_axis in allowed.flippable_axes
        flip_columns = column_axis in allowed.flippable_axes
        transforms = [IDENTITY]
        if flip_rows and flip_columns:
            transforms.append(HALF_TURN)
        if flip_columns:
            transforms.append(FLIP_LEFT_RIGHT)
        if flip_rows:
            transforms.append(FLIP_UP_DOWN)
        return tuple(transforms)

    @staticmethod
    def _is_quarter_turn(transform):
        return transform % 2 == 1

    def _transform(self, sections, chosen):
        height, width = sections.shape[-2:]
        pixel_maps = self._build_pixel_maps(sections.device, height, width)
        flat = sections.reshape(len(sections), -1, height * width)
        source_pixels = pixel_maps[chosen.to(sections.device)]
        source_pixels = source_pixels[:, None].expand(-1, flat.shape[1], -1)
        return flat.gather(2, source_pixels).reshape_as(sections)

    def _build_pixel_maps(self, device, height, width):
        key = (device, height, width)
        if key not in self._pixel_maps:
            pixels = torch.arange(height * width, device=device).reshape(height, width)
            mirrored = pixels.flip(-1)
            self._pixel_maps[key] = torch.stack(
                [
                    torch.rot90(
                        mirrored if transform >= FLIP_LEFT_RIGHT else pixels,
                        transform % 4,
                        dims=(-2, -1),
                    ).reshape(-1)
                    for transform in range(8)
                ]
            )
        return self._pixel_maps[key]


def build_plane_transforms(config):
    has_height = config.conditioning.height_enabled
    sides_match_xy = any(
        "xy" in group and len(group) > 1 for group in config.model.critic.plane_groups
    )
    side_rotatable = sides_match_xy and not has_height
    result = {}
    for plane, axes in PLANE_DIRECTIONS.items():
        is_side = "z" in axes
        result[plane] = PlaneTransforms(
            flippable_axes=tuple(
                axis for axis in axes if not (has_height and axis == "z")
            ),
            rotatable=side_rotatable or not is_side,
        )
    return result


def build_volume_transforms(config):
    # Swaps are reflections too and can reverse the tilt carried by LR.
    if not config.lr_bank.flips:
        return VolumeTransforms(axis_orders=((0, 1, 2),), flippable_axes=())
    side = build_plane_transforms(config)["xz"]
    z_swappable = side.rotatable
    z_flippable = "z" in side.flippable_axes
    return VolumeTransforms(
        axis_orders=tuple(
            order for order in permutations(range(3)) if z_swappable or order[0] == 0
        ),
        flippable_axes=(0, 1, 2) if z_flippable else (1, 2),
    )


def augment_volumes(volumes, transforms, generator):
    shape = volumes.shape[-3:]
    # An order that changes the shape could not be stacked with the others.
    axis_orders = [
        order
        for order in transforms.axis_orders
        if tuple(shape[axis] for axis in order) == shape
    ]
    flippable_axes = transforms.flippable_axes
    flip_patterns = 2 ** len(flippable_axes)
    choices = torch.randint(
        len(axis_orders) * flip_patterns, (len(volumes),), generator=generator
    ).tolist()
    augmented = []
    for volume, choice in zip(volumes, choices):
        order_index, flip_bits = divmod(choice, flip_patterns)
        volume = volume.permute(0, *(axis + 1 for axis in axis_orders[order_index]))
        flipped_dims = [
            axis + 1 for bit, axis in enumerate(flippable_axes) if flip_bits >> bit & 1
        ]
        augmented.append(volume.flip(flipped_dims) if flipped_dims else volume)
    return torch.stack(augmented)
