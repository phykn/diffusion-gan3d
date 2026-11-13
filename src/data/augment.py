from dataclasses import dataclass
from itertools import permutations

import torch

from src.axes import PLANE_DIRECTIONS, PLANES


@dataclass(frozen=True)
class PlanePolicy:
    flip_axes: tuple
    rotate_90: bool


def build_plane_policies(config):
    height = config.conditioning.height_enabled
    groups = config.model.critic.plane_groups
    # A side rotation would hide a depth gradient or exchange anisotropic axes.
    fixed = height or not any("xy" in group and len(group) > 1 for group in groups)
    return {
        plane: PlanePolicy(
            flip_axes=tuple(axis for axis in axes if not (height and axis == "z")),
            rotate_90=not (fixed and "z" in axes),
        )
        for plane, axes in PLANE_DIRECTIONS.items()
    }


class SectionAugment:
    def __init__(self, policies, probability):
        self.probability = float(probability)
        self.transforms = {
            PLANES.index(plane): _list_transforms(plane, policy)
            for plane, policy in policies.items()
        }
        self._maps = {}

    def __call__(self, sections, axis, generator):
        sections = tuple(sections)
        if self.probability <= 0.0:
            return sections
        batch, *_, height, width = sections[0].shape
        allowed = self.transforms[axis]
        if height != width:
            allowed = tuple(index for index in allowed if index % 2 == 0)
        choices = torch.tensor(tuple(index for index in allowed if index != 0) or (0,))
        selected = choices[torch.randint(len(choices), (batch,), generator=generator)]
        keep = torch.rand(batch, generator=generator) >= self.probability
        selected.masked_fill_(keep, 0)
        return tuple(self._transform(section, selected) for section in sections)

    def _transform(self, sections, selected):
        height, width = sections.shape[-2:]
        maps = self._build_maps(sections.device, height, width)
        indices = maps.index_select(0, selected.to(sections.device))
        flattened = sections.reshape(sections.shape[0], -1, height * width)
        indices = indices.unsqueeze(1).expand(-1, flattened.shape[1], -1)
        return flattened.gather(2, indices).reshape_as(sections)

    def _build_maps(self, device, height, width):
        key = (device, height, width)
        if key not in self._maps:
            source = torch.arange(height * width, device=device).reshape(height, width)
            maps = []
            for index in range(8):
                grid = torch.flip(source, dims=(-1,)) if index >= 4 else source
                maps.append(torch.rot90(grid, index % 4, dims=(-2, -1)).reshape(-1))
            self._maps[key] = torch.stack(maps)
        return self._maps[key]

    @property
    def rotated_axes(self):
        if self.probability <= 0.0:
            return ()
        return tuple(
            axis
            for axis, allowed in self.transforms.items()
            if any(index % 2 for index in allowed)
        )


def _list_transforms(plane, policy):
    if policy.rotate_90:
        return tuple(range(8 if policy.flip_axes else 4))
    rows, cols = PLANE_DIRECTIONS[plane]
    flips = policy.flip_axes
    # Without rotation: 2 flips both axes, 4 flips the columns, 6 flips the rows.
    return tuple(
        index
        for index, enabled in (
            (0, True),
            (2, len(flips) == 2),
            (4, cols in flips),
            (6, rows in flips),
        )
        if enabled
    )


@dataclass(frozen=True)
class VolumeSymmetries:
    orders: tuple
    flip_axes: tuple


def build_symmetries(config):
    # Swaps are reflections too and can reverse the tilt carried by LR.
    if not config.lr_bank.flips:
        return VolumeSymmetries(orders=((0, 1, 2),), flip_axes=())
    policies = build_plane_policies(config)
    z_policies = [
        policy for plane, policy in policies.items() if "z" in PLANE_DIRECTIONS[plane]
    ]
    free_z = not config.conditioning.height_enabled
    permute_z = free_z and any(policy.rotate_90 for policy in z_policies)
    flip_z = free_z and any("z" in policy.flip_axes for policy in z_policies)
    orders = tuple(
        order for order in permutations(range(3)) if permute_z or order[0] == 0
    )
    return VolumeSymmetries(orders=orders, flip_axes=((0,) if flip_z else ()) + (1, 2))


def augment_volumes(volumes, symmetries, generator):
    shape = volumes.shape[-3:]
    orders = [
        order
        for order in symmetries.orders
        if tuple(shape[axis] for axis in order) == shape
    ]
    flips = symmetries.flip_axes
    flip_count = 2 ** len(flips)
    choices = torch.randint(
        len(orders) * flip_count, (len(volumes),), generator=generator
    ).tolist()
    result = []
    for volume, choice in zip(volumes, choices):
        order_index, flip_mask = divmod(choice, flip_count)
        volume = volume.permute(0, *(axis + 1 for axis in orders[order_index]))
        dims = [axis + 1 for bit, axis in enumerate(flips) if flip_mask & (1 << bit)]
        result.append(volume.flip(dims) if dims else volume)
    return torch.stack(result)
