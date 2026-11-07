"""Orientation policy: which flips and rotations keep a section realistic.

plane_policies() is the one place this is decided; critic augmentation and
SR bank augmentation both read it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations

import torch

from src.plane import PLANE_DIRECTIONS, PLANES


@dataclass(frozen=True)
class PlanePolicy:
    flip_axes: tuple[str, ...]
    rotate_90: bool


def fixed_z(config) -> bool:
    """Rotating xz/yz swaps z with x or y. That hides a height gradient, and
    hides z anisotropy whenever xy is judged apart from the side planes."""
    groups = config.model.critic.plane_groups
    return config.conditioning.height_enabled or not any(
        "xy" in group and len(group) > 1 for group in groups
    )


def plane_policies(config) -> dict[str, PlanePolicy]:
    height = config.conditioning.height_enabled
    fixed = fixed_z(config)
    return {
        plane: PlanePolicy(
            flip_axes=tuple(axis for axis in axes if not (height and axis == "z")),
            rotate_90=not (fixed and "z" in axes),
        )
        for plane, axes in PLANE_DIRECTIONS.items()
    }


def policy_dict(policies: Mapping[str, PlanePolicy]) -> dict:
    """The v1 augmentation.planes form of the policies."""
    return {
        plane: {"flip_axes": list(policy.flip_axes), "rotate_90": policy.rotate_90}
        for plane, policy in policies.items()
    }


def _allowed_transforms(plane: str, policy: PlanePolicy) -> tuple[int, ...]:
    """Indices into the 8 square symmetries: rot90 k times, then flip if >= 4."""
    rows, cols = PLANE_DIRECTIONS[plane]
    flips = policy.flip_axes
    allowed = {0}
    if cols in flips:
        allowed.add(4)
    if rows in flips:
        allowed.add(6)
    if len(flips) == 2:
        allowed.add(2)
    if policy.rotate_90:
        allowed = set(range(8)) if flips else set(range(4))
    return tuple(sorted(allowed))


class CriticAugment:
    """With probability p, apply one non-identity allowed symmetry to a section.

    Pairs (and their height fields) of one sample get the same transform.
    """

    def __init__(self, policies: Mapping[str, PlanePolicy], probability: float):
        self.probability = float(probability)
        self.transforms = {
            PLANES.index(plane): _allowed_transforms(plane, policy)
            for plane, policy in policies.items()
        }
        self._maps: dict[tuple, torch.Tensor] = {}

    @property
    def rotated_axes(self) -> tuple[int, ...]:
        if self.probability <= 0.0:
            return ()
        return tuple(
            axis
            for axis, allowed in self.transforms.items()
            if any(index % 2 for index in allowed)
        )

    def __call__(
        self,
        inputs: Sequence[torch.Tensor],
        axis: int,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, ...]:
        tensors = tuple(inputs)
        first = tensors[0]
        if self.probability <= 0.0 or first.shape[0] == 0:
            return tensors
        batch = first.shape[0]
        square = first.shape[-2] == first.shape[-1]
        allowed = self.transforms[axis]
        if not square:
            allowed = tuple(index for index in allowed if index % 2 == 0)
        choices = torch.tensor(tuple(i for i in allowed if i != 0) or (0,))
        selected = choices[torch.randint(len(choices), (batch,), generator=generator)]
        keep = torch.rand(batch, generator=generator) >= self.probability
        selected.masked_fill_(keep, 0)
        return tuple(self._apply(tensor, selected) for tensor in tensors)

    def _apply(self, inputs: torch.Tensor, transforms: torch.Tensor) -> torch.Tensor:
        height, width = inputs.shape[-2:]
        maps = self._index_maps(inputs.device, height, width)
        indices = maps.index_select(0, transforms.to(inputs.device))
        flattened = inputs.reshape(inputs.shape[0], -1, height * width)
        indices = indices.unsqueeze(1).expand(-1, flattened.shape[1], -1)
        return flattened.gather(2, indices).reshape_as(inputs)

    def _index_maps(self, device, height, width) -> torch.Tensor:
        key = (device, height, width)
        if key not in self._maps:
            source = torch.arange(height * width, device=device).reshape(height, width)
            maps = []
            for index in range(8):
                grid = torch.flip(source, dims=(-1,)) if index >= 4 else source
                maps.append(torch.rot90(grid, index % 4, dims=(-2, -1)).reshape(-1))
            self._maps[key] = torch.stack(maps)
        return self._maps[key]


@dataclass(frozen=True)
class BankSymmetries:
    """Volume symmetries allowed for SR bank volumes (axis orders and flips)."""

    orders: tuple[tuple[int, int, int], ...]
    flip_axes: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.orders) * 2 ** len(self.flip_axes)


def bank_symmetries(config) -> BankSymmetries:
    """z swaps or flips only where the z plane policies allow them.

    With lr_bank.flips false only the identity is left: an axis swap is a
    reflection too and would reverse a tilt the LR volume carries.
    """
    if not config.lr_bank.flips:
        return BankSymmetries(orders=((0, 1, 2),), flip_axes=())
    policies = plane_policies(config)
    z_policies = [
        policy for plane, policy in policies.items() if "z" in PLANE_DIRECTIONS[plane]
    ]
    free_z = not config.conditioning.height_enabled
    permute_z = free_z and any(policy.rotate_90 for policy in z_policies)
    flip_z = free_z and any("z" in policy.flip_axes for policy in z_policies)
    orders = tuple(
        order for order in permutations(range(3)) if permute_z or order[0] == 0
    )
    return BankSymmetries(orders=orders, flip_axes=((0,) if flip_z else ()) + (1, 2))


def augment_volumes(
    volumes: torch.Tensor, symmetries: BankSymmetries, generator: torch.Generator
) -> torch.Tensor:
    """Apply one allowed symmetry per [B, C, D, H, W] volume (cubes only permute)."""
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
    for volume, choice in zip(volumes, choices, strict=True):
        order = orders[choice // flip_count]
        volume = volume.permute(0, *(axis + 1 for axis in order))
        dims = [
            axis + 1
            for bit, axis in enumerate(flips)
            if (choice % flip_count) & (1 << bit)
        ]
        result.append(volume.flip(dims) if dims else volume)
    return torch.stack(result)
