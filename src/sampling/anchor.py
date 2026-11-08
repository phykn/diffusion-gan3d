import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.data.crops import phase_channels
from src.sampling.plan import Tile, TilePlan

CONFLICT_TOLERANCE = 1e-5


@dataclass(frozen=True)
class PlaneAnchor:
    """A known section of the output.

    image is integer labels [H, W] or phase fractions [C, H, W]. axis 0, 1, 2
    is the xy, xz or yz plane at index along that axis. position is the
    top-left (row, col) in the plane; None centers the image. Coordinates are
    output voxels.
    """

    image: torch.Tensor
    axis: int
    index: int
    position: tuple[int, int] | None = None


@dataclass(frozen=True)
class _Placed:
    """An anchor's values in [-1, 1] at its place in a region."""

    values: torch.Tensor  # [K, h, w]
    axis: int
    index: int
    row: int
    col: int

    @property
    def axes(self) -> tuple[int, int]:
        return tuple(axis for axis in range(3) if axis != self.axis)


def check_strength(value) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("anchor_strength must be finite and non-negative.")
    return float(value)


def anchor_fractions(image: torch.Tensor, num_phases: int) -> torch.Tensor:
    """Phase fractions float32 [K, H, W] of an anchor image."""
    if not isinstance(image, torch.Tensor) or image.numel() == 0:
        raise ValueError("anchor.image must be a non-empty tensor.")
    if image.dtype.is_floating_point:
        if image.ndim != 3 or image.shape[0] != num_phases:
            raise ValueError("anchor phase fractions must have shape [C, H, W].")
        image = image.detach().to("cpu", torch.float32)
        valid = bool(
            torch.isfinite(image).all()
            & (image >= 0).all()
            & (image <= 1).all()
            & ((image.sum(0) - 1).abs() < 1e-5).all()
        )
        if not valid:
            raise ValueError(
                "anchor phase fractions must be finite, non-negative and sum to one."
            )
        return image
    if image.ndim != 2:
        raise ValueError("anchor labels must have shape [H, W].")
    return phase_channels(image.detach().cpu()[None], num_phases)[0]


def anchor_origin(anchor: PlaneAnchor, image_shape, plane_shape) -> tuple[int, int]:
    """Top-left (row, col) of an anchor image in its plane; centered by default."""
    height, width = image_shape
    if height > plane_shape[0] or width > plane_shape[1]:
        raise ValueError("anchor.image must fit inside the generated plane.")
    if anchor.position is None:
        return (plane_shape[0] - height) // 2, (plane_shape[1] - width) // 2
    position = tuple(anchor.position)
    if len(position) != 2 or any(type(value) is not int for value in position):
        raise ValueError("anchor.position must contain two integer coordinates.")
    row, col = position
    if (
        row < 0
        or col < 0
        or row + height > plane_shape[0]
        or col + width > plane_shape[1]
    ):
        raise ValueError("anchor.position places the image outside the plane.")
    return row, col


def place_anchors(
    anchors: Sequence, output_shape: Sequence[int], num_phases: int, margin: int = 0
) -> list[_Placed]:
    """Check anchors against the output; place them in the generated region."""
    placed = []
    for anchor in anchors:
        if not isinstance(anchor, PlaneAnchor):
            raise TypeError("anchors must be PlaneAnchor values.")
        # A bool would pass the range checks as 0 or 1.
        if type(anchor.axis) is not int or anchor.axis not in (0, 1, 2):
            raise ValueError("anchor.axis must be one of 0, 1, or 2.")
        if (
            type(anchor.index) is not int
            or not 0 <= anchor.index < output_shape[anchor.axis]
        ):
            raise ValueError("anchor.index is outside the generated volume.")
        values = anchor_fractions(anchor.image, num_phases).mul(2).sub(1)
        axes = tuple(axis for axis in range(3) if axis != anchor.axis)
        row, col = anchor_origin(
            anchor, values.shape[-2:], tuple(output_shape[axis] for axis in axes)
        )
        placed.append(
            _Placed(
                values, anchor.axis, anchor.index + margin, row + margin, col + margin
            )
        )
    return placed


def encode(
    placed: Sequence[_Placed], shape: Sequence[int], device=None, *, check=False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anchor values [1, K, *shape] in [-1, 1] and their voxels [1, 1, *shape].

    check raises ValueError where two anchors give one voxel different values.
    """
    phases = placed[0].values.shape[0]
    image = torch.zeros((1, phases, *shape), device=device)
    mask = torch.zeros((1, 1, *shape), device=device, dtype=torch.bool)
    for anchor in placed:
        height, width = anchor.values.shape[-2:]
        rows = slice(anchor.row, anchor.row + height)
        cols = slice(anchor.col, anchor.col + width)
        known = mask.select(anchor.axis + 2, anchor.index)[:, :, rows, cols]
        current = image.select(anchor.axis + 2, anchor.index)[:, :, rows, cols]
        values = anchor.values.to(device)[None]
        if check:
            conflict = known & (
                (current - values).abs().amax(1, keepdim=True) > CONFLICT_TOLERANCE
            )
            if bool(conflict.any()):
                raise ValueError("anchor planes contain conflicting intersections.")
        current.copy_(torch.where(known, current, values))
        known.fill_(True)
    return image, mask


def _crop(anchor: _Placed, tile: Tile) -> _Placed | None:
    """The part of an anchor inside a tile, in tile coordinates."""
    normal = tile.source[anchor.axis]
    if not normal.start <= anchor.index < normal.stop:
        return None
    rows, cols = (tile.source[axis] for axis in anchor.axes)
    height, width = anchor.values.shape[-2:]
    top, left = max(anchor.row, rows.start), max(anchor.col, cols.start)
    bottom = min(anchor.row + height, rows.stop)
    right = min(anchor.col + width, cols.stop)
    if top >= bottom or left >= right:
        return None
    return _Placed(
        anchor.values[
            :,
            top - anchor.row : bottom - anchor.row,
            left - anchor.col : right - anchor.col,
        ],
        anchor.axis,
        anchor.index - normal.start,
        top - rows.start,
        left - cols.start,
    )


@dataclass(frozen=True)
class AnchorGuide:
    """One tile's anchor target and the DPS guide step toward it.

    The denoiser never sees an anchor. Each reverse step but the last moves the
    noisy state down the gradient of the clean prediction's squared error on the
    anchor voxels, so nearby sections follow the anchor instead of meeting it at a
    seam (Chung et al., Diffusion Posterior Sampling, 2023).
    """

    image: torch.Tensor
    mask: torch.Tensor
    strength: float

    def __call__(self, predictor, values, time, latent, guidance, conditions):
        """Clean prediction and the step to subtract from the next noisy state."""
        with torch.enable_grad():
            state = values.detach().clone().requires_grad_(True)
            pred = predictor(state, time, latent, guidance, **conditions)
            error = (pred - self.image).square().mul(self.mask).sum()
            (gradient,) = torch.autograd.grad(error, state)
        # The RMS anchor error, not its norm, keeps the per-voxel step
        # independent of how much anchor area a volume or tile holds.
        rms = (error.detach() / self.mask.sum().clamp_min(1)).sqrt().clamp_min(1e-8)
        return pred.detach(), gradient * (self.strength / rms)


class TileAnchors:
    """Anchors checked and placed once per plan; guide(index, tile) per step."""

    def __init__(
        self,
        anchors: Sequence,
        plan: TilePlan,
        num_phases: int,
        strength: float,
        device=None,
    ):
        self.strength = check_strength(strength)
        self.device = device
        placed = place_anchors(anchors, plan.output_shape, num_phases, plan.margin)
        self.tiles = []
        for tile in plan.tiles:
            local = [
                part
                for part in (_crop(anchor, tile) for anchor in placed)
                if part is not None
            ]
            # Every voxel lies in some tile, so checking each tile checks all.
            if local:
                encode(local, tile.shape, check=True)
            self.tiles.append(local)

    @property
    def active(self) -> bool:
        return self.strength > 0 and any(self.tiles)

    def __call__(self, index: int, tile: Tile) -> AnchorGuide | None:
        local = self.tiles[index]
        if not local:
            return None
        image, mask = encode(local, tile.shape, self.device)
        return AnchorGuide(image, mask, self.strength)
