import math
from dataclasses import dataclass

import torch

from src.data.phases import encode_phases

CONFLICT_TOLERANCE = 1e-5
SUM_TOLERANCE = 1e-5
SMALLEST_ERROR = 1e-8


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
    position: tuple | None = None


@dataclass(frozen=True)
class _PlacedAnchor:
    values: torch.Tensor  # [K, h, w] in [-1, 1]
    axis: int
    index: int
    row: int
    col: int

    @property
    def plane_axes(self):
        return tuple(axis for axis in range(3) if axis != self.axis)


@dataclass(frozen=True)
class AnchorGuide:
    # The denoiser never sees an anchor. Each step moves the noisy state down the
    # gradient of the prediction's error on the anchor voxels (DPS, Chung 2023).
    image: torch.Tensor
    mask: torch.Tensor
    strength: float

    def __call__(self, predictor, noisy, time, latent, guidance, conditions):
        with torch.enable_grad():
            state = noisy.detach().clone().requires_grad_(True)
            prediction = predictor(state, time, latent, guidance, **conditions)
            error = (prediction - self.image).square().mul(self.mask).sum()
            (gradient,) = torch.autograd.grad(error, state)
        # The RMS error keeps the step the same however large the anchor area is.
        anchor_voxels = self.mask.sum().clamp_min(1)
        rms = (error.detach() / anchor_voxels).sqrt().clamp_min(SMALLEST_ERROR)
        return prediction.detach(), gradient * (self.strength / rms)


class TileAnchors:
    def __init__(self, anchors, plan, num_phases, strength, device=None):
        self.strength = check_strength(strength)
        self.device = device
        placed = place_anchors(anchors, plan.output_shape, num_phases, plan.offsets)
        self.anchors_per_tile = []
        if not placed:
            return
        for tile in plan.tiles:
            in_tile = [
                cropped
                for cropped in (_crop_to_tile(anchor, tile) for anchor in placed)
                if cropped is not None
            ]
            # Every voxel lies in some tile, so checking each tile checks all.
            if in_tile:
                encode_anchors(in_tile, tile.shape, check=True)
            self.anchors_per_tile.append(in_tile)

    def __call__(self, index, tile):
        in_tile = self.anchors_per_tile[index]
        if not in_tile:
            return None
        image, mask = encode_anchors(in_tile, tile.shape, self.device)
        return AnchorGuide(image, mask, self.strength)

    @property
    def is_active(self):
        return self.strength > 0 and any(self.anchors_per_tile)


def check_strength(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("anchor_strength must be finite and non-negative.")
    return float(value)


def place_anchors(anchors, output_shape, num_phases, offsets=(0, 0, 0)):
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
        values = _read_fractions(anchor.image, num_phases).mul(2).sub(1)
        plane_shape = tuple(
            output_shape[axis] for axis in range(3) if axis != anchor.axis
        )
        row, col = _locate_image(anchor, values.shape[-2:], plane_shape)
        row_offset, col_offset = (
            offsets[axis] for axis in range(3) if axis != anchor.axis
        )
        placed.append(
            _PlacedAnchor(
                values,
                anchor.axis,
                anchor.index + offsets[anchor.axis],
                row + row_offset,
                col + col_offset,
            )
        )
    return placed


def encode_anchors(placed, shape, device=None, *, check=False):
    # Returns the anchor values [1, K, *shape] and which voxels they cover.
    num_phases = placed[0].values.shape[0]
    image = torch.zeros((1, num_phases, *shape), device=device)
    mask = torch.zeros((1, 1, *shape), device=device, dtype=torch.bool)
    for anchor in placed:
        height, width = anchor.values.shape[-2:]
        rows = slice(anchor.row, anchor.row + height)
        cols = slice(anchor.col, anchor.col + width)
        covered = mask.select(anchor.axis + 2, anchor.index)[:, :, rows, cols]
        current = image.select(anchor.axis + 2, anchor.index)[:, :, rows, cols]
        values = anchor.values.to(device)[None]
        if check:
            difference = (current - values).abs().amax(1, keepdim=True)
            if bool((covered & (difference > CONFLICT_TOLERANCE)).any()):
                raise ValueError("anchor planes contain conflicting intersections.")
        current.copy_(torch.where(covered, current, values))
        covered.fill_(True)
    return image, mask


def _read_fractions(image, num_phases):
    if not isinstance(image, torch.Tensor) or image.numel() == 0:
        raise ValueError("anchor.image must be a non-empty tensor.")
    if not image.dtype.is_floating_point:
        if image.ndim != 2:
            raise ValueError("anchor labels must have shape [H, W].")
        return encode_phases(image.detach().cpu()[None], num_phases)[0]
    if image.ndim != 3 or image.shape[0] != num_phases:
        raise ValueError("anchor phase fractions must have shape [C, H, W].")
    image = image.detach().to("cpu", torch.float32)
    is_valid = bool(
        torch.isfinite(image).all()
        & (image >= 0).all()
        & (image <= 1).all()
        & ((image.sum(0) - 1).abs() < SUM_TOLERANCE).all()
    )
    if not is_valid:
        raise ValueError(
            "anchor phase fractions must be finite, non-negative and sum to one."
        )
    return image


def _locate_image(anchor, image_shape, plane_shape):
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


def _crop_to_tile(anchor, tile):
    across = tile.source[anchor.axis]
    if not across.start <= anchor.index < across.stop:
        return None
    rows, cols = (tile.source[axis] for axis in anchor.plane_axes)
    height, width = anchor.values.shape[-2:]
    top, left = max(anchor.row, rows.start), max(anchor.col, cols.start)
    bottom = min(anchor.row + height, rows.stop)
    right = min(anchor.col + width, cols.stop)
    if top >= bottom or left >= right:
        return None
    return _PlacedAnchor(
        anchor.values[
            :,
            top - anchor.row : bottom - anchor.row,
            left - anchor.col : right - anchor.col,
        ],
        anchor.axis,
        anchor.index - across.start,
        top - rows.start,
        left - cols.start,
    )
