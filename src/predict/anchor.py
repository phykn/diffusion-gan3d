from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.prepare.resize import phase_channels


@dataclass(frozen=True)
class PlaneAnchor:
    image: torch.Tensor
    axis: int
    index: int
    position: tuple[int, int] | None = None


@dataclass(frozen=True)
class AnchorTarget:
    """Anchor phase values in [-1, 1] and the voxels they cover."""

    image: torch.Tensor
    mask: torch.Tensor


def anchor_origin(
    anchor: PlaneAnchor, image_shape: Sequence[int], plane_shape: Sequence[int]
) -> tuple[int, int]:
    """Top-left (row, col) of an anchor image in its plane; centered by default."""
    height, width = image_shape
    if height > plane_shape[0] or width > plane_shape[1]:
        raise ValueError("anchor.image must fit inside the generated plane.")
    if anchor.position is None:
        return (plane_shape[0] - height) // 2, (plane_shape[1] - width) // 2
    if len(anchor.position) != 2 or any(
        type(value) is not int for value in anchor.position
    ):
        raise ValueError("anchor.position must contain two integer coordinates.")
    row, col = anchor.position
    if (
        row < 0
        or col < 0
        or row + height > plane_shape[0]
        or col + width > plane_shape[1]
    ):
        raise ValueError("anchor.position places the image outside the plane.")
    return row, col


def encode_anchors(
    anchors: Sequence[PlaneAnchor],
    batch_size: int,
    num_phases: int,
    volume_size: int | tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
    validate: bool = True,
) -> AnchorTarget | None:
    if not anchors:
        return None
    shape = (volume_size,) * 3 if isinstance(volume_size, int) else volume_size
    if (
        len(shape) != 3
        or any(type(size) is not int or size < 1 for size in shape)
        or type(batch_size) is not int
        or batch_size < 1
        or type(num_phases) is not int
        or num_phases < 1
    ):
        raise ValueError(
            "anchor volume shape, batch size and phase count must be positive integers."
        )

    mask = torch.zeros(
        batch_size,
        1,
        *shape,
        dtype=torch.bool,
        device=device,
    )
    values = torch.zeros((batch_size, num_phases, *shape), dtype=dtype, device=device)

    for anchor in anchors:
        if type(anchor.axis) is not int or anchor.axis not in (0, 1, 2):
            raise ValueError("anchor.axis must be one of 0, 1, or 2.")
        if type(anchor.index) is not int or not 0 <= anchor.index < shape[anchor.axis]:
            raise ValueError("anchor.index is outside the generated volume.")

        probs = anchor_probabilities(
            anchor.image, batch_size, num_phases, device, dtype, validate
        )
        height, width = probs.shape[-2:]
        plane_shape = tuple(
            size for axis, size in enumerate(shape) if axis != anchor.axis
        )
        row, col = anchor_origin(anchor, (height, width), plane_shape)

        mask_plane = mask.select(anchor.axis + 2, anchor.index).squeeze(1)
        mask_patch = mask_plane[:, row : row + height, col : col + width]
        value_patch = values.select(anchor.axis + 2, anchor.index)[
            :, :, row : row + height, col : col + width
        ]
        encoded = probs.mul(2).sub(1)

        if validate:
            conflict = mask_patch & ((value_patch - encoded).abs().amax(1) > 1e-5)
            if bool(conflict.any()):
                raise ValueError("anchor planes contain conflicting intersections.")
        value_patch.copy_(torch.where(mask_patch.unsqueeze(1), value_patch, encoded))
        mask_patch.fill_(True)

    return AnchorTarget(image=values, mask=mask)


def anchor_probabilities(image, batch_size, num_phases, device, dtype, validate):
    if image.numel() == 0:
        raise ValueError("anchor.image must not be empty.")
    if image.dtype.is_floating_point:
        if image.ndim == 3:
            image = image.unsqueeze(0).expand(batch_size, -1, -1, -1)
        if image.ndim != 4 or image.shape[:2] != (batch_size, num_phases):
            raise ValueError(
                "anchor phase fractions must have shape [C,H,W] or [B,C,H,W]."
            )
        if validate:
            valid = (
                torch.isfinite(image).all() & (image >= 0).all() & (image <= 1).all()
            )
            valid = valid & ((image.sum(1) - 1).abs() < 1e-5).all()
            if not bool(valid):
                raise ValueError(
                    "anchor phase fractions must be finite, non-negative and sum to one."
                )
        probs = image.to(device=device, dtype=dtype)
    else:
        if image.ndim == 2:
            image = image.unsqueeze(0).expand(batch_size, -1, -1)
        if image.ndim != 3 or image.shape[0] != batch_size:
            raise ValueError("anchor.image must have shape [H, W] or [B, H, W].")
        probs = phase_channels(image, num_phases).to(device=device, dtype=dtype)
    return probs


def _owned(value):
    # Tensors made under inference_mode cannot be saved for backward.
    if isinstance(value, torch.Tensor) and value.is_inference():
        return value.clone()
    return value


@dataclass(frozen=True)
class AnchorGuidance:
    """Steer the reverse chain toward anchor planes by posterior sampling.

    The denoiser stays unconditional. Each step moves the whole noisy state
    down the gradient of the clean prediction's squared error on the anchor
    voxels, so nearby slices follow the anchor instead of meeting it at a seam
    (Chung et al., Diffusion Posterior Sampling, 2023).
    """

    image: torch.Tensor
    mask: torch.Tensor
    strength: float

    def __call__(self, model, current, time, latent, **conditions):
        """Clean prediction and the step to subtract from the next noisy state."""
        with torch.inference_mode(False), torch.enable_grad():
            state = current.clone().requires_grad_(True)
            conditions = {name: _owned(value) for name, value in conditions.items()}
            pred = model(state, _owned(time), _owned(latent), **conditions)
            mask = _owned(self.mask)
            error = (pred.float() - _owned(self.image)).square().mul(mask).sum()
            gradient = None
            if error.requires_grad:
                (gradient,) = torch.autograd.grad(error, state, allow_unused=True)
        if gradient is None:
            # A prediction that ignores the noisy state leaves nothing to steer.
            gradient = torch.zeros_like(state)
        # The RMS anchor error, not its norm, keeps the per-voxel step
        # independent of how much anchor area a volume or tile holds.
        rms = (error.detach() / mask.sum().clamp_min(1)).sqrt().clamp_min(1e-8)
        return pred.detach(), gradient * (self.strength / rms)
