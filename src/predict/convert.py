"""Conversions of owned inference outputs; never use for shared training tensors."""

import math

import torch

LABEL_CHUNK_VOXELS = 1024**2


def label_chunk_depth(shape) -> int:
    return min(shape[0], max(1, LABEL_CHUNK_VOXELS // math.prod(shape[1:])))


@torch.no_grad()
def labels_from_channels(channels: torch.Tensor) -> torch.Tensor:
    """C,D,H,W ranks to CPU uint8, with at most one depth slab of int64 indices."""
    shape = channels.shape[1:]
    labels = torch.empty(shape, dtype=torch.uint8, device="cpu")
    depth = label_chunk_depth(shape)
    for start in range(0, shape[0], depth):
        region = slice(start, start + depth)
        labels[region].copy_(channels[:, region].argmax(0).to(torch.uint8))
    return labels


@torch.no_grad()
def owned_clean_to_probs_(clean: torch.Tensor) -> torch.Tensor:
    """Consume an exclusively owned decoded output (B,C,...), reusing fp32 storage."""
    probs = clean.float().add_(1).mul_(0.5).clamp_(0, 1)
    total = probs.sum(1, keepdim=True).clamp_min_(torch.finfo(probs.dtype).eps)
    return probs.div_(total)
