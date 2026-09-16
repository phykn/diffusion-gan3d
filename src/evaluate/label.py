import torch

from src.plane import AXES


def compute_vf(
    batches: dict[int, torch.Tensor],
    num_phases: int,
) -> torch.Tensor:
    if not batches or not set(batches).issubset(AXES):
        raise ValueError("batches must contain at least one valid axis.")
    if any(
        not isinstance(images, torch.Tensor) or images.ndim not in (3, 4)
        for images in batches.values()
    ):
        raise ValueError("training crops must have shape [B,H,W] or [B,C,H,W].")
    counts = []
    total = 0
    for images in batches.values():
        if images.numel() == 0:
            raise ValueError("training crops must not be empty.")
        if images.ndim == 4:
            if images.shape[1] != num_phases:
                raise ValueError("phase channels must match num_phases.")
            counts.append(images.float().sum(dim=(0, 2, 3)))
            total += images.numel() // num_phases
            continue
        values = images.to(torch.long)
        valid = ((values >= 0) & (values < num_phases)).all()
        if values.device.type == "cuda":
            torch._assert_async(
                valid, "training images contain a phase outside num_phases."
            )
        elif not bool(valid):
            raise ValueError("training images contain a phase outside num_phases.")
        counts.append(torch.bincount(values.flatten(), minlength=num_phases).float())
        total += values.numel()
    return torch.stack(counts).sum(0).div(total)


def phase_fraction(values, phase: int = 0) -> float:
    labels = torch.as_tensor(values)
    return float((labels == phase).to(torch.float64).mean())


def phase_fractions(values, num_phases: int) -> torch.Tensor:
    labels = torch.as_tensor(values).reshape(-1).to(torch.long)
    counts = torch.bincount(labels, minlength=num_phases)
    return counts.to(torch.float64).div(labels.numel())


def voxel_accuracy(actual, expected) -> float:
    actual = torch.as_tensor(actual)
    expected = torch.as_tensor(expected)
    return float((actual == expected).to(torch.float64).mean())
