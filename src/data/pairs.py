import torch


def sample_pairs(previous, current, axis, count, generator, height=None):
    dim = axis + 2
    size = previous.shape[dim]
    choices = torch.randint(previous.shape[0] * size, (count,), generator=generator)
    choices = choices.to(previous.device)
    indices = (choices // size, slice(None), choices % size)
    volumes = (
        (previous, current) if height is None else (previous, current, height.detach())
    )
    return tuple(volume.movedim(dim, 2)[indices] for volume in volumes)
