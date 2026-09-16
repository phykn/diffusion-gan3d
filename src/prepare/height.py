import torch


def height_field(shape, axis, origin, pixel_size, extent, device=None):
    """Cell-center height in [-1, 1]; origin and extent use source-image pixels."""
    origin = torch.as_tensor(origin, device=device, dtype=torch.float32).reshape(-1, 1)
    values = origin + (torch.arange(shape[axis], device=device) + 0.5) * pixel_size
    values = (2 * values / extent - 1).clamp(-1, 1)
    dims = [len(origin), 1] + [1] * len(shape)
    dims[axis + 2] = shape[axis]
    return values.reshape(dims).expand(len(origin), 1, *shape)
