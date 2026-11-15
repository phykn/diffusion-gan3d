from src.data.resize import downsample


def consistency_loss(high, low, tolerance, mask):
    reconstructed = downsample(high, tuple(low.shape[2:]))
    squared = (reconstructed - low).square()
    mask = mask.to(squared)
    error = (squared * mask).sum() / (mask.sum() * low.shape[1]).clamp_min(1)
    # A dead zone lets fine boundaries adapt without forcing block-shaped phases.
    return (error - tolerance).clamp_min(0), error
