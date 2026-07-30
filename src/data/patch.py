import numpy as np
from PIL import Image


def random_crop(labels: np.ndarray, size: int) -> np.ndarray:
    if labels.ndim != 2:
        raise ValueError("labels must be a two-dimensional array.")
    height, width = labels.shape
    if size <= 0 or size > min(height, width):
        raise ValueError("crop size must fit inside the image.")
    top = int(np.random.randint(0, height - size + 1))
    left = int(np.random.randint(0, width - size + 1))
    return labels[top : top + size, left : left + size]


def resize_labels(labels: np.ndarray, size: int) -> np.ndarray:
    if labels.ndim != 2 or labels.dtype != np.uint8:
        raise ValueError("labels must be a two-dimensional uint8 array.")
    if size <= 0:
        raise ValueError("resize size must be positive.")
    image = Image.fromarray(labels, mode="L")
    return np.asarray(
        image.resize((size, size), resample=Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
