from pathlib import Path

import numpy as np
from PIL import Image


def load_labels(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(image)
    if values.ndim != 2:
        raise ValueError(f"categorical image must be two-dimensional: {path}")
    if values.dtype != np.uint8:
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"categorical image must contain integer labels: {path}")
        if values.min() < 0 or values.max() > 255:
            raise ValueError(f"categorical image labels must fit uint8: {path}")
        values = values.astype(np.uint8)
    return np.array(values, copy=True)
