import numpy as np
import torch

from src.evaluate import metric_images


def test_metric_images_preserves_input_and_scales_normalized_grayscale() -> None:
    sections = np.asarray((((0.0, 0.5), (1.0, 0.25)),), dtype=np.float32)
    original = sections.copy()

    images = metric_images(sections)

    assert np.array_equal(sections, original)
    assert torch.equal(
        images[0, 0],
        torch.tensor(((0, 128), (255, 64)), dtype=torch.uint8),
    )
