import numpy as np
import pytest
import torch

from src.evaluate import (
    compute_fid,
    prepare_fid_images,
)
from src.evaluate import image as image_module


def test_prepare_fid_images_expands_binary_sections_to_uint8_rgb() -> None:
    sections = np.asarray(
        (
            ((0, 1), (1, 0)),
            ((1, 1), (0, 0)),
        ),
        dtype=np.uint8,
    )

    images = prepare_fid_images(sections)

    assert images.shape == (2, 3, 2, 2)
    assert images.dtype == torch.uint8
    assert torch.equal(images[:, 0], torch.from_numpy(sections).mul(255))
    assert torch.equal(images[:, 0], images[:, 1])
    assert torch.equal(images[:, 1], images[:, 2])


def test_fid_scores_real_and_generated_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = []

    class FakeMetric:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.updates = []
            created.append(self)

        def to(self, device):
            self.device = torch.device(device)
            return self

        def update(self, images, *, real):
            self.updates.append((images.clone(), real))

        def compute(self):
            return torch.tensor(4.5)

    real = np.zeros((2, 4, 4), dtype=np.uint8)
    generated = np.ones((2, 4, 4), dtype=np.uint8)
    monkeypatch.setattr(image_module, "FrechetInceptionDistance", FakeMetric)
    assert compute_fid(real, generated, "cpu", 64) == pytest.approx(4.5)

    assert len(created) == 1
    metric = created[0]
    assert metric.kwargs["feature"] == 64
    assert [real_flag for _, real_flag in metric.updates] == [True, False]
