import numpy as np
import pytest
import torch

import src.evaluate.fid as fid_module
from src.evaluate.fid import compute_fid
from src.evaluate.image import prepare_images


def test_prepare_images_expands_binary_sections_to_uint8_rgb() -> None:
    sections = np.asarray(
        (
            ((0, 1), (1, 0)),
            ((1, 1), (0, 0)),
        ),
        dtype=bool,
    )

    images = prepare_images(sections)

    assert images.shape == (2, 3, 2, 2)
    assert images.dtype == torch.uint8
    assert torch.equal(
        images[:, 0], torch.from_numpy(sections).to(torch.uint8).mul(255)
    )
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
    monkeypatch.setattr(fid_module, "FrechetInceptionDistance", FakeMetric)
    assert compute_fid(real, generated, "cpu", 64) == pytest.approx(4.5)

    assert len(created) == 1
    metric = created[0]
    assert metric.kwargs["feature"] == 64
    assert [real_flag for _, real_flag in metric.updates] == [True, False]


def test_prepare_images_preserves_input_and_scales_normalized_grayscale() -> None:
    sections = np.asarray((((0.0, 0.5), (1.0, 0.25)),), dtype=np.float32)
    original = sections.copy()

    images = prepare_images(sections)

    assert np.array_equal(sections, original)
    assert torch.equal(
        images[0, 0],
        torch.tensor(((0, 128), (255, 64)), dtype=torch.uint8),
    )


def test_uint8_images_keep_their_intensity_scale_even_when_dark():
    images = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.uint8)
    assert torch.equal(prepare_images(images)[:, 0], images)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_float_images_are_rejected(value):
    with pytest.raises(ValueError, match="finite and in"):
        prepare_images(torch.full((1, 2, 2), value))
