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


def test_fid_batches_all_samples_without_changing_score():
    class Features(torch.nn.Module):
        num_features = 2

        def __init__(self, limit):
            super().__init__()
            self.limit = limit
            self.seen = 0

        def forward(self, images):
            assert len(images) <= self.limit
            assert not torch.is_grad_enabled()
            self.seen += len(images)
            values = images.double().flatten(1)
            return torch.stack((values.mean(1), values.square().mean(1)), 1)

    rng = np.random.default_rng(3)
    real = rng.integers(0, 256, (17, 4, 4), dtype=np.uint8)
    generated = rng.integers(0, 256, (23, 4, 4), dtype=np.uint8)
    reference = Features(23)
    expected = compute_fid(real, generated, "cpu", reference, batch_size=23)
    bounded = Features(4)
    actual = compute_fid(real, generated, "cpu", bounded, batch_size=4)
    assert bounded.seen == reference.seen == 40
    assert actual == pytest.approx(expected, rel=1e-9, abs=1e-8)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_fid_rejects_invalid_batch_size_before_loading_model(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        compute_fid(
            torch.zeros(2, 4, 4), torch.zeros(2, 4, 4), "cpu", batch_size=batch_size
        )


def test_fid_requires_enough_images_before_loading_model():
    with pytest.raises(ValueError, match="at least two"):
        compute_fid(torch.zeros(1, 4, 4), torch.zeros(2, 4, 4), "cpu")
