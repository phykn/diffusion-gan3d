import numpy as np
import pytest
import torch

from src.evaluate.kid import compute_kid


class Features(torch.nn.Module):
    def forward(self, images):
        value = images.double().mean(dim=(1, 2, 3)) / 255
        return torch.stack((value, value.square()), dim=1)


def test_kid_is_the_unbiased_u_statistic_and_can_be_negative():
    images = np.asarray([np.full((4, 4), value) for value in (0.0, 0.3, 0.7, 1.0)])
    score = compute_kid(
        images, images, "cpu", Features(), subsets=3, subset_size=4, seed=7
    )
    from src.evaluate.image import prepare_images

    features = Features()(prepare_images(images))
    kernel = (features @ features.T / features.shape[1] + 1) ** 3
    expected = 2 * (kernel.sum() - kernel.diag().sum()) / (4 * 3) - 2 * kernel.mean()
    assert score.mean == pytest.approx(expected.item())
    assert score.mean < 0
    assert score.std == pytest.approx(0, abs=1e-12)


def test_kid_64_images_uses_bounded_subsets_and_preserves_seeded_caller_rng():
    images = torch.rand(64, 1, 4, 4)
    original = images.clone()
    state = torch.random.get_rng_state().clone()
    first = compute_kid(images, images.flip(0), "cpu", Features(), subsets=8, seed=42)
    assert torch.equal(state, torch.random.get_rng_state())
    second = compute_kid(images, images.flip(0), "cpu", Features(), subsets=8, seed=42)
    assert first == second
    assert first.subset_size == 50
    assert np.isfinite(first.mean) and np.isfinite(first.std)
    assert first.std > 0
    assert torch.equal(images, original)


@pytest.mark.parametrize(
    "count,options",
    [(1, {}), (4, {"subset_size": 5}), (4, {"subset_size": 1}), (4, {"subsets": 0})],
)
def test_kid_rejects_invalid_sample_sizes_before_loading_inception(count, options):
    with pytest.raises(ValueError):
        compute_kid(np.zeros((count, 4, 4)), np.zeros((count, 4, 4)), "cpu", **options)


def test_kid_intensity_scale_does_not_depend_on_batch_partition():
    images = torch.arange(4, dtype=torch.uint8).view(4, 1, 1).expand(4, 4, 4)
    options = {"feature": Features(), "subset_size": 4, "subsets": 2, "seed": 3}
    single = compute_kid(images, images.flip(0), "cpu", batch_size=1, **options)
    combined = compute_kid(images, images.flip(0), "cpu", batch_size=4, **options)
    assert single == combined
