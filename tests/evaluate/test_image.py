import numpy as np
import pytest
import torch

from src.evaluate import (
    fid_score,
    kid_score,
    make_fid_metric,
    make_kid_metric,
    metric_images,
)
from src.evaluate import image as image_module


def test_metric_images_expands_binary_sections_to_uint8_rgb() -> None:
    sections = np.asarray(
        (
            ((0, 1), (1, 0)),
            ((1, 1), (0, 0)),
        ),
        dtype=np.uint8,
    )

    images = metric_images(sections)

    assert images.shape == (2, 3, 2, 2)
    assert images.dtype == torch.uint8
    assert torch.equal(images[:, 0], torch.from_numpy(sections).mul(255))
    assert torch.equal(images[:, 0], images[:, 1])
    assert torch.equal(images[:, 1], images[:, 2])


@pytest.mark.parametrize("kind", ("kid", "fid"))
def test_distribution_metrics_reuse_real_features_and_reset_fake(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = []

    class FakeMetric:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.updates = []
            self.reset_count = 0
            created.append(self)

        def to(self, device):
            self.device = torch.device(device)
            return self

        def update(self, images, *, real):
            self.updates.append((images.clone(), real))

        def compute(self):
            if kind == "kid":
                return torch.tensor(0.125), torch.tensor(0.025)
            return torch.tensor(4.5)

        def reset(self):
            self.reset_count += 1

    real = np.zeros((2, 4, 4), dtype=np.uint8)
    generated = np.ones((2, 4, 4), dtype=np.uint8)
    if kind == "kid":
        monkeypatch.setattr(image_module, "KernelInceptionDistance", FakeMetric)
        metric = make_kid_metric(
            real,
            "cpu",
            feature=64,
            subsets=3,
            subset_size=2,
        )
        assert kid_score(metric, generated, "cpu") == pytest.approx((0.125, 0.025))
    else:
        monkeypatch.setattr(image_module, "FrechetInceptionDistance", FakeMetric)
        metric = make_fid_metric(real, "cpu", feature=64)
        assert fid_score(metric, generated, "cpu") == pytest.approx(4.5)

    assert created == [metric]
    assert metric.kwargs["feature"] == 64
    assert metric.kwargs["reset_real_features"] is False
    assert [real_flag for _, real_flag in metric.updates] == [True, False]
    assert metric.reset_count == 1


@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available.",
            ),
        ),
    ),
)
def test_kid_score_preserves_global_rng_state(device: str) -> None:
    class FakeKid:
        def update(self, _images, *, real: bool) -> None:
            assert not real

        def compute(self):
            torch.rand((), device=device)
            return torch.tensor(0.0), torch.tensor(0.0)

        def reset(self) -> None:
            pass

    torch.manual_seed(1234)
    cpu_before = torch.random.get_rng_state()
    cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []

    kid_score(FakeKid(), np.zeros((2, 4, 4), dtype=np.uint8), device, seed=0)

    assert torch.equal(torch.random.get_rng_state(), cpu_before)
    assert all(
        torch.equal(after, before)
        for after, before in zip(
            torch.cuda.get_rng_state_all(),
            cuda_before,
            strict=True,
        )
    )
