import pytest
import torch

from src.prepare.height import height_field, resolve_extent
from src.prepare.profile import (
    image_profile,
    rebin_profile,
    sample_profile,
    validate_profile,
)
from src.train.loss.spatial_profile import compute_profile_loss


def spec(mode="linear"):
    return {
        "axis": "z",
        "points": [[0, [0.1, 0.9]], [1, [0.5, 0.5]]],
        "interpolation": mode,
    }


def test_source_percent_and_batch_extents():
    field = height_field((2, 2, 2), 0, [250, 500], 5, [1000, 2000])
    assert field[0, 0, 0, 0, 0] == pytest.approx(2 * 252.5 / 1000 - 1)
    assert field[1, 0, 0, 0, 0] == pytest.approx(2 * 502.5 / 2000 - 1)
    with pytest.raises(ValueError, match="height_extent is required"):
        resolve_extent({"height_extents": {0: None}}, 0)


def test_profile_integrates_output_interval_and_context():
    values = sample_profile(spec(), 2, 64, 256, 2, 1024)
    assert values[0, 0].mean() == pytest.approx(0.225)
    full = sample_profile(spec(), 2, 10, -2, 2, 12)
    torch.testing.assert_close(full[..., 2:6], sample_profile(spec(), 2, 4, 2, 2, 12))
    assert full[0, 0, 0] == pytest.approx(0.1)
    assert full[0, 0, -1] == pytest.approx(0.5)


def test_step_boundary_is_integrated_not_point_sampled():
    profile = {
        "axis": "z",
        "points": [[0, [1, 0]], [0.3, [0, 1]], [1, [0, 1]]],
        "interpolation": "constant",
    }
    values = sample_profile(profile, 2, 2, 0, 5, 10)
    torch.testing.assert_close(values, torch.tensor([[[0.6, 0], [0.4, 1]]]))


@pytest.mark.parametrize(
    "points",
    [
        [[0.1, [0.5, 0.5]], [1, [0.5, 0.5]]],
        [[0, [0.5, 0.5]], [0, [0.5, 0.5]], [1, [0.5, 0.5]]],
        [[0, [0.2, 0.2]], [1, [0.5, 0.5]]],
        [[0, [float("nan"), 0]], [1, [0.5, 0.5]]],
    ],
)
def test_reject_invalid_profile(points):
    with pytest.raises(ValueError):
        validate_profile(spec() | {"points": points}, 2)


def test_rebin_preserves_mean_for_fractional_bins():
    values = torch.tensor([[[0.1, 0.2, 0.9], [0.9, 0.8, 0.1]]])
    torch.testing.assert_close(rebin_profile(values, 5).mean(-1), values.mean(-1))


@pytest.mark.parametrize("size", [2, 5, 11])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA required"
            ),
        ),
    ],
)
def test_rebin_matches_overlap_integral_and_gradients(size, device):
    values = torch.rand(2, 3, 7, device=device, requires_grad=True)
    edges = torch.linspace(0, 7, size + 1, device=device)
    left = torch.arange(7, device=device)
    weights = (
        torch.minimum(edges[1:, None], left[None] + 1)
        - torch.maximum(edges[:-1, None], left[None])
    ).clamp_min(0) / (7 / size)
    expected = values @ weights.T
    actual = rebin_profile(values, size)
    torch.testing.assert_close(actual, expected)
    gradient = torch.rand_like(actual)
    actual_grad = torch.autograd.grad(actual, values, gradient, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, values, gradient)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_rebin_rejects_empty_profile():
    with pytest.raises(ValueError, match="non-empty"):
        rebin_profile(torch.empty(1, 2, 0), 4)


def test_image_profile_rejects_unknown_direction():
    with pytest.raises(ValueError, match="direction"):
        image_profile(torch.ones(1, 2, 3, 4), direction=2)


def test_local_loss_distinguishes_equal_global_vf_and_masks_gradients():
    probs = torch.full((2, 2, 2, 3, 3), 0.5, requires_grad=True)
    target = torch.tensor([[[0.1, 0.9], [0.9, 0.1]]]).expand(2, -1, -1)
    torch.testing.assert_close(probs.mean((2, 3, 4)), target.mean(-1))
    loss = compute_profile_loss(probs, target, torch.tensor([True, False]), 2)
    assert loss.item() > 0
    loss.backward()
    assert probs.grad[0].abs().sum() > 0
    assert probs.grad[1].abs().sum() == 0


def test_label_profile_does_not_claim_soft_fraction_is_label_fraction():
    from src.evaluate.profile import phase_profile

    probs = torch.tensor([0.7, 0.3]).reshape(1, 2, 1, 1, 1).expand(1, 2, 4, 3, 3)
    soft = phase_profile(probs, 2)
    labels = phase_profile(probs.argmax(1), 2)
    torch.testing.assert_close(soft[0, :, 0], torch.tensor([0.7, 0.3]))
    torch.testing.assert_close(labels[0, :, 0], torch.tensor([1.0, 0.0]))


def test_gradient_weight_is_independent_of_profile_weight():
    probs = torch.full((1, 2, 2, 2, 2), 0.5, requires_grad=True)
    target = torch.tensor([[[0.1, 0.9], [0.9, 0.1]]])
    loss = compute_profile_loss(
        probs, target, torch.tensor([True]), 2, gradient_weight=1, profile_weight=0
    )
    assert loss > 0
    loss.backward()
    assert probs.grad.abs().sum() > 0
