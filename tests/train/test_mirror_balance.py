import pytest
import torch

from src.config.sample import build_train_config, load_sample
from src.config.train import normalize_train_config
from src.prepare.resize import phase_channels
from src.train.loss.mirror_balance import (
    MIRROR_PLANES,
    SLOPES,
    compute_mirror_balance_loss,
)


def sheared(size=16):
    """Stripes f(x - z): every xy and yz section is a shifted copy of one row."""
    z, y, x = torch.meshgrid(*(torch.arange(size),) * 3, indexing="ij")
    return phase_channels(((x - z) // 3 % 2)[None], 2)


def pair_table(probs, offset):
    """Mean of p(x) q(x + o) and q(x) p(x + o) over the x with x + o inside."""
    left, right = probs, probs
    for axis, step in enumerate(offset, start=2):
        length = probs.shape[axis] - abs(step)
        left = left.narrow(axis, max(0, -step), length)
        right = right.narrow(axis, max(0, step), length)
    table = torch.einsum("bp...,bq...->bpq", left, right) / left[0, 0].numel()
    return (table + table.transpose(1, 2)) / 2


def test_loss_matches_pair_tables_from_shifted_windows():
    torch.manual_seed(0)
    probs = torch.randn(2, 3, 10, 9, 8).softmax(1).requires_grad_(True)
    loss, logs = compute_mirror_balance_loss(probs, 2, 0.01)
    expected = []
    for name, (first, second) in MIRROR_PLANES.items():
        errors = []
        for gap in (1, 2):
            for slope_a, slope_b in SLOPES:
                plus, minus = [0, 0, 0], [0, 0, 0]
                plus[first] = minus[first] = gap * slope_a
                plus[second], minus[second] = gap * slope_b, -gap * slope_b
                difference = pair_table(probs, plus) - pair_table(probs, minus)
                errors.append(difference.abs().sum((1, 2)) / 2)
        errors = torch.stack(errors, -1)
        logged = logs[f"mirror_balance/{name}/difference"]
        torch.testing.assert_close(logged, errors.mean().detach())
        logged = logs[f"mirror_balance/{name}/max"]
        torch.testing.assert_close(logged, errors.amax(-1).mean().detach())
        expected.append((errors - 0.01).relu().mean())
    expected = torch.stack(expected).mean()
    torch.testing.assert_close(loss, expected)
    gradient, expected_gradient = (
        torch.autograd.grad(value, probs)[0] for value in (loss, expected)
    )
    torch.testing.assert_close(gradient, expected_gradient)


def test_shear_is_penalized_only_in_its_plane_and_both_tilts_alike():
    loss, logs = compute_mirror_balance_loss(sheared(), 3, 0)
    assert loss > 0
    assert logs["mirror_balance/zx/difference"] > 0.1
    assert logs["mirror_balance/zy/difference"] < 1e-6
    assert logs["mirror_balance/yx/difference"] < 1e-6
    flipped, _ = compute_mirror_balance_loss(sheared().flip(-1), 3, 0)
    torch.testing.assert_close(flipped, loss)


def test_mirror_symmetric_volumes_are_free():
    labels = torch.randint(2, (1, 12, 12, 12))
    symmetric = torch.cat((labels, labels.flip(-1)), -1)
    symmetric = torch.cat((symmetric, symmetric.flip(-2)), -2)
    assert compute_mirror_balance_loss(phase_channels(symmetric, 2), 4, 0)[0] < 1e-6


def test_tolerance_hides_small_differences():
    probs = sheared()
    loss, logs = compute_mirror_balance_loss(probs, 3, 0)
    assert compute_mirror_balance_loss(probs, 3, 1)[0] == 0
    assert logs["mirror_balance/zx/max"] <= 1


def test_descending_the_loss_undoes_a_shear():
    logits = (sheared() * 4 - 2).requires_grad_(True)
    optimizer = torch.optim.Adam([logits], lr=0.1)
    before = compute_mirror_balance_loss(logits.softmax(1), 3, 0)[0].item()
    for _ in range(50):
        optimizer.zero_grad()
        compute_mirror_balance_loss(logits.softmax(1), 3, 0)[0].backward()
        optimizer.step()
    assert compute_mirror_balance_loss(logits.softmax(1), 3, 0)[0] < 0.5 * before


def test_only_mirror_symmetric_samples_enable_it():
    default = load_sample("default")
    low = build_train_config(default, "low_res")
    assert low["loss"]["mirror_balance"] == {"weight": 10.0}
    sr = normalize_train_config(build_train_config(default, "sr"), "sr")
    assert sr["loss"]["mirror_balance"]["weight"] == 0
    anode = load_sample("simul_anode") | {"num_phases": 3}
    for stage in ("low_res", "sr"):
        cfg = normalize_train_config(build_train_config(anode, stage), stage)
        assert cfg["loss"]["mirror_balance"]["weight"] == 0


@pytest.mark.parametrize(
    "key,value",
    [("weight", -1), ("weight", True), ("tolerance", 1.1), ("max_gap", 0)],
)
def test_settings_are_validated(key, value):
    with pytest.raises(ValueError, match=f"mirror_balance.{key}"):
        normalize_train_config({"loss": {"mirror_balance": {key: value}}})
