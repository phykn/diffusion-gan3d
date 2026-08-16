from unittest.mock import patch

import pytest
import torch

from src.anchor import PlaneAnchor, build_anchors
from src.train import vf


def test_pool_preserves_each_axis_crop_as_an_empirical_target() -> None:
    batches = {
        0: torch.tensor([[[0, 0], [1, 1]], [[2, 2], [2, 2]]]),
        1: torch.tensor([[[1, 1], [1, 2]], [[0, 1], [2, 2]]]),
        2: torch.tensor([[[0, 0], [0, 0]], [[0, 1], [1, 2]]]),
    }
    expected = torch.tensor(
        (
            (0.5, 0.5, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.75, 0.25),
            (0.25, 0.25, 0.5),
            (1.0, 0.0, 0.0),
            (0.25, 0.5, 0.25),
        )
    )

    assert torch.equal(vf.build_pool(batches, num_phases=3), expected)


def test_target_resamples_for_anchor_minima_and_rejects_no_match() -> None:
    device = torch.device("cpu")
    condition = build_anchors(
        (PlaneAnchor(torch.ones(2, 2, dtype=torch.uint8), axis=0, index=0),),
        batch_size=1,
        num_phases=2,
        volume_size=2,
        device=device,
        dtype=torch.float32,
        reconcile=False,
    )
    assert condition is not None
    pool = torch.tensor(((0.75, 0.25), (0.5, 0.5)))

    with patch(
        "src.train.vf.torch.randint",
        side_effect=(torch.tensor(((0,),)), torch.tensor((0,))),
    ):
        target, resample_rate = vf.sample_target(
            pool,
            condition,
            batch_size=1,
            num_phases=2,
            device=device,
            max_samples=1,
        )

    assert torch.equal(target, pool[1:])
    assert resample_rate == 1.0

    with (
        patch("src.train.vf.torch.randint", return_value=torch.tensor(((0,),))),
        pytest.raises(ValueError, match="incompatible"),
    ):
        vf.sample_target(
            pool[:1],
            condition,
            batch_size=1,
            num_phases=2,
            device=device,
            max_samples=1,
        )


def test_target_averages_a_random_number_of_crops() -> None:
    pool = torch.tensor(
        (
            (1.0, 0.0),
            (0.5, 0.5),
            (0.0, 1.0),
            (0.25, 0.75),
        )
    )

    with patch(
        "src.train.vf.torch.randint",
        side_effect=(torch.tensor((3,)), torch.tensor(((0, 1, 2, 3),))),
    ):
        target, resample_rate = vf.sample_target(
            pool,
            None,
            batch_size=1,
            num_phases=2,
            device=torch.device("cpu"),
            max_samples=4,
        )

    assert torch.allclose(target, torch.tensor(((0.5, 0.5),)))
    assert resample_rate == 0.0
