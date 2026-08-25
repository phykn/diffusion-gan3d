import torch

from src.loss import vf


def test_compute_vf_aggregates_all_images_and_axes() -> None:
    batches = {
        0: torch.tensor([[[0, 0], [1, 1]], [[2, 2], [2, 2]]]),
        1: torch.tensor([[[1, 1], [1, 2]], [[0, 1], [2, 2]]]),
        2: torch.tensor([[[0, 0], [0, 0]], [[0, 1], [1, 2]]]),
    }
    expected = torch.tensor((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))

    torch.testing.assert_close(vf.compute_vf(batches, num_phases=3), expected)


def test_compute_vf_loss_uses_only_present_samples() -> None:
    probs = torch.empty(2, 2, 1, 2, 2)
    probs[0, 0] = 0.5
    probs[0, 1] = 0.5
    probs[1, 0] = 1.0
    probs[1, 1] = 0.0
    target = torch.tensor(((0.25, 0.75), (0.0, 1.0)))
    present = torch.tensor((True, False))
    expected = (
        target[0]
        * (target[0].log() - torch.tensor((0.5, 0.5)).log())
    ).sum()

    torch.testing.assert_close(
        vf.compute_vf_loss(probs, target, present),
        expected,
    )
    torch.testing.assert_close(
        vf.compute_vf_loss(probs, target, torch.zeros(2, dtype=torch.bool)),
        torch.tensor(0.0),
    )
