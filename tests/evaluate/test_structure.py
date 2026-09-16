import torch

from src.evaluate.structure import structure_metrics


def test_structure_metrics_detect_disconnection_and_match_measured_planes():
    labels = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    labels[:, :, 3, 3] = 1
    probs = torch.nn.functional.one_hot(labels, 2).movedim(-1, 1).float()
    real = {0: probs[:, :, 2]}
    metrics = structure_metrics(probs, real, {"xy": (0,)})
    assert metrics["structure/phase_1/percolation_xy"] == 1
    assert metrics["structure/phase_1/percolation_lower_bound_xy"] == 1
    assert metrics["structure/phase_1/percolation_xz"] == 0
    assert metrics["structure/xy/two_point_mae"] == 0
    assert metrics["structure/xy/chord_tv"] == 0
    probs[:, :, 4] = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1)
    assert (
        structure_metrics(probs, real, {"xy": (0,)})["structure/phase_1/percolation_xy"]
        == 0
    )
