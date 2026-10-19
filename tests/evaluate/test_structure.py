import torch

from src.evaluate.structure import structure_metrics


def test_structure_metrics_detect_disconnection_and_match_measured_planes():
    labels = torch.zeros(1, 8, 8, 8, dtype=torch.long)
    labels[:, :, 3, 3] = 1
    probs = torch.nn.functional.one_hot(labels, 2).movedim(-1, 1).float()
    real = {0: probs[:, :, 2]}
    metrics = structure_metrics(probs, real, {"xy": (0,)})
    assert metrics["structure/phase_1/percolation_z"] == 1
    assert metrics["structure/phase_1/percolation_lower_bound_z"] == 1
    assert metrics["structure/phase_1/percolation_y"] == 0
    assert metrics["structure/xy/two_point_mae"] == 0
    assert metrics["structure/xy/chord_tv"] == 0
    probs[:, :, 4] = torch.tensor([1.0, 0.0]).view(1, 2, 1, 1)
    assert (
        structure_metrics(probs, real, {"xy": (0,)})["structure/phase_1/percolation_z"]
        == 0
    )


def test_structure_metrics_report_particle_tilt_balance():
    z, y, x = torch.meshgrid(*(torch.arange(16) - 8,) * 3, indexing="ij")
    labels = ((z - 0.3 * y).abs() <= 1) & (y**2 + x**2 <= 36)
    probs = torch.nn.functional.one_hot(labels[None].long(), 2).movedim(-1, 1).float()
    metrics = structure_metrics(probs, {}, {})
    assert metrics["structure/phase_1/tilt_balance_zy"] == 1
    assert metrics["structure/phase_1/tilt_balance_zx"] == 0
    assert not any(key.startswith("structure/phase_0/tilt") for key in metrics)
