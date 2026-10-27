import torch

from src.evaluate.structure import label_sections, section_errors, structure_metrics


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
    assert metrics["structure/xy/cluster_error"] == 0
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


def test_cluster_error_sees_merged_regions_of_equal_area():
    separate = torch.zeros(1, 8, 8, dtype=torch.long)
    for y, x in ((0, 0), (0, 5), (5, 0), (5, 5)):
        separate[:, y : y + 2, x : x + 2] = 1
    merged = torch.zeros(1, 8, 8, dtype=torch.long)
    merged[:, 2:6, 2:6] = 1
    real, fake = (
        torch.nn.functional.one_hot(labels, 2).movedim(-1, 1).float()
        for labels in (separate, merged)
    )
    assert section_errors(real, real)[2] == 0
    # Phase 1: 4 vs 1 clusters and area 4 vs 16, each a gap of 1.2 relative to
    # the mean; phase 0 is one cluster of the same area on both sides.
    assert abs(section_errors(real, fake)[2] - 0.6) < 1e-9


def test_labelled_sections_split_exact_ties_evenly_and_keep_clear_phases():
    sections = torch.full((64, 2, 16, 16), 0.5)
    sections[:, :, :, :8] = torch.tensor([0.25, 0.75])[:, None, None]
    labels = label_sections(sections)
    assert torch.equal(labels[..., :8].argmax(1), torch.ones(64, 16, 8).long())
    assert abs(float(labels[..., 8:, 1].mean()) - 0.5) < 0.02
    assert torch.equal(labels, label_sections(sections))
