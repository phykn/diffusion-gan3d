import torch

from src.anchor import PlaneAnchor, encode_anchors
from src.evaluate.anchor import anchor_boundary_metrics


def test_boundary_metric_detects_a_detached_anchor_plane():
    condition = encode_anchors(
        (PlaneAnchor(torch.zeros(6, 6, dtype=torch.long), 0, 3),),
        1,
        2,
        6,
        torch.device("cpu"),
        torch.float32,
    )
    reference = torch.ones(1, 2, 6, 6, 6)
    reference[:, 1] = -1
    prediction = reference.clone()
    prediction[:, :, 2] *= -1
    values = anchor_boundary_metrics(prediction, condition)
    assert values["anchor/neighbor_agreement"] == 0.5
    assert values["anchor/neighbor_excess_jump"] == 0.5
