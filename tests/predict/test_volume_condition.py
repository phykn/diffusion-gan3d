import pytest
import torch

from src.anchor import PlaneAnchor
from src.predict.volume_condition import prepare_base, validate_base_anchors


@pytest.mark.parametrize("axis", (0, 1, 2))
@pytest.mark.parametrize(
    "kind", ("labels", "batched_labels", "fractions", "batched_fractions")
)
@pytest.mark.parametrize("full_plane", (False, True))
def test_preserved_base_accepts_matching_anchor_formats(axis, kind, full_plane):
    volume = torch.arange(6 * 7 * 8).reshape(6, 7, 8).remainder(3).to(torch.uint8)
    labels = volume[1:3, 2:5, 3:7]
    base = prepare_base(labels, 3, (6, 7, 8), 1, (1, 2, 3), 0, True)
    index = (1, 2, 3)[axis] + 1
    image = volume.select(axis, index) if full_plane else labels.select(axis, 1)
    if "fractions" in kind:
        image = torch.nn.functional.one_hot(image.long(), 3).permute(2, 0, 1).float()
    if kind.startswith("batched"):
        image = image.unsqueeze(0)
    position = (
        (0, 0) if full_plane else tuple(p for a, p in enumerate((1, 2, 3)) if a != axis)
    )
    anchor = PlaneAnchor(image, axis, index, position)

    assert validate_base_anchors(base, (anchor,), (6, 7, 8), 1) is base

    conflicting = torch.zeros_like(image)
    if "fractions" in kind:
        conflicting[..., 0, :, :] = 1
    with pytest.raises(ValueError, match="conflicts"):
        validate_base_anchors(
            base,
            (PlaneAnchor(conflicting, axis, anchor.index, position),),
            (6, 7, 8),
            1,
        )


def test_base_blending_depends_on_placement_not_on_an_explicit_offset():
    base = torch.zeros((8, 8, 8), dtype=torch.uint8)
    centered = prepare_base(base, 3, (12, 8, 8), 2, None, 2, False)
    explicit = prepare_base(base, 3, (12, 8, 8), 2, (2, 0, 0), 2, False)
    torch.testing.assert_close(centered.weight, explicit.weight)
    assert torch.all(centered.weight[0, 0, 4] == 1.0)
    edges = centered.weight[0, 0, :, 4, 4]
    assert edges[0] < 1 and edges[-1] < 1
    at_start = prepare_base(base, 3, (12, 8, 8), 2, (0, 0, 0), 2, False)
    assert at_start.weight[0, 0, 0, 4, 4] == 1 and at_start.weight[0, 0, -1, 4, 4] < 1
