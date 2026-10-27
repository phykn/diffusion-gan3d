import pytest
import torch

from src.config.data import get_domains
from src.data.slice import sample_pairs
from src.plane import get_axis


def test_numeric_config_planes_are_rejected():
    with pytest.raises(ValueError, match="plane names"):
        get_domains({"domains": {0: {0: ["data"]}}})


@pytest.mark.parametrize("plane", ["yx", "depth", "0", 3, -1, True, 1.0])
def test_invalid_plane_names_are_rejected(plane):
    with pytest.raises(ValueError, match="plane names"):
        get_domains({"domains": {0: {plane: ["data"]}}})


def test_numeric_and_named_aliases_cannot_duplicate_a_plane():
    with pytest.raises(ValueError, match="plane names"):
        get_domains({"domains": {0: {"xy": ["one"], 0: ["two"]}}})


@pytest.mark.parametrize("plane,axis", [("xy", 0), ("xz", 1), ("yz", 2)])
def test_named_slices_have_the_correct_normal_and_in_plane_directions(plane, axis):
    vol = torch.arange(3 * 4 * 5).reshape(1, 1, 3, 4, 5)
    expected = vol[0].unbind(axis + 1)
    slices, _ = sample_pairs(vol, vol, get_axis(plane), 6, expected[0].shape[-2:])
    for image in slices:
        assert any(torch.equal(image, section) for section in expected)


@pytest.mark.parametrize("domain", [False, 0.0])
def test_domain_ids_require_integers(domain):
    with pytest.raises(ValueError, match="domain IDs"):
        get_domains({"domains": {domain: {"xy": ["images"]}}})
