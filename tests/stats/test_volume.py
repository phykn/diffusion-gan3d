import torch

from src.stats.volume import interface_density


def test_interface_density_averages_given_axes():
    labels = torch.tensor([[0, 1], [0, 1]])

    assert interface_density(labels, (0,)) == 0.0
    assert interface_density(labels, (1,)) == 1.0
    assert interface_density(labels, (0, 1)) == 0.5
