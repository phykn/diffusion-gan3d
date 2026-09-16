import torch

from src.prepare.height import height_field


def test_height_field_uses_cell_centers_and_physical_pixel_scale():
    height = height_field((4, 4, 4), 1, [8, 16], 2, 32)
    torch.testing.assert_close(
        height[:, 0, 0, :, 0],
        torch.tensor(
            [[-0.4375, -0.3125, -0.1875, -0.0625], [0.0625, 0.1875, 0.3125, 0.4375]]
        ),
    )
