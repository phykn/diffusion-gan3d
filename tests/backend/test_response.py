import pytest
import torch

from app.backend.src.response import volume_response


@pytest.mark.parametrize("strided", [False, True])
def test_raw_response_serializes_labels_in_c_order(strided):
    volume = torch.arange(48, dtype=torch.uint8).reshape(4, 3, 4)
    if strided:
        volume = volume.transpose(0, 1)
    response = volume_response(volume, "cpu")
    assert response.body == bytes(volume.flatten().tolist())
    assert response.headers["x-volume-shape"] == ",".join(map(str, volume.shape))
