import numpy as np
import pytest
import torch
from PIL import Image

from src.build.trainer import build_trainer
from src.config import get_domains, load_train_config
from src.plane import PLANES, get_axis
from src.storage import save_model
from src.train.sr_loss import sample_slices


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
    slices = sample_slices(vol, get_axis(plane), 6)
    expected = vol[0].unbind(axis + 1)
    for image in slices:
        assert any(torch.equal(image, section) for section in expected)


def test_initial_weights_load_by_plane(tmp_path):
    torch.set_num_threads(1)
    images = tmp_path / "images"
    images.mkdir()
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(images / "sample.png")
    cfg = load_train_config("config/train/low_res.yaml")
    cfg["data"].update(
        domains={0: {plane: [str(images)] for plane in PLANES}},
        crop_size=8,
        lo_res_size=8,
    )
    cfg["model"]["generator"].update(
        channels=[4, 8], embedding_channels=8, latent_channels=4
    )
    cfg["model"]["critic"]["channels"] = [4, 8]
    cfg["model"]["gradient_checkpointing"] = False
    cfg["train"].update(mixed_precision=False, real_batch_size=1, num_workers=0)
    source = build_trainer(cfg, torch.device("cpu"))
    assert tuple(source.critics) == PLANES
    assert tuple(source.critic_optims) == PLANES
    weights = tmp_path / "weights"
    save_model(weights / "generator.pt", source.denoiser)
    save_model(weights / "critic_c.pt", source.connectivity_critic)
    for axis, plane in enumerate(PLANES):
        with torch.no_grad():
            source.critics[plane].input.bias.fill_(axis + 1)
        name = plane
        save_model(weights / f"critic_{name}.pt", source.critics[plane])
    cfg["train"]["initial_weights"] = str(weights)
    restored = build_trainer(cfg, torch.device("cpu"))
    for plane in PLANES:
        for key, value in source.critics[plane].state_dict().items():
            torch.testing.assert_close(
                restored.critics[plane].state_dict()[key], value, rtol=0, atol=0
            )
    for key, value in source.denoiser.state_dict().items():
        torch.testing.assert_close(
            restored.denoiser.state_dict()[key], value, rtol=0, atol=0
        )
