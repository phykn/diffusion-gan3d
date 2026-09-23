from unittest.mock import patch

import pytest
import torch

from src.model.denoiser import Denoiser3D
from src.model.diffusion import Diffusion
from src.predict.generator import Generator


@pytest.mark.parametrize("guidance", (0.0, 1.0, 1.5))
@pytest.mark.parametrize("strength", (0.0, 0.25, 0.8, 1.0))
@pytest.mark.parametrize("condition", (None, "vf", "profile"))
def test_strength_interpolates_anchor_logits_and_preserves_other_conditions(
    guidance, strength, condition
):
    model = Denoiser3D(
        num_phases=3,
        base_channels=4,
        channel_multipliers=(1, 2),
        embedding_channels=8,
        latent_channels=4,
        num_domains=1,
    ).eval()
    generator = Generator(model, Diffusion(2), torch.device("cpu"), 4, 3, 4, False)
    current = torch.zeros(1, 3, 4, 4, 4)
    time = torch.zeros(1, dtype=torch.long)
    latent = torch.ones(1, 4)
    domain = torch.zeros(1, dtype=torch.long)
    height = torch.zeros(1, 1, 4, 4, 4)
    mask = torch.zeros_like(height, dtype=torch.bool)
    mask[:, :, 1] = True
    image = torch.ones_like(current)
    vf = torch.tensor([[0.2, 0.3, 0.5]]) if condition is not None else None
    profile = vf[:, :, None].expand(-1, -1, 4) if condition == "profile" else None
    plain = torch.tensor([0.2, -0.4, 0.1]).reshape(1, 3, 1, 1, 1)
    vf_delta = torch.tensor([-0.1, 0.7, 0.4]).reshape(1, 3, 1, 1, 1)
    anchor_delta = torch.tensor([1.0, -0.5, 0.3]).reshape(1, 3, 1, 1, 1)

    def logits(x, t, z, domain, **conditions):
        assert x is current and t is time and z is latent
        assert conditions.get("height") is height
        result = plain.clone()
        if conditions.get("vf") is not None:
            assert conditions["vf"] is vf
            assert conditions.get("profile") is profile
            result = result + vf_delta
        if conditions.get("anchor_mask") is not None:
            assert conditions["anchor_mask"] is mask
            assert conditions["anchor_image"] is image
            result = result + anchor_delta
        return result.expand_as(current)

    with patch.object(model, "compute_logits", side_effect=logits) as forward:
        prediction = generator.predict(
            current,
            time,
            latent,
            domain,
            guidance=guidance,
            vf=vf,
            profile=profile,
            height=height,
            anchor_image=image,
            anchor_mask=mask,
            anchor_strength=strength,
        )

    residual = strength * anchor_delta
    if vf is not None:
        residual = residual + vf_delta
    expected = Denoiser3D.decode(plain + guidance * residual).expand_as(current)
    torch.testing.assert_close(prediction, expected)
    assert prediction.dtype == current.dtype
    expected_calls = 1
    if guidance != 0.0:
        if 0.0 < strength < 1.0:
            expected_calls = 2 + int(guidance != 1.0 and vf is not None)
        elif guidance != 1.0 and (strength > 0.0 or vf is not None):
            expected_calls = 2
    assert forward.call_count == expected_calls
