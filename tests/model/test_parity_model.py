"""v2 models against values v1 computed at commit 0608ab8 (fixtures/parity)."""

from pathlib import Path

import pytest
import torch

from src.model.critic import PairCritic2D
from src.model.denoiser import Denoiser3D
from src.model.diffusion import Diffusion

FIXTURE = Path(__file__).parents[1] / "fixtures" / "parity" / "model.pt"
TOLERANCE = dict(rtol=1e-5, atol=1e-5)


@pytest.fixture(scope="module")
def reference():
    return torch.load(FIXTURE, weights_only=False)


def _denoiser(case) -> Denoiser3D:
    model = Denoiser3D(**case["kwargs"]).eval()
    model.load_state_dict(case["state"], strict=True)
    return model


@pytest.mark.parametrize("name", ["lr_plain", "lr_conditioned", "sr"])
def test_denoiser_body_matches_v1_for_the_same_embedding(reference, name):
    case = reference[name]
    model = _denoiser(case)
    inputs = case["inputs"]
    with torch.no_grad():
        logits = model.forward_body(inputs["x"], inputs["emb"], **case["conditions"])
    torch.testing.assert_close(logits, case["outputs"]["logits"], **TOLERANCE)


@pytest.mark.parametrize("name", ["lr_plain", "lr_conditioned"])
def test_embedding_is_v1_without_the_domain_term(reference, name):
    # v1 with the null domain gave (time + latent) / sqrt(3) + vf; v2 divides
    # the same sum by sqrt(2) and adds the same vf term.
    case = reference[name]
    model = _denoiser(case)
    inputs = case["inputs"]
    with torch.no_grad():
        base = model.embed(inputs["time"], inputs["latent"])
        with_vf = model.embed(
            inputs["time"], inputs["latent"], inputs["vf"], inputs["vf_present"]
        )
    outputs = case["outputs"]
    torch.testing.assert_close(base * 2**0.5, outputs["embed_scaled"], **TOLERANCE)
    torch.testing.assert_close(with_vf - base, outputs["embed_vf"], **TOLERANCE)


def test_guidance_blend_matches_v1(reference):
    case = reference["guidance"]
    model = _denoiser(case)
    embeddings = case["embeddings"]
    model.embed = lambda time, latent, vf=None, vf_present=None: (
        embeddings["unconditional"] if vf is None else embeddings["conditional"]
    )
    inputs = case["inputs"]
    for guidance, expected in case["guided"].items():
        with torch.no_grad():
            logits = model.guided_logits(
                inputs["x"],
                inputs["time"],
                inputs["latent"],
                float(guidance),
                vf=inputs["vf"],
                vf_present=inputs["vf_present"],
                **case["conditions"],
            )
        torch.testing.assert_close(logits, expected, **TOLERANCE)


@pytest.mark.parametrize("name", ["critic", "critic_height"])
def test_critic_body_matches_v1_for_the_same_embedding(reference, name):
    case = reference[name]
    critic = PairCritic2D(**case["kwargs"]).eval()
    critic.load_state_dict(case["state"], strict=True)
    inputs = case["inputs"]
    with torch.no_grad():
        scores = critic.score(inputs["x"], inputs["emb"], inputs["height"])
        time_emb = critic.embed(inputs["time"], inputs["x"])
    levels = [level.logits_global for level in scores.levels] or [scores.logits_global]
    expected = case["outputs"]["levels"]
    assert len(levels) == len(expected) == 3
    for actual, wanted in zip(levels, expected, strict=True):
        torch.testing.assert_close(actual, wanted, **TOLERANCE)
    torch.testing.assert_close(time_emb, case["outputs"]["time_embedding"], **TOLERANCE)


def test_diffusion_matches_v1(reference):
    case = reference["diffusion"]
    inputs, outputs = case["inputs"], case["outputs"]
    diffusion = Diffusion(10, 0.1, 20.0)
    previous, current = diffusion.sample_pair(
        inputs["clean"],
        inputs["transitions"],
        inputs["previous_noise"],
        inputs["step_noise"],
    )
    posterior = diffusion.sample_posterior(
        current, inputs["pred"], inputs["transitions"], inputs["posterior_noise"]
    )
    noisy = diffusion.add_noise(
        inputs["clean"], torch.tensor([0, 1, 5, 10]), inputs["previous_noise"]
    )
    torch.testing.assert_close(previous, outputs["previous"], **TOLERANCE)
    torch.testing.assert_close(current, outputs["current"], **TOLERANCE)
    torch.testing.assert_close(posterior, outputs["posterior"], **TOLERANCE)
    torch.testing.assert_close(noisy, outputs["add_noise"], **TOLERANCE)
    for name, values in outputs["buffers"].items():
        torch.testing.assert_close(getattr(diffusion, name), values, **TOLERANCE)
