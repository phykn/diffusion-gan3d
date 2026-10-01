import inspect

import pytest
import torch

from src.build.model import build_models
from src.config.train import load_train_config, normalize_train_config
from src.model.critic import PairCritic2D, PlaneCritic2D
from src.train.loss.gan import get_generator_loss, score_plane


@pytest.mark.parametrize("mode,channels", [("pair", 4), ("single", 2)])
def test_critic_input_mode_controls_current_dependency(mode, channels):
    torch.manual_seed(13)
    model_class = PairCritic2D if mode == "pair" else PlaneCritic2D
    model = model_class(2, [4, 8], 8, 1)
    previous = torch.randn(2, 2, 8, 8, requires_grad=True)
    current = torch.randn_like(previous, requires_grad=True)
    time, domain = torch.tensor([0, 1]), torch.zeros(2, dtype=torch.long)
    if mode == "single":
        assert "x_current" not in inspect.signature(model.forward).parameters
        score = model(previous, time, domain)
        changed = model(previous, time, domain)
    else:
        score = model(previous, current, time, domain)
        changed = model(previous, current + 10, time, domain)
    assert model.input.in_channels == channels
    grads = torch.autograd.grad(
        get_generator_loss(score).combine(0.5), (previous, current), allow_unused=True
    )
    assert grads[0].isfinite().all() and grads[0].abs().sum() > 0
    if mode == "single":
        assert grads[1] is None
        torch.testing.assert_close(score.logits_global, changed.logits_global)
        torch.testing.assert_close(score.logits_local, changed.logits_local)
    else:
        assert grads[1].abs().sum() > 0
        assert not torch.equal(score.logits_global, changed.logits_global)


@pytest.mark.parametrize("stage", ["low_res", "sr"])
@pytest.mark.parametrize("mode", ["pair", "single"])
def test_selected_critic_mode_controls_built_model_gradients(stage, mode):
    cfg = load_train_config(f"tests/fixtures/config/train/{stage}.yaml", stage)
    cfg["model"]["critic"].update(input_mode=mode, channels=[4, 8])
    cfg["model"]["generator"].update(
        channels=[4, 8], embedding_channels=8, latent_channels=4
    )
    cfg["model"]["gradient_checkpointing"] = False
    cfg["conditioning"]["height_enabled"] = False
    _, critics, _ = build_models(cfg)
    for critic in critics.values():
        previous = torch.randn(2, 2, 8, 8, requires_grad=True)
        current = torch.randn_like(previous, requires_grad=True)
        scores = score_plane(
            critic,
            previous,
            current,
            torch.tensor([0, 1]),
            torch.zeros(2, dtype=torch.long),
        )
        gradients = torch.autograd.grad(
            get_generator_loss(scores).combine(0.5),
            (previous, current),
            allow_unused=True,
        )
        assert gradients[0].isfinite().all() and gradients[0].abs().sum() > 0
        if mode == "pair":
            assert gradients[1].isfinite().all() and gradients[1].abs().sum() > 0
        else:
            assert gradients[1] is None


@pytest.mark.parametrize("stage", ["low_res", "sr"])
@pytest.mark.parametrize("value", [None, True, "current", [], 1])
def test_invalid_critic_modes_are_rejected(stage, value):
    with pytest.raises(ValueError, match="input_mode"):
        normalize_train_config({"model": {"critic": {"input_mode": value}}}, stage)
