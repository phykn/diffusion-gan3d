import inspect

import pytest
import torch

from src.config.train import load_train_config, normalize_train_config
from src.model.critic import PairCritic2D, PlaneCritic2D
from src.train.loss.gan import get_generator_loss


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
def test_critic_mode_defaults_and_validation(stage):
    assert (
        load_train_config(f"config/train/{stage}.yaml", stage)["model"]["critic"][
            "input_mode"
        ]
        == "single"
    )
    # Missing mode in historical configurations must still restore pair weights.
    assert normalize_train_config({}, stage)["model"]["critic"]["input_mode"] == "pair"
    cfg = {"model": {"critic": {"input_mode": "single"}}}
    assert (
        normalize_train_config(cfg, stage)["model"]["critic"]["input_mode"] == "single"
    )
    for value in (None, True, "current", [], 1):
        with pytest.raises(ValueError, match="input_mode"):
            normalize_train_config({"model": {"critic": {"input_mode": value}}}, stage)
