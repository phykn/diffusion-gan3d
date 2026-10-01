import inspect

import pytest
import torch

from src.config.train import load_train_config, normalize_train_config
from src.model.critic import PlaneCritic2D
from src.train.loss.gan import get_generator_loss


def test_plane_critic_uses_only_previous_state():
    torch.manual_seed(13)
    model = PlaneCritic2D(2, [4, 8], 8, 1)
    previous = torch.randn(2, 2, 8, 8, requires_grad=True)
    current = torch.randn_like(previous, requires_grad=True)
    time, domain = torch.tensor([0, 1]), torch.zeros(2, dtype=torch.long)
    assert "x_current" not in inspect.signature(model.forward).parameters
    score = model(previous, time, domain)
    assert model.input.in_channels == 2
    grads = torch.autograd.grad(
        get_generator_loss(score).combine(0.5), (previous, current), allow_unused=True
    )
    assert grads[0].isfinite().all() and grads[0].abs().sum() > 0
    assert grads[1] is None


@pytest.mark.parametrize("stage", ["low_res", "sr"])
def test_critic_mode_defaults_and_validation(stage):
    assert (
        load_train_config(f"config/train/{stage}.yaml", stage)["model"]["critic"][
            "input_mode"
        ]
        == "single"
    )
    assert (
        normalize_train_config({}, stage)["model"]["critic"]["input_mode"] == "single"
    )
    cfg = {"model": {"critic": {"input_mode": "single"}}}
    assert (
        normalize_train_config(cfg, stage)["model"]["critic"]["input_mode"] == "single"
    )
    for value in (None, True, "current", "pair", [], 1):
        with pytest.raises(ValueError, match="input_mode"):
            normalize_train_config({"model": {"critic": {"input_mode": value}}}, stage)
