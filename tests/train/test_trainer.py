import pytest
import torch

from src.train.run import build_trainer


@pytest.fixture
def trainer(tiny_configs):
    return build_trainer(tiny_configs["lr"], torch.device("cpu"))


def test_step_reports_losses_and_burn_in(trainer):
    metrics = trainer.step(0, transition=0)
    for tag in (
        "loss/generator",
        "loss/critic",
        "loss/vf",
        "loss/real_transition",
        "loss/mirror_balance",
        "burn_in/generator_lr",
        "burn_in/ema_decay",
    ):
        assert tag in metrics, tag
    assert all(not isinstance(value, torch.Tensor) for value in metrics.values())


def test_non_finite_loss_stops_without_amp(trainer, monkeypatch):
    assert not trainer.denoiser_amp
    original = trainer.objective

    def broken(*args):
        loss, values = original(*args)
        return loss * float("nan"), values

    monkeypatch.setattr(trainer, "objective", broken)
    with pytest.raises(FloatingPointError, match="non-finite generator loss"):
        trainer.step(0)


def test_non_finite_gradient_stops_without_amp(trainer):
    parameter = next(trainer.denoiser.parameters())
    parameter.register_hook(lambda grad: grad * float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite generator gradient"):
        trainer.step(0)
