import copy
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from src.build.sr import build_sr_trainer
from src.build.trainer import build_trainer
from src.config.data import get_plane_groups
from src.config.train import load_train_config
from src.model.critic import CriticScores
from src.plane import PLANES
from src.train.run.loop import run_train
from src.train.sr import resume_sr_training, save_sr_training

GROUPS = [[["xy", "xz", "yz"]], [["xy"], ["xz", "yz"]], [["xy"], ["xz"], ["yz"]]]


def config(tmp_path, stage, groups):
    images = tmp_path / "images"
    images.mkdir(exist_ok=True)
    Image.fromarray((np.indices((8, 8)).sum(0) % 2).astype(np.uint8)).save(
        images / "sample.png"
    )
    cfg = load_train_config(f"tests/fixtures/config/train/{stage}.yaml", stage)
    cfg["data"].update(
        crop_size=8,
        lo_res_size=8,
        domains={0: {plane: [str(images)] for plane in PLANES}},
    )
    cfg["model"]["critic"].update(channels=[4, 8], plane_groups=groups)
    cfg["augmentation"]["probability"] = 0
    cfg["train"].update(total_steps=1, mixed_precision=False, volume_batch_size=1)
    if stage == "low_res":
        cfg["model"]["generator"].update(
            channels=[4, 8], embedding_channels=8, latent_channels=4
        )
        cfg["model"]["gradient_checkpointing"] = False
        cfg["model"]["diffusion"]["num_steps"] = 2
        cfg["conditioning"]["anchor"]["probability"] = 0.0
        cfg["loss"].update(r1_weight=0.01, r1_every_steps=1)
        cfg["train"].update(real_batch_size=2, slice_pairs_per_plane=2, num_workers=0)
    else:
        cfg["model"]["generator"].update(
            channels=[4, 8], embedding_channels=8, latent_channels=4
        )
        cfg["model"]["diffusion"]["num_steps"] = 2
        cfg["model"]["gradient_checkpointing"] = False
        cfg["data"]["hi_res_size"] = 12
        cfg["train"].update(
            real_batch_size=2, slice_pairs_per_plane=2, weights_every_steps=1
        )
        cfg["lr_bank"]["samples_per_domain"] = 2
    return cfg


@pytest.mark.parametrize("groups", GROUPS)
def test_lr_groups_train_once_and_save_one_file_per_group(tmp_path, groups):
    torch.set_num_threads(1)
    cfg = config(tmp_path, "low_res", groups)
    trainer = build_trainer(cfg, torch.device("cpu"))
    assert len(trainer.critics) == len(groups) == len(trainer.critic_optims)
    assert trainer.active_axes == (0, 1, 2)
    params = [id(p) for critic in trainer.critics.values() for p in critic.parameters()]
    assert len(set(params)) == len(params)
    for optimizer in trainer.critic_optims.values():
        optimizer.step = Mock(wraps=optimizer.step)
    before = copy.deepcopy(trainer.denoiser.state_dict())
    run_train(trainer, steps=1, save_every=1, run_dir=tmp_path / "run")
    assert any(
        not torch.equal(before[key], value)
        for key, value in trainer.denoiser.state_dict().items()
    )
    for name, optimizer in trainer.critic_optims.items():
        assert optimizer.step.call_count == 1
        assert (tmp_path / "run" / f"critic_{name}.pt").is_file()
        assert optimizer.state
    cfg["train"]["initial_weights"] = str(tmp_path / "run")
    restored = build_trainer(cfg, torch.device("cpu"))
    for name, critic in trainer.critics.items():
        for key, value in critic.state_dict().items():
            torch.testing.assert_close(
                restored.critics[name].state_dict()[key], value, rtol=0, atol=0
            )


class ScalarCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def forward(self, previous, current, time, domain):
        score = self.weight * previous.flatten(1).mean(1)
        return CriticScores(score, score[:, None, None, None])


@pytest.mark.parametrize("groups", GROUPS)
def test_lr_shared_update_averages_plane_gradients(tmp_path, groups):
    cfg = config(tmp_path, "low_res", groups)
    trainer = build_trainer(cfg, torch.device("cpu"))
    trainer.r1_gamma = 0
    trainer.critics = nn.ModuleDict(
        {name: ScalarCritic() for name in trainer.critic_groups}
    )
    trainer.critic_optims = {
        name: torch.optim.SGD(model.parameters(), lr=1)
        for name, model in trainer.critics.items()
    }
    real = {
        axis: (
            torch.full((2, trainer.num_phases, 8, 8), float(axis + 1)),
            torch.zeros(2, trainer.num_phases, 8, 8),
        )
        for axis in range(3)
    }
    fake = {
        axis: (
            torch.zeros(2, trainer.num_phases, 8, 8),
            torch.zeros(2, trainer.num_phases, 8, 8),
        )
        for axis in range(3)
    }
    trainer.diffusion.sample_pair = lambda images, transition: (
        images,
        torch.zeros_like(images),
    )
    trainer.update_critics(
        0,
        fake,
        {axis: (pair[0] + 1) * 0.5 for axis, pair in real.items()},
        0,
        {axis: 0 for axis in range(3)},
    )
    for name, axes in trainer.critic_groups.items():
        # At weight=0, d softplus(-w*x)/dw = -x/2. Fake x is zero.
        expected = (
            (1 + trainer.critic_local_weight)
            * sum(axis + 1 for axis in axes)
            / (2 * len(axes))
            / len(groups)
        )
        assert trainer.critics[name].weight.item() == pytest.approx(expected)


@pytest.mark.parametrize("groups", GROUPS)
def test_sr_groups_step_once_per_critic_update_and_restore_state(tmp_path, groups):
    torch.set_num_threads(1)
    cfg = config(tmp_path, "sr", groups)
    bank = {0: torch.rand(2, 2, 8, 8, 8).softmax(1)}
    trainer = build_sr_trainer(cfg, bank, torch.device("cpu"))
    assert len(trainer.critics) == len(groups) == len(trainer.critic_optims)
    for optimizer in trainer.critic_optims.values():
        optimizer.step = Mock(wraps=optimizer.step)
    trainer.step(0)
    for optimizer in trainer.critic_optims.values():
        assert optimizer.step.call_count == 1
    path = tmp_path / "last.pt"
    save_sr_training(trainer, path)
    expected = copy.deepcopy(trainer.denoiser.state_dict())
    restored = build_sr_trainer(cfg, bank, torch.device("cpu"))
    resume_sr_training(restored, torch.load(path, weights_only=True))
    for key, value in expected.items():
        torch.testing.assert_close(
            restored.denoiser.state_dict()[key], value, rtol=0, atol=0
        )
    for group in trainer.critic_optims:
        torch.testing.assert_close(
            restored.critic_optims[group].state_dict(),
            trainer.critic_optims[group].state_dict(),
        )
    restored.step(restored.completed_steps)
    assert restored.completed_steps == 2
    other = copy.deepcopy(cfg)
    other["model"]["critic"]["plane_groups"] = (
        GROUPS[1] if len(groups) != 2 else GROUPS[0]
    )
    with pytest.raises(ValueError, match="plane_groups cannot change"):
        resume_sr_training(
            build_sr_trainer(other, bank, torch.device("cpu")),
            torch.load(path, weights_only=True),
        )


def test_sr_groups_are_per_domain_and_use_only_observed_members(tmp_path):
    cfg = config(tmp_path, "sr", GROUPS[0])
    paths = cfg["data"]["domains"][0]["xy"]
    cfg["data"]["domains"] = {0: {"xz": paths, "xy": paths}, 1: {"yz": paths}}
    bank = {domain: torch.full((2, 2, 8, 8, 8), 0.5) for domain in (0, 1)}
    trainer = build_sr_trainer(cfg, bank, torch.device("cpu"))
    assert set(trainer.critics) == {"0_xy_xz_yz", "1_xy_xz_yz"}
    assert trainer.critic_groups_by_domain == {
        0: {"0_xy_xz_yz": (0, 1)},
        1: {"1_xy_xz_yz": (2,)},
    }
    metrics = trainer.step(0)
    for name, optimizer in trainer.critic_optims.items():
        assert bool(optimizer.state) == name.startswith(f"{metrics.domain}_")


@pytest.mark.parametrize(
    "groups",
    [
        None,
        [],
        ["xy"],
        [[]],
        [["xy", "xy"], ["xz", "yz"]],
        [["xy"], ["xy", "xz", "yz"]],
        [["xy", "xz"]],
        [["xy", "xz", "yx"]],
        [[0, 1, 2]],
    ],
)
def test_invalid_group_partitions_are_rejected(tmp_path, groups):
    with pytest.raises(ValueError, match="group|plane|xy"):
        get_plane_groups(config(tmp_path, "low_res", groups))


def test_group_order_is_canonical_and_missing_groups_are_rejected(tmp_path):
    cfg = config(tmp_path, "low_res", [["yz", "xz"], ["xy"]])
    assert get_plane_groups(cfg) == {"xy": (0,), "xz_yz": (1, 2)}
    del cfg["model"]["critic"]["plane_groups"]
    with pytest.raises(ValueError, match="plane_groups"):
        get_plane_groups(cfg)
