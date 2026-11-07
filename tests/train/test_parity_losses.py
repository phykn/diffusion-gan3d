"""Training losses against the v1 reference (SPEC Parity checks: losses)."""

import ast
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config.io import PROJECT_ROOT
from src.model.critic import CriticScores
from src.train.losses.consistency import consistency_loss
from src.train.losses.gan import critic_objective, relativistic_loss
from src.train.losses.group_statistics import group_statistics_loss
from src.train.losses.mirror_balance import mirror_balance_loss
from src.train.losses.spatial_profile import profile_loss
from src.train.losses.transition import real_transition_loss
from src.train.losses.volume_fraction import vf_loss

CASES = torch.load(
    Path(__file__).parents[1] / "fixtures" / "parity" / "losses.pt", weights_only=False
)
ATOL = 1e-5


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=1e-5)


def _grad(value, inputs):
    return torch.autograd.grad(value, inputs, retain_graph=True, allow_unused=True)


def _scores(prev, curr, weights):
    joined = torch.cat([prev, curr], 1)
    first = torch.tanh(joined * weights[0]).flatten(1).sum(1)
    pooled = F.avg_pool2d(joined, 2)
    second = torch.tanh(pooled * weights[1]).flatten(1).sum(1)
    return CriticScores(first, (CriticScores(first), CriticScores(second)))


def _leaf(values):
    return values.clone().requires_grad_(True)


def test_gan():
    case = CASES["gan"]
    real = [_leaf(value) for value in case["real"]]
    fake = [_leaf(value) for value in case["fake"]]
    real_scores = _scores(*real, case["weights"])
    fake_scores = _scores(*fake, case["weights"])
    adversarial = relativistic_loss(real_scores, fake_scores)
    generator = relativistic_loss(fake_scores, real_scores)
    _close(adversarial.detach(), case["adversarial"])
    _close(generator.detach(), case["generator"])
    for actual, expected in zip(
        _grad(generator, fake), case["generator_grad"], strict=True
    ):
        _close(actual, expected)
    objective = critic_objective(
        adversarial, real_scores, fake_scores, (real[0],), (fake[0],), 0.5, 0.1, 8
    )
    _close(objective.total.detach(), case["total"])
    _close(objective.r1.detach(), case["r1"])
    _close(objective.r2.detach(), case["r2"])
    for actual, expected in zip(
        _grad(objective.total, (*real, *fake)), case["total_grad"], strict=True
    ):
        _close(actual, expected)


def test_vf():
    case = CASES["vf"]
    leaf = _leaf(case["logits"])
    value = vf_loss(leaf.softmax(1), case["target"], case["present"])
    _close(value.detach(), case["value"])
    _close(_grad(value, leaf)[0], case["grad"])


def _check_diagnostics(actual, expected):
    assert set(actual) == set(expected)
    for key, value in expected.items():
        _close(actual[key].float(), value.float())


def test_real_transition():
    for name in ("real_transition", "real_transition_height"):
        case = CASES[name]
        leaves = {axis: _leaf(value) for axis, value in case["fake_logits"].items()}
        value, diagnostics = real_transition_loss(
            case["real"],
            {axis: leaf.softmax(1) for axis, leaf in leaves.items()},
            case["max_gap"],
            case["real_heights"],
            case["fake_heights"],
            case["rotations"],
        )
        _close(value.detach(), case["value"])
        _check_diagnostics(diagnostics, case["diagnostics"])
        grads = _grad(value, tuple(leaves.values()))
        for axis, grad in zip(leaves, grads, strict=True):
            _close(grad, case["grad"][axis])


def test_group_statistics():
    for name in ("group_statistics", "group_statistics_height", "group_statistics_sr"):
        case = CASES[name]
        leaf = _leaf(case["logits"])
        value, diagnostics = group_statistics_loss(
            leaf.softmax(1),
            case["groups"],
            3,
            0.01,
            case["preserve_height"],
            case["coarse"],
        )
        _close(value.detach(), case["value"])
        _check_diagnostics(diagnostics, case["diagnostics"])
        _close(_grad(value, leaf)[0], case["grad"])


def test_mirror_balance():
    case = CASES["mirror_balance"]
    leaf = _leaf(case["logits"])
    value, diagnostics = mirror_balance_loss(leaf.softmax(1), 2, 0.0)
    _close(value.detach(), case["value"])
    _check_diagnostics(diagnostics, case["diagnostics"])
    _close(_grad(value, leaf)[0], case["grad"])


def test_spatial_profile():
    case = CASES["spatial_profile"]
    leaf = _leaf(case["logits"])
    value = profile_loss(
        leaf.softmax(1), case["target"], torch.tensor([True, True]), 4, 30.0
    )
    _close(value.detach(), case["value"])
    _close(_grad(value, leaf)[0], case["grad"])


def test_consistency():
    for name in ("consistency", "consistency_mask"):
        case = CASES[name]
        leaf = _leaf(case["logits"])
        value, error = consistency_loss(
            leaf.softmax(1), case["low"], 0.001, case["mask"]
        )
        _close(value.detach(), case["value"])
        _close(error.detach(), case["error"])
        _close(_grad(value, leaf)[0], case["grad"])


def test_pair_losses_use_the_shared_tables():
    """transition, group_statistics and mirror_balance import src/stats/pairs.py
    and define no pair-table function of their own."""
    folder = PROJECT_ROOT / "src" / "train" / "losses"
    for name in ("transition", "group_statistics", "mirror_balance"):
        tree = ast.parse((folder / f"{name}.py").read_text(encoding="utf-8"))
        imports = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        }
        assert "src.stats.pairs" in imports, name
        defined = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        }
        assert not defined & {
            "phase_pairs",
            "section_pairs",
            "volume_pairs",
            "volume_phase_pairs",
            "pair_correlations",
            "phase_correlations",
        }, name
