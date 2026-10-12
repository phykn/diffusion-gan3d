from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from src.model.layers import NULL_DOMAIN
from src.train.anchor_bank import AnchorBank
from src.train.batch import RealBatch
from src.train.conditioning import ConditionSampler


def test_anchor_transitions_prioritize_the_final_step() -> None:
    sampler = object.__new__(ConditionSampler)
    sampler.settings = SimpleNamespace(profile_settings={"enabled": False})
    sampler.timesteps = 11

    with (
        patch("src.train.conditioning.torch.rand", return_value=torch.tensor(0.1)),
        patch("src.train.conditioning.torch.randint") as randint,
    ):
        assert sampler.sample_transition(anchored=True) == 0
        randint.assert_not_called()

    with (
        patch("src.train.conditioning.torch.rand", return_value=torch.tensor(0.5)),
        patch(
            "src.train.conditioning.torch.randint", return_value=torch.tensor(7)
        ) as randint,
    ):
        assert sampler.sample_transition(anchored=True) == 7
        randint.assert_called_once_with(1, 11, ())

    with patch(
        "src.train.conditioning.torch.randint",
        return_value=torch.tensor(6),
    ) as randint:
        assert sampler.sample_transition(anchored=False) == 6
        randint.assert_called_once_with(11, ())


def test_get_batches_uses_one_domain_for_all_axes() -> None:
    sampler = object.__new__(ConditionSampler)
    sampler.settings = SimpleNamespace(
        height_data=None, profile_settings={"enabled": False}
    )
    sampler.device = torch.device("cpu")
    streams = {
        domain: {axis: _stream(torch.full((1, 2, 2), domain)) for axis in (0, 1, 2)}
        for domain in (0, 1)
    }
    sampler.streams = streams
    sampler.active_axes = (0, 1, 2)
    sampler.axis_domains = {axis: (0, 1) for axis in (0, 1, 2)}

    batches = sampler.get_batches(1)

    assert all(bool((batch == 1).all()) for batch in batches.images.values())
    assert all(stream.next.call_count == 0 for stream in streams[0].values())
    assert all(stream.next.call_count == 1 for stream in streams[1].values())


def test_missing_axes_borrow_from_axis_providers() -> None:
    sampler = object.__new__(ConditionSampler)
    sampler.settings = SimpleNamespace(
        height_data=None, profile_settings={"enabled": False}
    )
    sampler.device = torch.device("cpu")
    sampler.streams = {
        0: {0: _stream(torch.full((1, 2, 2), 10))},
        1: {
            0: _stream(torch.full((1, 2, 2), 20)),
            1: _stream(torch.full((1, 2, 2), 21)),
            2: _stream(torch.full((1, 2, 2), 22)),
        },
    }
    sampler.axis_domains = {0: (0, 1), 1: (1,), 2: (1,)}
    sampler.active_axes = (0, 1, 2)

    sources = sampler.select_batch_domains(0)
    batches = sampler.get_batches(0, sources)
    critic_domains = sampler.make_critic_domains(0, 0, sources)

    assert sources == {0: 0, 1: 1, 2: 1}
    assert [int(batches.images[axis][0, 0, 0]) for axis in (0, 1, 2)] == [10, 21, 22]
    assert critic_domains == {0: 0, 1: NULL_DOMAIN, 2: NULL_DOMAIN}


def test_domain_dropout_masks_every_axis_critic() -> None:
    sources = {0: 0, 1: 1, 2: 1}

    critic_domains = ConditionSampler.make_critic_domains(0, NULL_DOMAIN, sources)

    assert critic_domains == {axis: NULL_DOMAIN for axis in (0, 1, 2)}


def test_domain_dropout_probability_controls_the_model_condition() -> None:
    sampler = object.__new__(ConditionSampler)
    sampler.settings = SimpleNamespace(
        profile_settings={"enabled": False}, domain_dropout=0.0
    )
    assert sampler.sample_domain_condition(2) == 2

    sampler.settings.domain_dropout = 1.0
    assert sampler.sample_domain_condition(2) == NULL_DOMAIN


def test_anchor_training_alternates_external_and_multi_anchor_modes() -> None:
    sampler = object.__new__(ConditionSampler)
    sampler.settings = SimpleNamespace(
        profile_settings={"enabled": False},
        anchor_training_probability=0.5,
        volume_batch_size=1,
    )
    sampler.use_multi_anchor_next = False
    sampler.anchor_bank = AnchorBank()
    sampler.device = torch.device("cpu")
    sampler.sample_real_anchor = Mock(return_value=Mock(source="real"))

    with patch("src.train.conditioning.torch.rand", return_value=torch.tensor(0.9)):
        assert sampler.sample_anchor(RealBatch({}), 2, owned_axes=()) is None
    assert not sampler.use_multi_anchor_next

    sampler.settings.anchor_training_probability = 1.0
    sources = [
        sampler.sample_anchor(RealBatch({}), 2, owned_axes=()).source for _ in range(3)
    ]

    assert sources == [
        "real",
        "real",
        "real",
    ]  # Replay waits for a measured-conditioned sample.


def test_joint_cfg_dropout_uses_four_categorical_anchor_vf_states() -> None:
    sampler = object.__new__(ConditionSampler)
    sampler.settings = SimpleNamespace(
        profile_settings={"enabled": False},
        volume_batch_size=4,
        cfg_drop_each_probability=0.1,
    )
    sampler.device = torch.device("cpu")

    with patch(
        "src.train.conditioning.torch.rand",
        return_value=torch.tensor((0.05, 0.15, 0.25, 0.35)),
    ):
        presence = sampler.sample_condition_presence(has_anchor=True)

    assert presence.anchor.tolist() == [False, False, True, True]
    assert presence.vf.tolist() == [False, True, False, True]


def test_single_condition_dropout_matches_joint_marginal_visibility() -> None:
    sampler = object.__new__(ConditionSampler)
    sampler.settings = SimpleNamespace(
        profile_settings={"enabled": False},
        volume_batch_size=2,
        cfg_drop_each_probability=0.1,
    )
    sampler.device = torch.device("cpu")

    with patch(
        "src.train.conditioning.torch.rand",
        return_value=torch.tensor((0.19, 0.21)),
    ):
        presence = sampler.sample_condition_presence(has_anchor=False)

    assert presence.anchor.tolist() == [False, False]
    assert presence.vf.tolist() == [False, True]


def _stream(images):
    return Mock(next=Mock(side_effect=lambda: {"image": images.clone()}))
