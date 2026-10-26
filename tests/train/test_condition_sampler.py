from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from src.model.layers import NULL_DOMAIN
from src.train.conditioning import ConditionSampler


def test_transitions_are_uniform() -> None:
    sampler = object.__new__(ConditionSampler)
    sampler.timesteps = 11

    with patch(
        "src.train.conditioning.torch.randint",
        return_value=torch.tensor(6),
    ) as randint:
        assert sampler.sample_transition() == 6
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


def test_domain_keep_probability_controls_the_model_condition() -> None:
    sampler = object.__new__(ConditionSampler)
    sampler.settings = SimpleNamespace(
        profile_settings={"enabled": False}, domain_keep_probability=1.0
    )
    assert sampler.sample_domain_condition(2) == 2

    sampler.settings.domain_keep_probability = 0.0
    assert sampler.sample_domain_condition(2) == NULL_DOMAIN


def test_cfg_dropout_hides_vf_with_the_configured_probability() -> None:
    sampler = object.__new__(ConditionSampler)
    sampler.settings = SimpleNamespace(volume_batch_size=2, cfg_dropout_probability=0.2)

    with patch(
        "src.train.conditioning.torch.rand",
        return_value=torch.tensor((0.19, 0.21)),
    ):
        present = sampler.sample_vf_presence()

    assert present.tolist() == [False, True]


def _stream(images):
    return Mock(next=Mock(side_effect=lambda: {"image": images.clone()}))
