import torch

from src.rng import NOISE, RunRandom


def test_a_state_restores_every_stream():
    random = RunRandom(3)
    state = random.state()
    first = {
        name: torch.rand(4, generator=gen) for name, gen in random.generators.items()
    }
    random.load_state(state)
    again = {
        name: torch.rand(4, generator=gen) for name, gen in random.generators.items()
    }
    assert first.keys() == again.keys() and NOISE in first
    for name in first:
        assert torch.equal(first[name], again[name]), name


def test_noise_saved_on_another_device_type_restarts_alone(capsys):
    random = RunRandom(3)
    state = random.state()
    # A 44-byte MPS generator state cannot enter a CPU generator.
    state[NOISE] = torch.arange(44, dtype=torch.uint8)
    expected = torch.rand(4, generator=RunRandom(3)["data"])
    random.load_state(state)
    assert torch.equal(torch.rand(4, generator=random["data"]), expected)
    noise = torch.rand(4, generator=random[NOISE])
    random.load_state(state)
    assert torch.equal(torch.rand(4, generator=random[NOISE]), noise)
    assert "another device type" in capsys.readouterr().out
