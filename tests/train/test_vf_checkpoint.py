import torch

from src.model.denoiser import Denoiser3D


def test_vf_adapter_state_loads_strictly_and_supports_both_paths() -> None:
    source = _denoiser()
    state = {
        name: value.detach().clone() for name, value in source.state_dict().items()
    }

    assert {
        "vf_mlp.0.weight",
        "vf_mlp.0.bias",
        "vf_mlp.2.weight",
        "vf_mlp.2.bias",
    } <= state.keys()

    restored = _denoiser()
    result = restored.load_state_dict(state, strict=True)

    assert not result.missing_keys
    assert not result.unexpected_keys

    current = torch.randn(1, 3, 4, 4, 4)
    time = torch.zeros(1, dtype=torch.long)
    latent = torch.randn(1, 4)
    vf = torch.tensor([[0.5, 0.1, 0.4]])
    with torch.no_grad():
        unconditional = restored(current, time, latent, vf=None)
        conditional = restored(current, time, latent, vf=vf)

    assert unconditional.shape == current.shape
    assert conditional.shape == current.shape
    assert bool(torch.isfinite(unconditional).all())
    assert bool(torch.isfinite(conditional).all())


def _denoiser() -> Denoiser3D:
    return Denoiser3D(
        num_phases=3,
        base_channels=4,
        channel_multipliers=(1, 2),
        embedding_channels=8,
        latent_channels=4,
    )
