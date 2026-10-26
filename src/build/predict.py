from pathlib import Path

import torch

from src.build.model import build_denoiser, build_diffusion
from src.config.data import get_resolution
from src.config.files import find_saved_config
from src.config.train import load_train_config
from src.device import inference_amp
from src.predict.generator import Generator
from src.storage import load_model


def resolve_weights(weights: str | Path) -> Path:
    path = Path(weights).expanduser().resolve()
    if path.is_dir():
        path = path / "generator.pt"
    if not path.is_file():
        raise FileNotFoundError(f"generator weights do not exist: {path}")
    return path


def load_generator(
    weights: str | Path,
    device: torch.device,
) -> Generator:
    path = resolve_weights(weights)
    cfg = load_train_config(find_saved_config(path))
    return build_generator(path, cfg, device)


def build_generator(
    weights: str | Path,
    cfg: dict,
    device: torch.device,
) -> Generator:
    path = Path(weights).resolve()
    denoiser = build_denoiser(cfg).to(device)
    try:
        load_model(path, denoiser)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"weights file is not compatible with the configured denoiser: {path}"
        ) from exc
    # Anchor guidance differentiates only the noisy state, never the weights.
    denoiser.eval().requires_grad_(False)
    return make_generator(
        denoiser, cfg, device, get_resolution(cfg["data"]).lo_res_size
    )


def make_generator(denoiser, cfg: dict, device: torch.device, patch_size: int):
    """Sample a loaded LR or SR denoiser with its config's diffusion and conditions."""
    data = cfg["data"]
    return Generator(
        denoiser,
        build_diffusion(cfg).to(device),
        device=device,
        patch_size=patch_size,
        num_phases=data["num_phases"],
        latent_channels=cfg["model"]["generator"]["latent_channels"],
        use_amp=inference_amp(cfg["train"]["mixed_precision"], device),
        height_data=data if cfg["conditioning"]["height_enabled"] else None,
    )
