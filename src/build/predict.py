from pathlib import Path

import torch

from src.build.model import build_denoiser, build_diffusion
from src.config import (
    find_train_config,
    get_sizes,
    load_generation_settings,
    load_train_config,
)
from src.predict.generator import Generator
from src.storage import load_model


def load_generator(
    weights: str | Path,
    device: torch.device,
) -> Generator:
    path = Path(weights).resolve()
    config = find_train_config(path)
    cfg = load_train_config(config)
    denoiser = build_denoiser(cfg, checkpointing=False).to(device)
    try:
        load_model(path, denoiser)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"weights file is not compatible with the configured denoiser: {path}"
        ) from exc
    denoiser.eval()
    data = cfg["data"]
    model = cfg["model"]
    train = cfg["train"]
    generation = load_generation_settings()
    use_amp = train["mixed_precision"] and device.type == "cuda"
    return Generator(
        denoiser,
        build_diffusion(cfg).to(device),
        device=device,
        patch_size=get_sizes(data)[1],
        num_phases=data["num_phases"],
        latent_channels=model["generator"]["latent_channels"],
        use_amp=use_amp,
        anchor_spread=generation.anchor_spread,
    )
