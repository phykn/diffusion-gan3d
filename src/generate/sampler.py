import secrets
from pathlib import Path

import torch

from ..diffusion import DiffusionProcess
from ..model import Denoiser3D
from ..train import TrainConfig, load_train_config
from ..train.checkpoint import FORMAT_VERSION


def latest_checkpoint(run_root: str | Path) -> Path:
    root = Path(run_root)
    paths = tuple(root.glob("*/last.pt"))
    if not paths:
        raise FileNotFoundError(f"no last.pt checkpoint was found under {root}.")
    return max(paths, key=lambda path: path.stat().st_mtime_ns)


def load_ema_denoiser(
    checkpoint: str | Path,
    *,
    device: torch.device,
) -> tuple[Denoiser3D, TrainConfig, int]:
    checkpoint_path = Path(checkpoint).resolve()
    cfg = load_train_config(checkpoint_path.parent / "train.yaml")
    model = Denoiser3D(
        num_phases=cfg.data.num_phases,
        base_channels=cfg.model.base_channels,
        channel_multipliers=cfg.model.channel_multipliers,
        embedding_channels=cfg.model.embedding_channels,
        latent_channels=cfg.model.latent_channels,
        gradient_checkpointing=False,
    ).to(device)
    try:
        values = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if values["format_version"] != FORMAT_VERSION:
            raise ValueError("checkpoint format version is not supported.")
        model.load_state_dict(values["models"]["ema_denoiser"], strict=True)
        step = int(values["step"])
    except (KeyError, TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"checkpoint does not contain a compatible EMA denoiser: "
            f"{checkpoint_path}"
        ) from exc
    model.eval()
    return model, cfg, step


@torch.no_grad()
def generate_labels(
    model: Denoiser3D,
    cfg: TrainConfig,
    *,
    device: torch.device,
    size: int | None = None,
    seed: int | None = None,
    mixed_precision: bool | None = None,
) -> tuple[torch.Tensor, int]:
    size = cfg.data.patch_size if size is None else size
    if not isinstance(size, int) or size < 1:
        raise ValueError("size must be a positive integer.")
    if size % model.downsample_factor:
        raise ValueError(
            f"size must be divisible by {model.downsample_factor}."
        )
    if seed is None:
        seed = secrets.randbits(63)
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    use_amp = (
        cfg.train.mixed_precision
        if mixed_precision is None
        else mixed_precision
    )
    use_amp = bool(use_amp and device.type == "cuda")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    terminal = torch.randn(
        1,
        cfg.data.num_phases,
        size,
        size,
        size,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    diffusion = DiffusionProcess(
        cfg.diffusion.timesteps,
        beta_min=cfg.diffusion.beta_min,
        beta_max=cfg.diffusion.beta_max,
    ).to(device)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=use_amp,
    ):
        clean = diffusion.reverse_chain(
            model,
            terminal,
            cfg.model.latent_channels,
            generator=generator,
        )
    labels = clean.argmax(dim=1).squeeze(0).to(torch.uint8).cpu()
    return labels, seed
