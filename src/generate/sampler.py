from collections.abc import Sequence
from pathlib import Path

import torch

from ..anchor import PlaneAnchor, assemble_anchors
from ..diffusion import DiffusionProcess
from ..model import Denoiser3D
from ..train import TrainConfig, load_train_config
from ..train.weights import MODEL_WEIGHTS_NAME


def latest_model_weights(run_root: str | Path) -> Path:
    root = Path(run_root)
    paths = tuple(root.glob(f"*/{MODEL_WEIGHTS_NAME}"))
    if not paths:
        raise FileNotFoundError(
            f"no {MODEL_WEIGHTS_NAME} file was found under {root}."
        )
    return max(paths, key=lambda path: path.stat().st_mtime_ns)


def load_denoiser_weights(
    weights: str | Path,
    *,
    device: torch.device,
) -> tuple[Denoiser3D, TrainConfig]:
    weights_path = Path(weights).resolve()
    cfg = load_train_config(weights_path.parent / "train.yaml")
    model = Denoiser3D(
        num_phases=cfg.data.num_phases,
        base_channels=cfg.model.base_channels,
        channel_multipliers=cfg.model.channel_multipliers,
        embedding_channels=cfg.model.embedding_channels,
        latent_channels=cfg.model.latent_channels,
        gradient_checkpointing=False,
    ).to(device)
    try:
        state_dict = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict, strict=True)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"weights file is not compatible with the configured denoiser: "
            f"{weights_path}"
        ) from exc
    model.eval()
    return model, cfg


@torch.no_grad()
def generate_labels(
    model: Denoiser3D,
    cfg: TrainConfig,
    *,
    device: torch.device,
    size: int | None = None,
    mixed_precision: bool | None = None,
    anchors: Sequence[PlaneAnchor] = (),
) -> torch.Tensor:
    size = cfg.data.patch_size if size is None else size
    if not isinstance(size, int) or size < 1:
        raise ValueError("size must be a positive integer.")
    if size % model.downsample_factor:
        raise ValueError(
            f"size must be divisible by {model.downsample_factor}."
        )
    use_amp = (
        cfg.train.mixed_precision
        if mixed_precision is None
        else mixed_precision
    )
    use_amp = bool(use_amp and device.type == "cuda")
    terminal = torch.randn(
        1,
        cfg.data.num_phases,
        size,
        size,
        size,
        device=device,
        dtype=torch.float32,
    )
    anchor = assemble_anchors(
        anchors,
        batch_size=terminal.shape[0],
        num_phases=cfg.data.num_phases,
        volume_size=size,
        device=device,
        dtype=terminal.dtype,
    )
    if anchor is not None and not cfg.anchor.enabled:
        raise ValueError("selected weights were trained with anchors disabled.")
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
            model_kwargs=(
                None
                if anchor is None
                else {
                    "anchor_image": anchor.image,
                    "anchor_mask": anchor.mask,
                }
            ),
        )
    labels = clean.argmax(dim=1).squeeze(0).to(torch.uint8).cpu()
    return labels
