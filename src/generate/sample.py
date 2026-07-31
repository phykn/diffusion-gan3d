from collections.abc import Sequence
from pathlib import Path

import torch

from ..anchor import PlaneAnchor, build_anchors
from ..build import build_denoiser, build_diffusion
from ..model import Denoiser3D
from ..train import TrainConfig, load_config
from ..train.weights import WEIGHTS_NAME


def find_weights(run_root: str | Path) -> Path:
    root = Path(run_root)
    paths = tuple(root.glob(f"*/{WEIGHTS_NAME}"))
    if not paths:
        raise FileNotFoundError(f"no {WEIGHTS_NAME} file was found under {root}.")
    return max(paths, key=lambda path: path.stat().st_mtime_ns)


def load_model(
    weights: str | Path,
    *,
    device: torch.device,
) -> tuple[Denoiser3D, TrainConfig]:
    path = Path(weights).resolve()
    cfg = load_config(path.parent / "train.yaml")
    model = build_denoiser(cfg.data, cfg.model, checkpointing=False).to(device)
    try:
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state, strict=True)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(
            f"weights file is not compatible with the configured denoiser: {path}"
        ) from exc
    model.eval()
    return model, cfg


class Sampler:
    def __init__(
        self,
        model: Denoiser3D,
        cfg: TrainConfig,
        *,
        device: torch.device,
        mixed_precision: bool | None = None,
    ) -> None:
        if mixed_precision is not None and not isinstance(mixed_precision, bool):
            raise TypeError("mixed_precision must be boolean or None.")
        self.model = model
        self.cfg = cfg
        self.device = device
        self.use_amp = bool(
            (cfg.train.mixed_precision if mixed_precision is None else mixed_precision)
            and device.type == "cuda"
        )
        self.diffusion = build_diffusion(cfg.diffusion).to(device)

    @torch.no_grad()
    def generate(
        self,
        *,
        size: int | None = None,
        anchors: Sequence[PlaneAnchor] = (),
    ) -> torch.Tensor:
        size = self.cfg.data.patch_size if size is None else size
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise ValueError("size must be a positive integer.")
        if size % self.model.downsample_factor:
            raise ValueError(
                f"size must be divisible by {self.model.downsample_factor}."
            )
        terminal = torch.randn(
            1,
            self.cfg.data.num_phases,
            size,
            size,
            size,
            device=self.device,
            dtype=torch.float32,
        )
        anchor = build_anchors(
            anchors,
            batch_size=1,
            num_phases=self.cfg.data.num_phases,
            volume_size=size,
            device=self.device,
            dtype=terminal.dtype,
        )
        if anchor is not None and not self.cfg.anchor.enabled:
            raise ValueError("selected weights were trained with anchors disabled.")
        kwargs = (
            None
            if anchor is None
            else {
                "anchor_image": anchor.image,
                "anchor_mask": anchor.mask,
            }
        )
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.use_amp,
        ):
            clean = self.diffusion.sample(
                self.model,
                terminal,
                self.cfg.model.latent_channels,
                model_kwargs=kwargs,
            )
        return clean.argmax(dim=1).squeeze(0).to(torch.uint8).cpu()
