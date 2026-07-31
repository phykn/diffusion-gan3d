from collections.abc import Sequence
from pathlib import Path

import torch

from ..anchor import PlaneAnchor, build_anchors
from ..diffusion import Diffusion
from ..model.denoiser import Denoiser3D
from ..train.weights import WEIGHTS_NAME


def find_weights(run_root: str | Path) -> Path:
    root = Path(run_root)
    paths = tuple(root.glob(f"*/{WEIGHTS_NAME}"))
    if not paths:
        raise FileNotFoundError(f"no {WEIGHTS_NAME} file was found under {root}.")
    return max(paths, key=lambda path: path.stat().st_mtime_ns)


class Sampler:
    def __init__(
        self,
        model: Denoiser3D,
        diffusion: Diffusion,
        *,
        device: torch.device,
        patch_size: int,
        num_phases: int,
        latent_channels: int,
        anchor_enabled: bool,
        use_amp: bool,
    ) -> None:
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.patch_size = patch_size
        self.num_phases = num_phases
        self.latent_channels = latent_channels
        self.anchor_enabled = anchor_enabled
        self.use_amp = use_amp

    @torch.no_grad()
    def sample(
        self,
        *,
        anchors: Sequence[PlaneAnchor] = (),
    ) -> torch.Tensor:
        terminal = torch.randn(
            1,
            self.num_phases,
            self.patch_size,
            self.patch_size,
            self.patch_size,
            device=self.device,
            dtype=torch.float32,
        )
        anchor = build_anchors(
            anchors,
            batch_size=1,
            num_phases=self.num_phases,
            volume_size=self.patch_size,
            device=self.device,
            dtype=terminal.dtype,
        )
        if anchor is not None and not self.anchor_enabled:
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
                self.latent_channels,
                model_kwargs=kwargs,
            )
        probabilities = (clean.float() + 1.0).mul_(0.5).clamp_(0.0, 1.0)
        probabilities.div_(
            probabilities.sum(dim=1, keepdim=True).clamp_min_(
                torch.finfo(probabilities.dtype).eps
            )
        )
        return probabilities.squeeze(0).cpu()

    @torch.no_grad()
    def generate(
        self,
        *,
        anchors: Sequence[PlaneAnchor] = (),
    ) -> torch.Tensor:
        probabilities = self.sample(anchors=anchors)
        return probabilities.argmax(dim=0).to(torch.uint8)
