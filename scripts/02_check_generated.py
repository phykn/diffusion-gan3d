import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.generate import generate_labels, latest_checkpoint, load_ema_denoiser
from src.train import load_train_config

DEFAULT_CONFIG = PROJECT_ROOT / "config" / "train.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--size", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint is None:
        root = load_train_config(args.config).output.run_root
        checkpoint = latest_checkpoint(root)
    else:
        checkpoint = args.checkpoint
    model, cfg, step = load_ema_denoiser(checkpoint, device=device)
    labels, seed = generate_labels(
        model,
        cfg,
        device=device,
        size=args.size,
        seed=args.seed,
    )
    middle = tuple(size // 2 for size in labels.shape)
    slices = (
        labels[middle[0], :, :],
        labels[:, middle[1], :],
        labels[:, :, middle[2]],
    )

    figure, axes = plt.subplots(1, 3, figsize=(10, 4))
    for axis, image in enumerate(slices):
        axes[axis].imshow(
            image,
            cmap="tab10",
            vmin=-0.5,
            vmax=cfg.data.num_phases - 0.5,
            interpolation="nearest",
        )
        axes[axis].set_title(f"axis {axis}")
        axes[axis].axis("off")
    figure.suptitle(f"EMA checkpoint step={step}, seed={seed}")
    figure.tight_layout()
    plt.show()
    print(f"checkpoint={Path(checkpoint).resolve()}")
    print(f"step={step} seed={seed} shape={tuple(labels.shape)}")


if __name__ == "__main__":
    main()
