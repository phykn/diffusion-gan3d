import argparse
from datetime import datetime
from pathlib import Path

import torch

from src.build import build_trainer
from src.train.config import load_config
from src.utils import save_yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "train.yaml"
RUN_ROOT = Path(__file__).resolve().parent / "run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    checkpoint = cfg.train.checkpoint
    for name in ("model", "critic_0", "critic_1", "critic_2"):
        path = getattr(checkpoint, name)
        if path is not None:
            print(f"Loading {name} checkpoint: {path}")
    trainer = build_trainer(cfg, device=device)
    run_dir = _make_run_dir(RUN_ROOT)
    save_yaml(run_dir / "train.yaml", cfg.as_dict())
    trainer.fit(
        steps=cfg.train.steps,
        save_every=cfg.train.save_every_steps,
        critic_warmup_steps=cfg.train.critic_warmup_steps,
        run_dir=run_dir,
    )


def _make_run_dir(root: Path) -> Path:
    name = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


if __name__ == "__main__":
    main()
