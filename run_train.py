import argparse
from datetime import datetime
from pathlib import Path

import torch

from src.build import build_trainer
from src.utils import load_yaml, save_yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "train.yaml"
RUN_ROOT = Path(__file__).resolve().parent / "run"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    train = cfg["train"]
    for name in (
        "generator",
        "critic_0",
        "critic_1",
        "critic_2",
        "critic_c",
    ):
        path = train.get(name)
        if path is not None:
            print(f"Loading {name} checkpoint: {path}")
    trainer = build_trainer(cfg, device)
    run_dir = make_run_dir(RUN_ROOT)
    save_yaml(run_dir / "train.yaml", cfg)
    trainer.fit(
        steps=cfg["train"]["total_steps"],
        save_every=cfg["train"]["save_every_steps"],
        run_dir=run_dir,
        checkpoint_every=cfg["train"].get("checkpoint_every_steps"),
    )


def make_run_dir(root: Path) -> Path:
    name = datetime.now().astimezone().strftime("%m%d%H%M")
    for sequence in range(1, 100):
        suffix = "" if sequence == 1 else f"{sequence:02d}"
        run_dir = root / f"{name}{suffix}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return run_dir
    raise FileExistsError(f"too many runs already exist for minute {name}.")


if __name__ == "__main__":
    main()
