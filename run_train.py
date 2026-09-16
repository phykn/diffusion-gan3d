import argparse
import copy
from datetime import datetime
from pathlib import Path

import torch

from src.build.trainer import build_trainer
from src.config import load_train_config, normalize_train_config, save_yaml
from src.train.run import run_train
from src.train.state import resume_training

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "train" / "low_res.yaml"
RUN_ROOT = Path(__file__).resolve().parent / "run"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--steps", type=int, help="Total target steps, including completed steps."
    )
    parser.add_argument("--data", type=Path, help="Data YAML; overrides config.data.")
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "directory for this run (created as a new directory; "
            "default: automatic timestamp under run/)"
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    payload = None
    if args.resume:
        if args.config is not None or args.data is not None:
            raise ValueError(
                "--resume uses saved settings; only --steps may override them."
            )
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        if payload.get("format") != "diffusion-gan3d.lr.train.v4":
            raise ValueError("--resume requires an LR training checkpoint.")
        cfg = normalize_train_config(payload["config"])
    else:
        cfg = load_train_config(args.config or DEFAULT_CONFIG, data=args.data)
    if args.steps is not None:
        cfg["train"]["total_steps"] = args.steps
    start = 0 if payload is None else payload["step"]
    if cfg["train"]["total_steps"] <= start:
        raise ValueError("total steps must exceed completed steps.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    build_cfg = copy.deepcopy(cfg)
    if payload:
        build_cfg["train"]["initial_weights"] = None
    trainer = build_trainer(build_cfg, device)
    cfg["data"] = trainer.cfg["data"]
    trainer.cfg = cfg
    if payload:
        resume_training(trainer, payload)
    if args.run_dir is None:
        run_dir = make_run_dir(RUN_ROOT)
    else:
        run_dir = args.run_dir.expanduser()
        run_dir.mkdir(parents=True, exist_ok=False)
    save_yaml(run_dir / "train.yaml", cfg)
    run_train(
        trainer,
        steps=cfg["train"]["total_steps"],
        save_every=cfg["train"]["weights_every_steps"],
        run_dir=run_dir,
        checkpoint_every=cfg["train"].get("archive_every_steps"),
        start_step=start,
    )


def make_run_dir(root: Path) -> Path:
    name = datetime.now().astimezone().strftime("%m%d%H%M")
    sequence = 1
    while True:
        suffix = "" if sequence == 1 else f"{sequence:02d}"
        run_dir = root / f"{name}{suffix}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            sequence += 1
            continue
        return run_dir


if __name__ == "__main__":
    main()
