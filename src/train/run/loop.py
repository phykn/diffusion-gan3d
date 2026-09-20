import json
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config.files import save_yaml
from src.storage import save_model
from src.train.metrics import write_metrics
from src.train.sr import export_sr, save_sr_training
from src.train.state import describe_data, save_training
from src.train.trainer import Trainer


def make_run_dir(
    root: Path, stage: str, run_dir: Path | None = None, nickname: str = ""
) -> Path:
    if stage not in ("low_res", "sr"):
        raise ValueError("stage must be low_res or sr.")
    if run_dir is not None:
        path = run_dir.expanduser().resolve()
        path.mkdir(parents=True, exist_ok=False)
        return path
    name = datetime.now().astimezone().strftime("%m%d%H%M")
    label = f"_{nickname}" if nickname else ""
    sequence = 1
    while True:
        suffix = "" if sequence == 1 else f"_{sequence:02d}"
        path = root.resolve() / f"{name}{suffix}_{stage}{label}"
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            sequence += 1
            continue
        return path


def save_weights(trainer: Trainer, root: Path, stage: str = "low_res") -> None:
    root.mkdir(parents=True, exist_ok=True)
    if stage == "sr":
        export_sr(trainer, root / "generator.pt")
    else:
        save_model(root / "generator.pt", trainer.ema_denoiser)
    for group, critic in trainer.critics.items():
        save_model(root / f"critic_{group}.pt", critic)
    if trainer.connectivity_critic is not None:
        save_model(root / "critic_c.pt", trainer.connectivity_critic)


def run_train(
    trainer: Trainer,
    steps: int,
    save_every: int,
    run_dir: str | Path,
    checkpoint_every: int | None = None,
    start_step: int = 0,
    before_step: Callable[[int], None] | None = None,
) -> Path:
    for name, value in (
        ("steps", steps),
        ("save_every", save_every),
        ("checkpoint_every", checkpoint_every),
    ):
        if name == "checkpoint_every" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    root = Path(run_dir)
    if not 0 <= start_step < steps:
        raise ValueError("start_step must be less than total steps.")
    stage = trainer.cfg["stage"]
    root.mkdir(parents=True, exist_ok=True)
    save_yaml(root / "train.yaml", trainer.cfg)
    (root / "data_manifest.json").write_text(
        json.dumps(describe_data(trainer), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    weights = root / "generator.pt"
    with (
        SummaryWriter(root / "tensorboard") as writer,
        (root / "metrics.jsonl").open("w", encoding="utf-8") as log,
        tqdm(
            range(start_step, steps),
            desc="SR train" if stage == "sr" else "Diffusion GAN3D",
            dynamic_ncols=True,
        ) as bar,
    ):
        try:
            for step in bar:
                if before_step is not None:
                    before_step(step)
                metrics = trainer.step(step)
                done = step + 1
                write_metrics(writer, done, metrics)
                log.write(
                    json.dumps({"step": done, **asdict(metrics)}, allow_nan=False)
                    + "\n"
                )
                log.flush()
                if done % save_every == 0 or done == steps:
                    save_weights(trainer, root, stage)
                    if stage == "sr":
                        save_sr_training(trainer, root / "checkpoints" / "last.pt")
                    else:
                        save_training(root / "checkpoints" / "last.pt", trainer)
                if checkpoint_every is not None and done % checkpoint_every == 0:
                    checkpoint_root = root / "checkpoints" / f"step_{done:08d}"
                    save_weights(trainer, checkpoint_root, stage)
        except KeyboardInterrupt:
            save_weights(trainer, root, stage)
            raise
    return weights
