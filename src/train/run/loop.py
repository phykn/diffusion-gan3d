import signal
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import Event, current_thread, main_thread

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config.files import save_yaml
from src.storage import save_model, torch_save
from src.train.checkpoint import complete_checkpoint
from src.train.metrics import write_metrics
from src.train.state import export_sr, save_sr_training, save_training
from src.train.trainer import Trainer


def check_run_dir(run_dir: Path | None) -> None:
    """Reject an existing explicit run directory before any expensive setup."""
    if run_dir is not None and run_dir.expanduser().exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")


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
    save_generator(trainer, root / "generator.pt", stage)
    for group, critic in trainer.critics.items():
        save_model(root / f"critic_{group}.pt", critic)
    if trainer.connectivity_critic is not None:
        save_model(root / "critic_c.pt", trainer.connectivity_critic)
    if trainer.seam_critic is not None:
        save_model(root / "critic_s.pt", trainer.seam_critic)


def save_generator(trainer, path, stage, overwrite=True):
    if stage == "sr":
        export_sr(trainer, path, overwrite=overwrite)
    else:
        torch_save(trainer.ema_denoiser.state_dict(), path, overwrite=overwrite)


def save_checkpoint(trainer, root, stage):
    folder = root / "checkpoints"
    path = folder / f"step_{trainer.completed_steps:08d}.pt"
    generator = folder / f"generator_{trainer.completed_steps:08d}.pt"
    for target in (path, generator, path.with_suffix(".complete")):
        if target.exists():
            raise FileExistsError(target)
    folder.mkdir(parents=True, exist_ok=True)
    if stage == "sr":
        save_sr_training(trainer, path)
    else:
        save_training(trainer, path)
    save_generator(trainer, generator, stage, overwrite=False)
    complete_checkpoint(path)


@contextmanager
def deferred_sigint():
    """Let a training step finish before handling a terminal interrupt.

    A second interrupt stops at once; run_train then saves nothing mid-step.
    """
    requested = Event()

    def interrupt(*_):
        if requested.is_set():
            raise KeyboardInterrupt
        requested.set()
        tqdm.write("Stopping after the current step; press Ctrl+C again to stop now.")

    in_main_thread = current_thread() is main_thread()
    if in_main_thread:
        previous = signal.signal(signal.SIGINT, interrupt)
    try:
        yield requested
    finally:
        if in_main_thread:
            signal.signal(signal.SIGINT, previous)


def run_train(
    trainer: Trainer,
    steps: int,
    save_every: int,
    run_dir: str | Path,
    checkpoint_every: int | None = None,
    start_step: int = 0,
    before_step: Callable[[int], None] | None = None,
    config: dict | None = None,
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
    save_yaml(root / "train.yaml", trainer.cfg if config is None else config)
    weights = root / "generator.pt"
    with (
        deferred_sigint() as stop,
        SummaryWriter(root / "tensorboard") as writer,
        tqdm(
            range(start_step, steps),
            desc="SR train" if stage == "sr" else "Diffusion GAN3D",
            dynamic_ncols=True,
        ) as bar,
    ):
        at_boundary = True
        checkpoint_step = None
        try:
            for step in bar:
                if stop.is_set():
                    raise KeyboardInterrupt
                at_boundary = False
                if before_step is not None:
                    before_step(step)
                metrics = trainer.step(step)
                at_boundary = True
                done = step + 1
                write_metrics(writer, done, metrics)
                writer.flush()
                if done == steps or (
                    checkpoint_every is not None and done % checkpoint_every == 0
                ):
                    save_checkpoint(trainer, root, stage)
                    checkpoint_step = trainer.completed_steps
                if done % save_every == 0 or done == steps:
                    save_weights(trainer, root, stage)
                if stop.is_set():
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            # A direct exception inside a step may leave partially updated
            # optimizers. Never overwrite a safe checkpoint with that state.
            if at_boundary:
                if checkpoint_step != trainer.completed_steps:
                    save_checkpoint(trainer, root, stage)
                save_weights(trainer, root, stage)
            raise
    return weights
