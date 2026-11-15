import signal
from contextlib import contextmanager
from threading import Event, current_thread, main_thread

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config.build import to_dict
from src.config.io import save_yaml
from src.console import paint, warning
from src.export import write_export
from src.train.checkpoint import FORMAT, VERSION, save_checkpoint
from src.train.metrics import selection_metrics, write_metrics
from src.train.selection import (
    CheckpointSelection,
    SRCheckpointSelection,
    build_reference,
)

DONE = ".done"


class TrainingRun:
    """One stage's step loop in its run folder; Ctrl+C saves at a step boundary."""

    def __init__(self, trainer, root, payload=None):
        self.trainer = trainer
        self.root = root
        self.path_maps = [] if payload is None else payload["path_maps"]
        self.archived_step = None
        config = trainer.config
        reference = build_reference(trainer.samplers)
        samples = config.selection.samples
        if config.stage == "lr":
            self.selection = CheckpointSelection(
                reference, samples, config.conditioning.height_enabled
            )
        else:
            self.selection = SRCheckpointSelection.from_bank(
                reference, samples, trainer.bank
            )
        if payload is not None:
            trainer.load_state_dict(payload)
            self.selection.load_state_dict(payload["selection"])

    def train(self):
        trainer = self.trainer
        config = trainer.config
        total = config.train.total_steps
        start = trainer.completed_steps
        if not 0 <= start < total:
            raise ValueError("train.total_steps must exceed the completed steps.")
        self.root.mkdir(parents=True, exist_ok=False)
        save_yaml(self.root / "train.yaml", config)
        self.selection.restore(trainer, self.root)
        with (
            deferred_sigint() as stop,
            SummaryWriter(self.root / "tensorboard") as writer,
            tqdm(
                range(start, total),
                desc=f"{config.stage.upper()} train",
                dynamic_ncols=True,
            ) as bar,
        ):
            # True while trainer and saved files agree, so an interrupt may save.
            at_boundary = True
            try:
                for step in bar:
                    if stop.is_set():
                        raise KeyboardInterrupt
                    at_boundary = False
                    metrics = trainer.step(step)
                    at_boundary = True
                    write_metrics(writer, step + 1, metrics)
                    writer.flush()
                    at_boundary = False
                    self.save(step + 1, writer)
                    at_boundary = True
                    if stop.is_set():
                        raise KeyboardInterrupt
            except KeyboardInterrupt:
                if at_boundary:
                    if self.archived_step != trainer.completed_steps:
                        self.archive()
                    self.save_weights()
                raise
        (self.root / DONE).touch()
        return self.root

    def save(self, done, writer):
        """Score and export every weights_every_steps; checkpoint every
        archive_every_steps; both at the last step."""
        train = self.trainer.config.train
        last = done == train.total_steps
        weights = last or done % train.weights_every_steps == 0
        if weights:
            self.score(writer)
            writer.flush()
        if last or done % train.archive_every_steps == 0:
            self.archive()
        if weights:
            self.save_weights()

    def score(self, writer):
        entry = self.selection.update(self.trainer, self.root)
        best = self.selection.best()
        write_metrics(writer, entry["step"], selection_metrics(entry, best))
        line = f"step {entry['step']:>7}  structure {entry['structure']:.3f}"
        if best["step"] == entry["step"]:
            tqdm.write(paint(f"{line}  * new best", "green"))
        else:
            best_text = f"best {best['structure']:.3f} @ {best['step']}"
            tqdm.write(f"{line}  {paint(best_text, 'dim')}")

    def archive(self):
        trainer = self.trainer
        payload = {
            "format": FORMAT,
            "version": VERSION,
            "stage": trainer.config.stage,
            "step": trainer.completed_steps,
            "config": to_dict(trainer.config),
            "selection": self.selection.state_dict(),
            "path_maps": self.path_maps,
            **trainer.state_dict(),
        }
        save_checkpoint(payload, self.root, self.current_export())
        self.archived_step = trainer.completed_steps

    def save_weights(self):
        payload = (
            self.selection.best_payload(self.trainer)
            if self.selection.best()
            else self.current_export()
        )
        write_export(self.root / "generator.pt", payload)

    def current_export(self):
        trainer = self.trainer
        return trainer.export_payload(trainer.completed_steps, trainer.ema.state_dict())


@contextmanager
def deferred_sigint():
    """Let the current step finish on Ctrl+C; a second one stops at once."""
    requested = Event()

    def interrupt(*_):
        if requested.is_set():
            raise KeyboardInterrupt
        requested.set()
        warning("Stopping after the current step; press Ctrl+C again to stop now.")

    in_main_thread = current_thread() is main_thread()
    if in_main_thread:
        previous = signal.signal(signal.SIGINT, interrupt)
    try:
        yield requested
    finally:
        if in_main_thread:
            signal.signal(signal.SIGINT, previous)
