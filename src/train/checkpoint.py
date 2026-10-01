import os
import re
from pathlib import Path

_NAME = re.compile(r"step_(\d+)_(\d+)\.pt")


def complete_checkpoint(path: Path) -> None:
    """Publish completion only after the checkpoint has been flushed and closed."""
    with path.with_suffix(".complete").open("x", encoding="ascii") as file:
        file.write(f"{path.stat().st_size}\n")
        file.flush()
        os.fsync(file.fileno())


def _is_complete(path: Path) -> bool:
    try:
        return path.with_suffix(".complete").read_text(encoding="ascii") == (
            f"{path.stat().st_size}\n"
        )
    except (FileNotFoundError, UnicodeError):
        return False


def resolve_checkpoint(value: str | Path) -> Path:
    """Find the latest completed step, or validate an explicit completed file."""
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        root = path / "checkpoints" if (path / "checkpoints").is_dir() else path
        candidates = []
        for candidate in root.glob("step_*.pt"):
            match = _NAME.fullmatch(candidate.name)
            if match:
                candidates.append((tuple(map(int, match.groups())), candidate))
        for _, candidate in sorted(candidates, reverse=True):
            if _is_complete(candidate):
                return candidate
        raise FileNotFoundError(f"No completed training checkpoint found: {value}")
    if not path.is_file():
        raise FileNotFoundError(f"No completed training checkpoint found: {value}")
    if not _is_complete(path):
        raise ValueError(f"Training checkpoint is incomplete: {path}")
    return path
