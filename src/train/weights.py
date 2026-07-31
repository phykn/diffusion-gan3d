import os
from pathlib import Path
from uuid import uuid4

import torch
from torch import nn

WEIGHTS_NAME = "model.pt"


def save_weights(
    run_dir: str | Path,
    model: nn.Module,
) -> Path:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch module.")

    root = Path(run_dir)
    path = root / WEIGHTS_NAME
    _save_atomic(model.state_dict(), path)
    return path


def _save_atomic(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        torch.save(data, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
