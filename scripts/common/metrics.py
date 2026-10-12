import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_metrics(log_dir: str | Path) -> list[dict]:
    """Read all TensorBoard scalars by step, without reservoir sampling."""
    events = EventAccumulator(str(log_dir), size_guidance={"scalars": 0}).Reload()
    rows = {}
    for tag in events.Tags()["scalars"]:
        for event in events.Scalars(tag):
            row = rows.setdefault(event.step, {"step": event.step})
            row[tag] = None if math.isnan(event.value) else event.value
    return [rows[step] for step in sorted(rows)]
