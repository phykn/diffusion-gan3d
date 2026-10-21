from collections.abc import Mapping, Sequence

import numpy as np


def summarize(
    rows: Sequence[Mapping[str, object]],
    conditions: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, object]]:
    """Per-condition mean and sample standard deviation, skipping missing values."""
    result = []
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        summary: dict[str, object] = {
            "condition": condition,
            "samples": len(selected),
        }
        for metric in metrics:
            values = np.asarray(
                [row[metric] for row in selected if row[metric] is not None],
                dtype=float,
            )
            summary[f"{metric}_mean"] = float(values.mean()) if values.size else None
            summary[f"{metric}_std"] = (
                float(values.std(ddof=1)) if values.size > 1 else None
            )
        result.append(summary)
    return result
