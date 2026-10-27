import runpy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def test_hr_preview_only_resizes_displayed_planes(tmp_path, monkeypatch):
    script = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/check/06_hr.py")
    )
    low = (torch.arange(3 * 5 * 7).reshape(3, 5, 7) % 3).long()
    high = torch.zeros(8, 12, 16, dtype=torch.uint8)
    reference = F.interpolate(low[None, None].float(), size=high.shape, mode="nearest")[
        0, 0
    ]
    interpolate = F.interpolate
    calls = []

    def resize_plane(values, *args, **kwargs):
        assert values.ndim == 4, "preview must not allocate a full HR volume"
        calls.append(values.shape)
        return interpolate(values, *args, **kwargs)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    show = Axes.imshow
    images = []

    def capture_image(panel, values, *args, **kwargs):
        images.append(np.asarray(values).copy())
        return show(panel, values, *args, **kwargs)

    monkeypatch.setattr(F, "interpolate", resize_plane)
    monkeypatch.setattr(Axes, "imshow", capture_image)
    path = tmp_path / "comparison.png"
    script["comparison"](low, high, 3, path, show=False)
    assert path.is_file()
    assert len(calls) == 3
    for axis in range(3):
        np.testing.assert_array_equal(
            images[3 + axis], reference.select(axis, high.shape[axis] // 2).numpy()
        )
