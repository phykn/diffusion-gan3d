import numpy as np
import pytest

from src.evaluate.tilt import component_slopes, tilt_balance


def tilted_disc(slope: float, size: int = 32, radius: int = 10) -> np.ndarray:
    z, y, x = np.indices((size,) * 3) - size // 2
    disc = (np.abs(z - slope * y) <= 1) & (y**2 + x**2 <= radius**2)
    return disc.astype(np.uint8)


def test_component_slopes_measure_signed_disc_tilt():
    slopes = component_slopes(tilted_disc(0.3), 1)
    assert slopes["zy"] == pytest.approx([0.3], abs=0.02)
    assert slopes["zx"] == pytest.approx([0.0], abs=0.02)
    assert slopes["yx"] == pytest.approx([0.0], abs=0.02)


def test_tilt_balance_is_signed_and_cancels_for_mirror_pairs():
    volume = tilted_disc(0.3)
    mirror = volume[:, ::-1].copy()
    assert tilt_balance([volume], 1)["zy"]["balance"] == 1
    assert tilt_balance([mirror], 1)["zy"]["balance"] == -1
    pooled = tilt_balance([volume, mirror], 1)["zy"]
    assert pooled["count"] == 2
    assert pooled["balance"] == 0
    assert pooled["mean"] == pytest.approx(0.0, abs=1e-9)


def test_tilt_balance_skips_specks_and_empty_phases():
    volume = tilted_disc(0.3)
    volume[0, 0, 0] = 1
    assert tilt_balance([volume], 1)["zy"]["count"] == 1
    assert tilt_balance([volume], 2)["zy"] == {
        "count": 0,
        "positive": 0.0,
        "negative": 0.0,
        "balance": 0.0,
        "mean": 0.0,
        "sem": 0.0,
    }
    with pytest.raises(ValueError, match="3D"):
        component_slopes(volume[0], 1)
