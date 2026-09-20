import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest
import tifffile
import yaml
from PIL import Image

from simul.src.export import generate
from simul.src.geometry import pack, place, radius_profile

GEOMETRY = {
    "size": 20,
    "big_radius": 3,
    "small_radius": 2,
    "big_vf": 0.15,
    "small_vf": 0.05,
    "big_elongation": 1.0,
}


def _config(root: Path, count: int) -> dict:
    return {
        "output": {"data_dir": root, "count": count},
        "geometry": GEOMETRY,
    }


class SimulationExportTest(unittest.TestCase):
    def test_exports_only_volumes_and_axis_slices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "generated"
            result = generate(_config(root, 4))
            path = result.volumes[0]
            volume = tifffile.imread(path)
            stem = path.stem

            slices = {}
            for axis in range(3):
                target = next(
                    value
                    for value in result.slices[axis]
                    if value.name == f"{stem}_{axis}_003.png"
                )
                with Image.open(target) as image:
                    slices[axis] = np.asarray(image)
            directories = sorted(
                path.relative_to(root).as_posix()
                for path in root.rglob("*")
                if path.is_dir()
            )

        self.assertEqual(len(result.volumes), 4)
        self.assertTrue(all(path.suffix == ".tiff" for path in result.volumes))
        self.assertEqual(
            [len(result.slices[axis]) for axis in range(3)],
            [80, 80, 80],
        )
        self.assertEqual(
            directories,
            ["slices", "slices/0", "slices/1", "slices/2", "volumes"],
        )
        self.assertEqual(volume.ndim, 3)
        np.testing.assert_array_equal(slices[0], volume[3])
        np.testing.assert_array_equal(slices[1], volume[:, 3])
        np.testing.assert_array_equal(slices[2], volume[:, :, 3])

    def test_separate_exports_write_the_requested_volumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = generate(_config(Path(tmp) / "first", 3))
            second = generate(_config(Path(tmp) / "second", 3))

            self.assertEqual(len(first.volumes), 3)
            self.assertEqual(len(second.volumes), 3)
            self.assertTrue(all(path.is_file() for path in first.volumes))
            self.assertTrue(all(path.is_file() for path in second.volumes))

    def test_nonempty_output_is_rejected_before_creating_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "generated"
            occupied = root / "slices" / "1"
            occupied.mkdir(parents=True)
            (occupied / "existing.png").touch()

            with self.assertRaises(FileExistsError):
                generate(_config(root, 1))

            self.assertFalse((root / "volumes").exists())
            self.assertFalse((root / "slices" / "0").exists())
            self.assertFalse((root / "slices" / "2").exists())

    def test_invalid_geometry_is_rejected_before_writing_output(self):
        for field, value in (
            ("size", 0),
            ("big_radius", 0),
            ("big_vf", -0.1),
            ("big_elongation", 0.0),
        ):
            with self.subTest(field=field):
                geometry = dict(GEOMETRY)
                geometry[field] = value
                with self.assertRaises(ValueError):
                    pack(**geometry)


if __name__ == "__main__":
    unittest.main()


def test_radius_gradient_uses_exported_height_and_clamps_context():
    profile = radius_profile(5, 3, [0.5, 1.5])
    np.testing.assert_allclose(
        profile, [0.5, 0.5, 0.5, 0.5, 0.75, 1, 1.25, 1.5, 1.5, 1.5, 1.5]
    )


def test_placed_particle_axes_follow_height_without_overlap():
    volume = np.zeros((40,) * 3, dtype=np.uint8)
    occupied = np.zeros_like(volume, dtype=bool)
    particles = []
    scales = radius_profile(32, 4, [0.6, 1.4])
    for label in (2, 1):
        place(volume, occupied, particles, label, 2500, 3.0, 2.0, 2.0, scales)
    assert len(particles) > 10
    # Geometry is checked for every accepted center, not inferred from random
    # section fluctuations. Reconstruct the occupancy to check non-overlap.
    from simul.src.geometry import make_offsets

    reconstructed = np.zeros_like(volume, dtype=int)
    for particle in particles:
        center, axes = np.asarray(particle.center), np.asarray(particle.axes)
        label = volume[particle.center]
        radius = float(np.prod(axes) ** (1 / 3))
        assert radius == pytest.approx((3 if label == 2 else 2) * scales[center[0]])
        reconstructed[tuple((make_offsets(axes) + center).T)] += 1
    assert reconstructed.max() == 1
    np.testing.assert_array_equal(reconstructed.astype(bool), occupied)


@pytest.mark.parametrize("gradient", [[0, 1], [-1, 2], [1], [1, float("inf")]])
def test_invalid_gradient_is_rejected_before_export(tmp_path, gradient):
    cfg = _config(tmp_path / "output", 1)
    cfg["geometry"] = {**GEOMETRY, "radius_gradient": gradient}
    with pytest.raises(ValueError, match="radius_gradient"):
        generate(cfg)
    assert not (tmp_path / "output").exists()


def test_gradient_export_records_truth_and_physical_axis(tmp_path):
    cfg = _config(tmp_path / "output", 1)
    cfg["geometry"] = {**GEOMETRY, "radius_gradient": [0.7, 1.3]}
    result = generate(cfg)
    metadata = yaml.safe_load((tmp_path / "output" / "simulation.yaml").read_text())
    assert metadata["thickness_axis"] == "z"
    assert metadata["geometry"]["radius_gradient"] == [0.7, 1.3]
    truth = tifffile.imread(result.volumes[0])
    with Image.open(result.slices[1][8]) as image:
        np.testing.assert_array_equal(image, truth[:, 8, :])
