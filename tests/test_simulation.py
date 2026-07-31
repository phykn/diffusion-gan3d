import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from src.build import generate_data
from src.simul.config import GeometryConfig, OutputConfig, SimulationConfig

GEOMETRY = GeometryConfig(
    size=20,
    big_radius=3,
    small_radius=2,
    big_fraction=0.15,
    small_fraction=0.05,
    big_elongation=1.0,
)


def _config(root: Path, count: int) -> SimulationConfig:
    return SimulationConfig(
        output=OutputConfig(data_dir=root, count=count),
        geometry=GEOMETRY,
    )


class SimulationExportTest(unittest.TestCase):
    def test_exports_only_volumes_and_axis_slices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "generated"
            result = generate_data(_config(root, 4))
            path = result.volumes[0]
            volume = tifffile.imread(path)
            stem = path.stem
            with tifffile.TiffFile(path) as file:
                axes = file.series[0].axes

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
        self.assertEqual(axes, "ZYX")
        np.testing.assert_array_equal(slices[0], volume[3])
        np.testing.assert_array_equal(slices[1], volume[:, 3])
        np.testing.assert_array_equal(slices[2], volume[:, :, 3])

    def test_separate_exports_write_the_requested_volumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = generate_data(_config(Path(tmp) / "first", 3))
            second = generate_data(_config(Path(tmp) / "second", 3))

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
                generate_data(_config(root, 1))

            self.assertFalse((root / "volumes").exists())
            self.assertFalse((root / "slices" / "0").exists())
            self.assertFalse((root / "slices" / "2").exists())


if __name__ == "__main__":
    unittest.main()
