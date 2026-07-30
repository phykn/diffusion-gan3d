import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from src.data import (
    LabelPatchDataset,
    build_batch_stream,
    load_axis_paths,
    sample_volume_pair_slices,
)
from src.data.patch import random_crop, resize_labels


def _save_labels(path: Path, labels: np.ndarray) -> None:
    Image.fromarray(labels.astype(np.uint8)).save(path)


class LabelTransformTest(unittest.TestCase):
    def test_random_crop_uses_the_sampled_coordinates(self):
        labels = np.arange(36, dtype=np.uint8).reshape(6, 6)

        with patch("numpy.random.randint", side_effect=(1, 2)):
            cropped = random_crop(labels, 3)

        np.testing.assert_array_equal(cropped, labels[1:4, 2:5])

    def test_nearest_resize_preserves_categorical_labels(self):
        expected = np.array([[0, 1], [2, 0]], dtype=np.uint8)
        labels = np.repeat(np.repeat(expected, 2, axis=0), 2, axis=1)

        resized = resize_labels(labels, 2)

        np.testing.assert_array_equal(resized, expected)
        self.assertEqual(resized.dtype, np.uint8)

    def test_one_image_can_fill_a_replacement_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "labels.png"
            _save_labels(path, np.full((8, 8), 2, dtype=np.uint8))
            dataset = LabelPatchDataset(
                [path],
                crop_size=4,
                patch_size=4,
                num_phases=3,
            )
            stream = build_batch_stream(
                dataset,
                batch_size=3,
                num_workers=0,
                pin_memory=False,
            )

            batch = stream.next()

        self.assertEqual(batch.shape, torch.Size([3, 4, 4]))
        self.assertEqual(batch.dtype, torch.long)
        self.assertTrue(bool((batch == 2).all()))


class AxisDataTest(unittest.TestCase):
    def test_axis_folders_require_one_label_image_per_axis(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folders = {}
            for axis in range(3):
                folder = root / str(axis)
                folder.mkdir()
                _save_labels(
                    folder / f"axis_{axis}.png",
                    np.full((4, 4), axis, dtype=np.uint8),
                )
                (folder / "ignored.txt").write_text("not an image", encoding="utf-8")
                folders[axis] = folder

            paths = load_axis_paths(folders)

        self.assertEqual(set(paths), {0, 1, 2})
        for axis in range(3):
            self.assertEqual(
                tuple(path.name for path in paths[axis]),
                (f"axis_{axis}.png",),
            )

    def test_volume_pairs_always_use_matching_volume_and_plane_coordinates(self):
        previous = torch.arange(
            2 * 3 * 4 * 4 * 4,
            dtype=torch.float32,
        ).reshape(2, 3, 4, 4, 4)
        current = previous + 10_000.0

        for axis in range(3):
            with self.subTest(axis=axis):
                random = torch.Generator().manual_seed(100 + axis)
                previous_slices, current_slices = sample_volume_pair_slices(
                    previous,
                    current,
                    axis=axis,
                    count=7,
                    generator=random,
                )

                self.assertEqual(previous_slices.shape, current_slices.shape)
                self.assertTrue(
                    torch.equal(
                        current_slices - previous_slices,
                        torch.full_like(previous_slices, 10_000.0),
                    )
                )


if __name__ == "__main__":
    unittest.main()
