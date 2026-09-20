import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from src.build.data import build_datasets, build_stream
from src.config.data import get_domains
from src.data.augment import crop_images
from src.data.dataset import RealDataset
from src.data.loader import FolderBatchSampler
from src.data.slice import sample_pairs


def _save_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image.astype(np.uint8)).save(path)


class LabelTransformTest(unittest.TestCase):
    def test_random_crop_returns_source_origin(self):
        image = np.arange(36, dtype=np.uint8).reshape(6, 6)
        dataset = RealDataset([["unused"]], crop_size=3, patch_size=3, num_phases=36)
        with patch("numpy.random.randint", side_effect=(1, 2)):
            cropped, origin = dataset.crop_with_origin(image)
        np.testing.assert_array_equal(cropped, image[1:4, 2:5])
        self.assertEqual(origin, (1, 2))

    def test_dataset_resizes_phase_channels_without_losing_fractional_occupancy(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "image.png"
            image = np.tile(np.array([0, 1], dtype=np.uint8), (8, 4))
            _save_image(path, image)
            dataset = RealDataset([[path]], crop_size=4, patch_size=2, num_phases=2)
            with patch("numpy.random.randint", side_effect=(2, 3)):
                actual = dataset[path]
        torch.testing.assert_close(actual, torch.full((2, 2, 2), 0.5))

    def test_crop_larger_than_the_image_is_rejected(self):
        dataset = RealDataset([["unused"]], crop_size=4, patch_size=4, num_phases=2)
        with self.assertRaisesRegex(ValueError, "crop size must fit"):
            dataset.crop_with_origin(np.zeros((3, 8), dtype=np.uint8))

    def test_partial_crop_option_is_not_supported(self):
        with self.assertRaises(TypeError):
            RealDataset([["unused"]], 8, 4, 2, allow_part=True)

    def test_one_image_can_fill_a_replacement_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "image.png"
            _save_image(path, np.full((8, 8), 2, dtype=np.uint8))
            dataset = RealDataset([[path]], crop_size=4, patch_size=4, num_phases=3)
            stream = build_stream(
                dataset, batch_size=3, num_workers=0, pin_memory=False
            )
            batch = stream.next()
        self.assertEqual(batch.shape, torch.Size([3, 3, 4, 4]))
        self.assertEqual(batch.dtype, torch.float32)
        self.assertTrue(bool((batch[:, 2] == 1).all()))
        self.assertEqual(batch[:, :2].count_nonzero(), 0)


class AxisDataTest(unittest.TestCase):
    def test_tensor_crop_uses_rectangular_shape_and_random_coordinates(self):
        images = torch.arange(2 * 5 * 7).reshape(2, 5, 7)

        with patch(
            "torch.randint",
            side_effect=(torch.tensor([1, 2]), torch.tensor([3, 1])),
        ):
            actual = crop_images(images, (3, 4))

        expected = torch.stack(
            (
                images[0, 1:4, 3:7],
                images[1, 2:5, 1:5],
            )
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_volume_pairs_always_use_matching_volume_and_plane_coordinates(self):
        previous = torch.arange(
            2 * 3 * 4 * 4 * 4,
            dtype=torch.float32,
        ).reshape(2, 3, 4, 4, 4)
        current = previous + 10_000.0

        for axis in range(3):
            with self.subTest(axis=axis):
                previous_slices, current_slices = sample_pairs(
                    previous,
                    current,
                    axis,
                    count=7,
                    crop_shape=3,
                )

                self.assertEqual(previous_slices.shape, current_slices.shape)
                self.assertEqual(previous_slices.shape[-2:], (3, 3))
                self.assertTrue(
                    torch.equal(
                        current_slices - previous_slices,
                        torch.full_like(previous_slices, 10_000.0),
                    )
                )


class DomainDataTest(unittest.TestCase):
    def test_domains_may_omit_axes_when_the_global_union_is_complete(self):
        domains = get_domains(
            {
                "domains": {
                    0: {"xy": ["axis_0"]},
                    1: {"xz": ["axis_1"], "yz": ["axis_2"]},
                }
            }
        )

        self.assertEqual(set(domains[0]), {0})
        self.assertEqual(set(domains[1]), {1, 2})

    def test_domains_may_collectively_provide_only_one_axis(self):
        domains = get_domains({"domains": {0: {"xy": ["axis_0"]}}})

        self.assertEqual(set(domains[0]), {0})

    def test_build_datasets_accepts_one_axis(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "axis_0"
            folder.mkdir()
            _save_image(folder / "sample.png", np.zeros((4, 4), dtype=np.uint8))
            cfg = {
                "data": {
                    "domains": {0: {"xy": [folder]}},
                    "crop_size": 4,
                    "num_phases": 2,
                    "lo_res_size": 4,
                }
            }

            datasets = build_datasets(cfg)

        self.assertEqual(set(datasets[0]), {0})

    def test_domain_datasets_keep_axis_folders_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            domains = {}
            for domain in range(2):
                axes = {}
                for axis in range(3):
                    folder = root / str(domain) / str(axis)
                    folder.mkdir(parents=True)
                    value = 3 * domain + axis
                    _save_image(
                        folder / "sample.png",
                        np.full((4, 4), value, dtype=np.uint8),
                    )
                    axes[("xy", "xz", "yz")[axis]] = [folder]
                domains[domain] = axes
            cfg = {
                "data": {
                    "domains": domains,
                    "crop_size": 4,
                    "num_phases": 6,
                    "lo_res_size": 4,
                }
            }

            datasets = build_datasets(cfg)
            samples = {
                (domain, axis): datasets[domain][axis][
                    datasets[domain][axis].path_groups[0][0]
                ]
                for domain in range(2)
                for axis in range(3)
            }

        self.assertEqual(set(datasets), {0, 1})
        for domain in range(2):
            for axis in range(3):
                expected = 3 * domain + axis
                self.assertTrue(
                    bool((samples[domain, axis].argmax(0) == expected).all())
                )

    def test_domain_ids_are_contiguous_and_start_at_zero(self):
        with self.assertRaisesRegex(ValueError, "contiguous"):
            get_domains({"domains": {1: {}}})

    def test_training_rejects_crops_larger_than_the_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folders = {}
            for axis in range(3):
                folder = root / str(axis)
                folder.mkdir()
                _save_image(folder / "sample.png", np.zeros((2, 4), dtype=np.uint8))
                folders[("xy", "xz", "yz")[axis]] = [folder]
            cfg = {
                "data": {
                    "domains": {0: folders},
                    "crop_size": 4,
                    "num_phases": 2,
                    "lo_res_size": 4,
                }
            }

            datasets = build_datasets(cfg)
            dataset = datasets[0][0]
            with self.assertRaisesRegex(ValueError, "crop size must fit"):
                dataset[dataset.path_groups[0][0]]

    def test_resolution_stream_builds_each_batch_from_one_folder(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folders = {}
            for axis in range(3):
                thin = root / str(axis) / "thin"
                square = root / str(axis) / "square"
                thin.mkdir(parents=True)
                square.mkdir()
                for index in range(2):
                    _save_image(
                        thin / f"thin_{index}.png",
                        np.full((4, 4), 1, dtype=np.uint8),
                    )
                for index in range(4):
                    _save_image(
                        square / f"square_{index}.png",
                        np.full((4, 4), 2, dtype=np.uint8),
                    )
                folders[("xy", "xz", "yz")[axis]] = [thin, square]
            cfg = {
                "data": {
                    "domains": {0: folders},
                    "crop_size": 4,
                    "num_phases": 3,
                    "lo_res_size": 4,
                }
            }

            dataset = build_datasets(cfg)[0][0]
            sampler = FolderBatchSampler(dataset, batch_size=3)
            with patch(
                "torch.randint",
                side_effect=(torch.tensor(1), torch.tensor([0, 1, 2])),
            ) as randint:
                paths = next(iter(sampler))
            self.assertEqual(randint.call_args_list[0].args, (2, ()))
            stream = build_stream(
                dataset,
                batch_size=3,
                num_workers=0,
                pin_memory=False,
            )
            batch = stream.next()

        self.assertEqual(tuple(len(group) for group in dataset.path_groups), (2, 4))
        self.assertEqual(paths, list(dataset.path_groups[1][:3]))
        self.assertIn(
            batch.shape,
            (torch.Size([3, 3, 4, 4]),),
        )


if __name__ == "__main__":
    unittest.main()
