import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from src.build import build_datasets, build_stream, find_slices, get_domains
from src.dataset import FolderBatchSampler, SliceDataset
from src.train.engine import Trainer


def _save_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image.astype(np.uint8)).save(path)


class LabelTransformTest(unittest.TestCase):
    def test_random_crop_uses_the_sampled_coordinates(self):
        image = np.arange(36, dtype=np.uint8).reshape(6, 6)

        dataset = SliceDataset(["unused"], crop_size=3)

        with patch("numpy.random.randint", side_effect=(1, 2)):
            cropped = dataset.crop(image)

        np.testing.assert_array_equal(cropped, image[1:4, 2:5])

    def test_default_crop_size_is_64(self):
        dataset = SliceDataset(["unused"])

        self.assertEqual(dataset.crop_size, 64)
        self.assertEqual(dataset.patch_size, 64)

    def test_dataset_resizes_the_crop_to_patch_size(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "image.png"
            img = np.arange(64, dtype=np.uint8).reshape(8, 8)
            _save_image(path, img)
            dataset = SliceDataset([path], crop_size=4, patch_size=2)

            with patch("numpy.random.randint", side_effect=(2, 3)):
                actual = dataset[0]

        expected = np.asarray(
            Image.fromarray(img[2:6, 3:7]).resize(
                (2, 2),
                resample=Image.Resampling.NEAREST,
            )
        )
        np.testing.assert_array_equal(actual.numpy(), expected)

    def test_default_resize_keeps_the_existing_square_output(self):
        image = np.arange(32, dtype=np.uint8).reshape(4, 8)
        dataset = SliceDataset(["unused"], crop_size=8, patch_size=2)

        actual = dataset.resize(image)

        expected = np.asarray(
            Image.fromarray(image).resize(
                (2, 2),
                resample=Image.Resampling.NEAREST,
            )
        )
        np.testing.assert_array_equal(actual, expected)

    def test_partial_crop_preserves_aspect_ratio_at_the_configured_scale(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "image.png"
            img = np.arange(32, dtype=np.uint8).reshape(4, 8)
            _save_image(path, img)
            dataset = SliceDataset(
                [path],
                crop_size=8,
                patch_size=4,
                allow_partial_crop=True,
            )

            actual = dataset[0]

        expected = np.asarray(
            Image.fromarray(img).resize(
                (4, 2),
                resample=Image.Resampling.NEAREST,
            )
        )
        self.assertEqual(actual.shape, torch.Size([2, 4]))
        np.testing.assert_array_equal(actual.numpy(), expected)

    def test_partial_crop_is_rejected_by_default(self):
        image = np.zeros((3, 8), dtype=np.uint8)
        dataset = SliceDataset(["unused"], crop_size=4, patch_size=4)

        with self.assertRaisesRegex(ValueError, "crop size must fit"):
            dataset.crop(image)

    def test_partial_crop_flag_requires_a_boolean(self):
        with self.assertRaisesRegex(TypeError, "must be a boolean"):
            SliceDataset(["unused"], allow_partial_crop=1)

    def test_partial_crop_uses_a_random_window_on_the_long_axis(self):
        image = np.arange(48, dtype=np.uint8).reshape(4, 12)
        dataset = SliceDataset(
            ["unused"],
            crop_size=8,
            patch_size=8,
            allow_partial_crop=True,
        )

        with patch("numpy.random.randint", side_effect=(0, 3)):
            cropped = dataset.crop(image)

        np.testing.assert_array_equal(cropped, image[:, 3:11])

    def test_partial_crop_keeps_an_extremely_thin_image_nonempty(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "image.png"
            _save_image(path, np.arange(2, dtype=np.uint8).reshape(1, 2))
            dataset = SliceDataset(
                [path],
                crop_size=8,
                patch_size=2,
                allow_partial_crop=True,
            )

            actual = dataset[0]

        self.assertEqual(actual.shape, torch.Size([1, 1]))

    def test_dataset_reuses_decoded_images_across_random_crops(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "image.png"
            _save_image(path, np.arange(64, dtype=np.uint8).reshape(8, 8))
            dataset = SliceDataset([path], crop_size=4, patch_size=4)

            with patch.object(dataset, "decode", wraps=dataset.decode) as decode:
                first = dataset[0]
                second = dataset[0]

        self.assertEqual(decode.call_count, 1)
        self.assertEqual(first.shape, second.shape)

    def test_one_image_can_fill_a_replacement_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "image.png"
            _save_image(path, np.full((8, 8), 2, dtype=np.uint8))
            dataset = SliceDataset(
                [path],
                crop_size=4,
                patch_size=4,
            )
            stream = build_stream(
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
    def test_axis_folders_require_one_image_per_axis(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folders = {}
            for axis in range(3):
                folder = root / str(axis)
                folder.mkdir()
                _save_image(
                    folder / f"axis_{axis}.png",
                    np.full((4, 4), axis, dtype=np.uint8),
                )
                (folder / "ignored.txt").write_text("not an image", encoding="utf-8")
                folders[axis] = [folder]

            paths = find_slices(folders)

        self.assertEqual(set(paths), {0, 1, 2})
        for axis in range(3):
            self.assertEqual(
                tuple(path.name for path in paths[axis]),
                (f"axis_{axis}.png",),
            )

    def test_axis_can_combine_multiple_folders(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folders = {}
            for axis in range(3):
                first = root / str(axis) / "first"
                second = root / str(axis) / "second"
                first.mkdir(parents=True)
                second.mkdir()
                _save_image(first / "a.png", np.full((4, 4), axis, dtype=np.uint8))
                _save_image(second / "b.tif", np.full((4, 4), axis, dtype=np.uint8))
                folders[axis] = [first, second]

            paths = find_slices(folders)

        for axis in range(3):
            self.assertEqual(
                tuple(path.name for path in paths[axis]), ("a.png", "b.tif")
            )

    def test_tensor_crop_uses_rectangular_shape_and_random_coordinates(self):
        images = torch.arange(2 * 5 * 7).reshape(2, 5, 7)

        with patch(
            "torch.randint",
            side_effect=(torch.tensor([1, 2]), torch.tensor([3, 1])),
        ):
            actual = Trainer.crop_images(images, (3, 4))

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
        trainer = object.__new__(Trainer)
        trainer.slice_pairs_per_axis = 7
        trainer.patch_size = 3

        for axis in range(3):
            with self.subTest(axis=axis):
                previous_slices, current_slices = trainer.sample_pairs(
                    previous,
                    current,
                    axis,
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
                    0: {0: ["axis_0"]},
                    1: {1: ["axis_1"], 2: ["axis_2"]},
                }
            }
        )

        self.assertEqual(set(domains[0]), {0})
        self.assertEqual(set(domains[1]), {1, 2})

    def test_domains_may_collectively_provide_only_one_axis(self):
        domains = get_domains({"domains": {0: {0: ["axis_0"]}}})

        self.assertEqual(set(domains[0]), {0})

    def test_build_datasets_accepts_one_axis(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "axis_0"
            folder.mkdir()
            _save_image(folder / "sample.png", np.zeros((4, 4), dtype=np.uint8))
            cfg = {
                "data": {
                    "domains": {0: {0: [folder]}},
                    "num_phase": 2,
                    "crop_partial": False,
                    "crop_size": 4,
                    "input_size": 4,
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
                    axes[axis] = [folder]
                domains[domain] = axes
            cfg = {
                "data": {
                    "domains": domains,
                    "num_phase": 2,
                    "crop_partial": False,
                    "crop_size": 4,
                    "input_size": 4,
                }
            }

            datasets = build_datasets(cfg)
            samples = {
                (domain, axis): datasets[domain][axis][0]
                for domain in range(2)
                for axis in range(3)
            }

        self.assertEqual(set(datasets), {0, 1})
        for domain in range(2):
            for axis in range(3):
                expected = 3 * domain + axis
                self.assertTrue(bool((samples[domain, axis] == expected).all()))

    def test_domain_ids_are_contiguous_and_start_at_zero(self):
        with self.assertRaisesRegex(ValueError, "contiguous"):
            get_domains({"domains": {1: {}}})

    def test_build_datasets_passes_partial_crop_setting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folders = {}
            for axis in range(3):
                folder = root / str(axis)
                folder.mkdir()
                _save_image(folder / "sample.png", np.zeros((2, 4), dtype=np.uint8))
                folders[axis] = [folder]
            cfg = {
                "data": {
                    "domains": {0: folders},
                    "num_phase": 2,
                    "crop_partial": True,
                    "crop_size": 4,
                    "input_size": 4,
                }
            }

            datasets = build_datasets(cfg)
            sample = datasets[0][0][0]

        self.assertTrue(datasets[0][0].allow_partial_crop)
        self.assertEqual(sample.shape, torch.Size([2, 4]))

    def test_partial_crop_stream_builds_each_batch_from_one_folder(self):
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
                        np.full((2, 4), 1, dtype=np.uint8),
                    )
                for index in range(4):
                    _save_image(
                        square / f"square_{index}.png",
                        np.full((4, 4), 2, dtype=np.uint8),
                    )
                folders[axis] = [thin, square]
            cfg = {
                "data": {
                    "domains": {0: folders},
                    "num_phase": 2,
                    "crop_partial": True,
                    "crop_size": 4,
                    "input_size": 4,
                }
            }

            dataset = build_datasets(cfg)[0][0]
            sampler = FolderBatchSampler(dataset, batch_size=3)
            with patch(
                "torch.randint",
                side_effect=(torch.tensor(1), torch.tensor([0, 1, 2])),
            ) as randint:
                indices = next(iter(sampler))
            self.assertEqual(randint.call_args_list[0].args, (2, ()))
            stream = build_stream(
                dataset,
                batch_size=3,
                num_workers=0,
                pin_memory=False,
            )
            batch = stream.next()

        self.assertEqual(dataset.batch_groups, ((0, 1), (2, 3, 4, 5)))
        self.assertEqual(indices, [2, 3, 4])
        self.assertIn(
            batch.shape,
            (torch.Size([3, 2, 4]), torch.Size([3, 4, 4])),
        )


if __name__ == "__main__":
    unittest.main()
