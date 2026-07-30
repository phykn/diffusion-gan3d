import copy
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

import yaml

from src.train.config import DataConfig, load_train_config

ROOT = Path(__file__).resolve().parents[2]


def _config_values() -> dict[str, object]:
    return {
        "data": {
            "folder": {
                0: "data/slices/0",
                1: "data/slices/1",
                2: "data/slices/2",
            },
            "crop_size": 10,
            "patch_size": 10,
            "num_phases": 3,
            "batch_size": 2,
            "num_workers": 0,
        },
        "model": {
            "base_channels": 4,
            "channel_multipliers": [1, 2],
            "embedding_channels": 8,
            "latent_channels": 4,
            "critic_channels": [4, 8],
            "gradient_checkpointing": False,
        },
        "diffusion": {
            "timesteps": 3,
            "beta_min": 0.1,
            "beta_max": 1.0,
        },
        "optim": {
            "generator_lr": 0.0002,
            "critic_lr": 0.0001,
            "beta1": 0.5,
            "beta2": 0.9,
            "r1_gamma": 0.0,
            "r1_interval": 4,
        },
        "train": {
            "checkpoint": "weights/last.pt",
            "steps": 5,
            "volume_batch_size": 1,
            "slices_per_axis": 2,
            "mixed_precision": False,
            "ema_decay": 0.9,
            "save_every_steps": 2,
        },
        "output": {
            "run_root": "runs",
        },
    }


def _write_config(root: Path, values: dict[str, object]) -> Path:
    path = root / "config" / "train.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(values, sort_keys=False),
        encoding="utf-8",
    )
    return path


class TrainConfigTest(unittest.TestCase):
    def test_relative_paths_resolve_from_the_config_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = _write_config(root, _config_values())

            config = load_train_config(path)

        self.assertEqual(
            config.data.folder[0],
            (root / "config" / "data" / "slices" / "0").resolve(),
        )
        self.assertEqual(
            config.train.checkpoint,
            (root / "config" / "weights" / "last.pt").resolve(),
        )
        self.assertEqual(
            config.output.run_root,
            (root / "config" / "runs").resolve(),
        )

    def test_sections_are_strict(self):
        cases = (
            ("missing", lambda values: values.pop("output"), "missing sections"),
            (
                "unknown",
                lambda values: values.update({"legacy": {}}),
                "unknown sections",
            ),
        )
        for name, mutate, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                values = _config_values()
                mutate(values)
                path = _write_config(Path(temp), values)

                with self.assertRaisesRegex(ValueError, message):
                    load_train_config(path)

    def test_fields_are_strict(self):
        with tempfile.TemporaryDirectory() as temp:
            values = _config_values()
            values["data"]["volume_path"] = "reference.tif"
            path = _write_config(Path(temp), values)

            with self.assertRaisesRegex(ValueError, "volume_path"):
                load_train_config(path)

    def test_patch_divisibility_follows_the_configured_model_depth(self):
        shallow = _config_values()
        with tempfile.TemporaryDirectory() as temp:
            shallow_config = load_train_config(
                _write_config(Path(temp), shallow)
            )
        self.assertEqual(shallow_config.data.patch_size, 10)

        deep_valid = copy.deepcopy(shallow)
        deep_valid["data"]["crop_size"] = 12
        deep_valid["data"]["patch_size"] = 12
        deep_valid["model"]["critic_channels"] = [4, 8, 16]
        with tempfile.TemporaryDirectory() as temp:
            deep_config = load_train_config(
                _write_config(Path(temp), deep_valid)
            )
        self.assertEqual(deep_config.data.patch_size, 12)

        deep_invalid = copy.deepcopy(deep_valid)
        deep_invalid["data"]["crop_size"] = 10
        deep_invalid["data"]["patch_size"] = 10
        with tempfile.TemporaryDirectory() as temp:
            path = _write_config(Path(temp), deep_invalid)
            with self.assertRaisesRegex(ValueError, "factor \\(4\\)"):
                load_train_config(path)


class DataLeakContractTest(unittest.TestCase):
    def test_training_data_config_has_no_3d_reference_fields(self):
        forbidden = ("volume", "fraction", "reference", "bulk", "ground_truth")
        for field in fields(DataConfig):
            with self.subTest(field=field.name):
                self.assertFalse(
                    any(token in field.name.lower() for token in forbidden)
                )

    def test_training_runtime_has_no_known_3d_reference_path(self):
        paths = [
            ROOT / "config" / "train.yaml",
            ROOT / "run_train.py",
            *sorted((ROOT / "src" / "train").glob("*.py")),
            ROOT / "src" / "data" / "dataset.py",
            ROOT / "src" / "data" / "axes.py",
            ROOT / "src" / "data" / "image.py",
        ]
        forbidden = (
            "generated/volumes",
            "generated\\volumes",
            "reference_volume",
            "bulk_fraction",
            "ground_truth_volume",
            "tifffile.imread",
            "tifffile.asarray",
        )

        for path in paths:
            text = path.read_text(encoding="utf-8").lower()
            for token in forbidden:
                with self.subTest(path=path.relative_to(ROOT), token=token):
                    self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main()
