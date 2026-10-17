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


@pytest.mark.parametrize("field", ["count", "size"])
@pytest.mark.parametrize("value", [True, 1.5, 0])
def test_invalid_dimensions_leave_no_export(tmp_path, field, value):
    root = tmp_path / "output"
    cfg = _config(root, 1)
    cfg["geometry"] = dict(GEOMETRY)
    section = "output" if field == "count" else "geometry"
    cfg[section][field] = value
    with pytest.raises(ValueError, match="positive integer"):
        generate(cfg)
    assert not root.exists()


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
        positions = make_offsets(axes) + center
        # Particles may cross the padded edge; only their inside voxels exist.
        inside = ((positions >= 0) & (positions < volume.shape)).all(1)
        reconstructed[tuple(positions[inside].T)] += 1
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
    assert metadata["height_axis"] == "z"
    assert metadata["geometry"]["radius_gradient"] == [0.7, 1.3]
    truth = tifffile.imread(result.volumes[0])
    with Image.open(result.slices[1][8]) as image:
        np.testing.assert_array_equal(image, truth[:, 8, :])


def test_jammed_packing_has_no_boundary_density_profile():
    # A hard wall or unwrapped pruning made boundary slabs ~0.05 denser here.
    geometry = {**GEOMETRY, "size": 48, "big_radius": 4, "big_elongation": 3.0}
    geometry.update(big_vf=0.5, small_vf=0.0)
    differences = []
    with pytest.warns(UserWarning, match="jammed"):
        for seed in range(20):
            volume = pack(**geometry, rng=np.random.default_rng(seed))
            big = (volume == 2).mean(axis=(1, 2))
            edges = np.r_[big[:6], big[-6:]].mean()
            differences.append(edges - big[18:30].mean())
    assert abs(np.mean(differences)) < 0.02


def test_seeded_export_is_repeatable_and_records_measured_fractions(tmp_path):
    first = _config(tmp_path / "first", 1)
    second = _config(tmp_path / "second", 1)
    first["output"]["seed"] = second["output"]["seed"] = 5
    volumes = [tifffile.imread(generate(cfg).volumes[0]) for cfg in (first, second)]
    np.testing.assert_array_equal(*volumes)
    metadata = yaml.safe_load((tmp_path / "first" / "simulation.yaml").read_text())
    assert metadata["seed"] == 5
    assert metadata["measured"][0]["big_vf"] == pytest.approx((volumes[0] == 2).mean())


ANODE = {
    "kind": "anode",
    "size": 32,
    "graphite_radius": 6,
    "graphite_aspect": 0.35,
    "graphite_tilt": 10,
    "graphite_vf": 0.3,
    "silicon_radius": 2,
    "silicon_vf": [0.0, 0.2],
    "size_spread": 0.1,
    "min_section": 6,
}


def test_anode_export_follows_silicon_height_profile(tmp_path):
    cfg = {"output": {"data_dir": tmp_path / "anode", "count": 1, "seed": 0}}
    cfg["geometry"] = ANODE
    result = generate(cfg)
    volume = tifffile.imread(result.volumes[0])
    assert set(np.unique(volume)) == {0, 1, 2}
    silicon = (volume == 2).mean(axis=(1, 2))
    assert silicon[-8:].mean() > 4 * silicon[:8].mean()
    metadata = yaml.safe_load((tmp_path / "anode" / "simulation.yaml").read_text())
    assert metadata["geometry"]["kind"] == "anode"
    assert set(metadata["measured"][0]) == {"graphite_vf", "silicon_vf"}


def test_anode_flakes_are_flat_along_thickness():
    from simul.src.anode import graphite_shapes

    shapes = graphite_shapes(10, 0.3, 0.0, 0.0, 1.0, np.random.default_rng(0))
    offsets, _ = shapes[0]
    extent = offsets.max(0) - offsets.min(0)
    assert extent[0] < 0.5 * extent[1:].max()


@pytest.mark.parametrize(
    "field, value",
    [("graphite_vf", [0.5, 1.2]), ("silicon_vf", [[0.1]]), ("graphite_aspect", 0)],
)
def test_invalid_anode_geometry_is_rejected(tmp_path, field, value):
    cfg = {"output": {"data_dir": tmp_path / "anode", "count": 1}}
    cfg["geometry"] = {**ANODE, field: value}
    with pytest.raises(ValueError, match=field):
        generate(cfg)
    assert not (tmp_path / "anode").exists()


def test_unknown_geometry_kind_is_rejected(tmp_path):
    cfg = {"output": {"data_dir": tmp_path / "x", "count": 1}}
    cfg["geometry"] = {**GEOMETRY, "kind": "foam"}
    with pytest.raises(ValueError, match="kind"):
        generate(cfg)


def test_anode_shape_sections_have_no_specks():
    from scipy import ndimage

    from simul.src.anode import flake_rotation, make_shape

    rng = np.random.default_rng(3)
    axes = np.asarray((3.0, 10.0, 7.0))
    offsets, _ = make_shape(axes, flake_rotation(30.0, rng), 12)
    mask = np.zeros((25, 25, 25), dtype=bool)
    mask[tuple((offsets + 12).T)] = True
    for axis in range(3):
        for layer in np.moveaxis(mask, axis, 0):
            labels, count = ndimage.label(layer)
            assert count <= 1
            assert count == 0 or layer.sum() >= 12
