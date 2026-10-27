from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import run_train
from simul.src.anode import depth_profile
from src.config.data import describe_resolution, get_resolution
from src.config.files import load_yaml, save_yaml
from src.config.sample import (
    SAMPLES_DIR,
    STAGE_NAMES,
    build_train_config,
    load_sample,
)
from src.config.train import load_train_config, normalize_train_config
from src.prepare.profile import validate_profile


@pytest.mark.parametrize("path", sorted(SAMPLES_DIR.glob("*.yaml")), ids=str)
@pytest.mark.parametrize("stage", ["low_res", "sr"])
def test_shipped_samples_build_valid_stage_configs(path, stage):
    # Simulated images are generated locally, so do not count phases from them.
    cfg = build_train_config(load_sample(path.stem) | {"num_phases": 3}, stage)
    assert cfg["stage"] == stage
    assert cfg["nickname"] == path.stem
    normalize_train_config(cfg, stage)


def test_default_sample_counts_phases_from_its_images():
    cfg = load_train_config(build_train_config(load_sample("default"), "low_res"))
    assert cfg["model"]["critic"]["plane_groups"] == [["xy", "xz", "yz"]]
    assert cfg["data"]["num_phases"] == 2


def test_sample_needs_images_resolution_and_every_structure_value(tmp_path):
    path = tmp_path / "bare.yaml"
    save_yaml(path, {"images": {"xy": "a"}})
    with pytest.raises(
        ValueError, match="needs resolution and structure; see .*default.yaml"
    ):
        load_sample(path)
    sample = {"images": {"xy": "a"}, "resolution": {"crop_size": 128}}
    save_yaml(path, sample | {"structure": {"isotropic": True}})
    with pytest.raises(ValueError, match="resolution.lo_res_size is required"):
        load_sample(path)
    sample["resolution"] |= {"lo_res_size": 64, "sr_scale": 2}
    save_yaml(path, sample | {"structure": {"isotropic": True}})
    with pytest.raises(ValueError, match="structure.depth_gradient must be true"):
        load_sample(path)


def test_structure_decides_stage_settings():
    sample = load_sample("simul_anode") | {"num_phases": 3}
    low = build_train_config(sample, "low_res")
    sr = build_train_config(sample, "sr")
    for cfg in (low, sr):
        assert cfg["model"]["critic"]["plane_groups"] == [["xy"], ["xz", "yz"]]
        assert cfg["conditioning"]["height_enabled"] is True
    assert low["conditioning"]["spatial_profile"] == {"enabled": True}
    assert low["loss"]["spatial_profile_weight"] == 30.0
    assert low["augmentation"]["probability"] == 0.2
    assert sr["augmentation"]["probability"] == 0.2
    assert "spatial_profile" not in sr["conditioning"]
    assert low["data"]["domains"][0]["xz"] == ["data/generated/simul_anode/slices/1"]


def write_sample(tmp_path, **sample):
    path = tmp_path / "trial.yaml"
    structure = {"isotropic": True, "depth_gradient": False, "mirror_symmetric": True}
    resolution = {"crop_size": 128, "lo_res_size": 64, "sr_scale": 2}
    base = {
        "images": {"xy": "a", "xz": "b"},
        "resolution": resolution,
        "structure": structure,
        "num_phases": 2,
    }
    save_yaml(path, base | sample)
    return path


def test_tune_overrides_only_the_named_values(tmp_path):
    path = write_sample(
        tmp_path,
        structure={
            "isotropic": False,
            "depth_gradient": False,
            "mirror_symmetric": True,
        },
        tune={"lr": {"loss": {"r1_weight": 2.0}}},
    )
    cfg = build_train_config(load_sample(path), "low_res")
    assert cfg["loss"]["r1_weight"] == 2.0
    assert cfg["loss"]["r2_weight"] == 0.1
    assert cfg["model"]["critic"]["plane_groups"] == [["xy"], ["xz"]]
    assert cfg["data"]["domains"] == {0: {"xy": ["a"], "xz": ["b"]}}
    assert build_train_config(load_sample(path), "sr")["loss"]["r1_weight"] == 0.5


@pytest.mark.parametrize(
    "sample,match",
    [
        ({"seed": 1}, "unknown sample setting: seed"),
        ({"structure": {"layered": True}}, "unknown structure setting"),
        ({"structure": {"isotropic": "no"}}, "true or false"),
        ({"images": "data"}, "map xy, xz and yz"),
        ({"tune": {"low_res": {}}}, "unknown tune setting"),
        ({"images": {"zz": "a"}}, "xy, xz or yz"),
    ],
)
def test_sample_rejects_unknown_or_invalid_settings(tmp_path, sample, match):
    with pytest.raises(ValueError, match=match):
        build_train_config(load_sample(write_sample(tmp_path, **sample)), "low_res")


@pytest.mark.parametrize("stage", ["lr", "sr"])
def test_tune_typos_fail_before_any_stage_trains(tmp_path, capsys, stage):
    path = write_sample(tmp_path, tune={stage: {"train": {"total_step": 10}}})
    with pytest.raises(ValueError, match="unknown training setting: train.total_step"):
        run_train.main([str(path), "--show"])
    assert capsys.readouterr().out == ""


def test_missing_sample_lists_available_names():
    with pytest.raises(FileNotFoundError, match="available: .*default"):
        load_sample("missing")


@pytest.mark.parametrize(
    "graphite,silicon,count",
    [([0.4, 0.3, 0.2], [0.0, 0.1, 0.2, 0.3], 5), (0.3, 0.1, 2)],
)
def test_simulated_depth_profile_is_a_valid_vf_profile(graphite, silicon, count):
    spec = depth_profile({"graphite_vf": graphite, "silicon_vf": silicon})
    knots, _ = validate_profile(spec, 3)
    assert len(knots) == count


def test_anode_simulation_config_gives_its_depth_profile():
    spec = depth_profile(load_yaml("simul/config_anode.yaml")["geometry"])
    _, values = validate_profile(spec, 3)
    assert values[-1].tolist() == [0.48, 0.22, 0.3]


def test_cli_trains_lr_then_sr_and_reuses_finished_stages(tmp_path, capsys):
    images = tmp_path / "images"
    images.mkdir()
    Image.fromarray((np.indices((20, 20)).sum(0) % 3).astype(np.uint8)).save(
        images / "sample.png"
    )
    model = {
        "generator": {
            "channels": [4, 8],
            "embedding_channels": 8,
            "latent_channels": 4,
        },
        "critic": {"channels": [4, 8]},
        "diffusion": {"num_steps": 2},
        "gradient_checkpointing": False,
    }
    train = {
        "total_steps": 1,
        "mixed_precision": False,
        "real_batch_size": 2,
        "slice_pairs_per_plane": 2,
        "weights_every_steps": 1,
        "archive_every_steps": 1,
    }
    sample = tmp_path / "tiny.yaml"
    save_yaml(
        sample,
        {
            "images": {plane: str(images) for plane in ("xy", "xz", "yz")},
            "structure": {
                "isotropic": True,
                "depth_gradient": False,
                "mirror_symmetric": True,
            },
            "resolution": {"crop_size": 16, "lo_res_size": 8, "sr_scale": 2},
            "tune": {
                "lr": {
                    "model": model,
                    "loss": {"r1_weight": 0.0},
                    "train": train,
                },
                "sr": {
                    "model": model,
                    "lr_bank": {"samples_per_domain": 1},
                    "train": train,
                },
            },
        },
    )
    name = tmp_path / "trial"
    args = [str(sample), "--run-dir", str(name), "--device", "cpu"]
    run_train.main([*args, "--lr-only"])
    assert (tmp_path / "trial-lr" / ".done").is_file()
    assert not (tmp_path / "trial-sr").exists()
    run_train.main(args)
    lr_dir, sr_dir = tmp_path / "trial-lr", tmp_path / "trial-sr"
    for path in (lr_dir, sr_dir):
        assert (path / ".done").is_file()
        assert (path / "generator.pt").is_file()
        assert load_yaml(path / "train.yaml")["nickname"] == "tiny"
    saved = load_yaml(sr_dir / "train.yaml")
    assert saved["data"]["num_phases"] == 3
    assert saved["source"]["weights"] == str(lr_dir)
    assert saved["data"]["domains"][0]["xy"] == [str(images)]

    capsys.readouterr()
    run_train.main(args)
    assert "reuse finished" in capsys.readouterr().out
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("trial")) == [
        "trial-lr",
        "trial-sr",
    ]

    (tmp_path / "broken-lr").mkdir()
    with pytest.raises(SystemExit, match="did not finish"):
        run_train.main([str(sample), "--run-dir", str(tmp_path / "broken")])


def test_cli_show_prints_both_stages(capsys):
    run_train.main(["default", "--show"])
    out = capsys.readouterr().out
    assert "# low_res\nstage: low_res" in out
    assert "# sr\nstage: sr" in out


def test_cli_resume_rejects_a_sample():
    with pytest.raises(SystemExit):
        run_train.parse_args(["default", "--resume", "run/x"])


@pytest.mark.parametrize("option", ["--sr-steps", "--bank-size"])
def test_cli_resume_rejects_options_of_the_other_stage(option):
    with pytest.raises(SystemExit):
        run_train.parse_args(["--resume", "run/x", option, "5"])


@pytest.mark.parametrize(
    "options", (["--resume", "run/x"], ["--sr-steps", "5"], ["--bank-size", "5"])
)
def test_cli_lr_only_rejects_resume_and_sr_options(options):
    with pytest.raises(SystemExit):
        run_train.parse_args(["--lr-only", *options])


def test_cli_expands_home_in_run_paths():
    args = run_train.parse_args(["--resume", "~/x", "--run-dir", "~/y"])
    assert args.resume == Path.home() / "x"
    assert args.run_dir == Path.home() / "y"


def test_sr_scale_sets_the_hr_size(tmp_path):
    path = write_sample(
        tmp_path, resolution={"crop_size": 512, "lo_res_size": 64, "sr_scale": 3}
    )
    for stage in STAGE_NAMES:
        data = build_train_config(load_sample(path), stage)["data"]
        assert (data["crop_size"], data["lo_res_size"], data["hi_res_size"]) == (
            512,
            64,
            192,
        )
    assert describe_resolution(get_resolution(data)) == (
        "crop 512 px -> LR 64 voxels (1 voxel = 8 px), "
        "HR 192 voxels (1 voxel = 2.66667 px)"
    )


@pytest.mark.parametrize(
    "resolution, message",
    [
        ({"sr_scale": 1}, "sr_scale must be an integer of at least 2"),
        ({"sr_scale": 2.5}, "sr_scale must be an integer of at least 2"),
        ({"sr_scale": True}, "sr_scale must be an integer of at least 2"),
        ({"hi_res_size": 128}, "unknown resolution setting: hi_res_size"),
    ],
)
def test_resolution_takes_an_integer_sr_scale(tmp_path, resolution, message):
    base = {"crop_size": 128, "lo_res_size": 64, "sr_scale": 2}
    path = write_sample(tmp_path, resolution=base | resolution)
    with pytest.raises(ValueError, match=message):
        load_sample(path)
