from pathlib import Path

import pytest
import torch
import yaml

from src.config.sample import build_stage_configs, load_sample
from src.export import build_denoiser, export_payload, write_export

ROOT = Path(__file__).parents[1]
TINY = Path(__file__).parent / "fixtures" / "tiny"
TINY_DEFAULT = TINY / "tiny_default.yaml"


@pytest.fixture(scope="session")
def tiny_sample() -> Path:
    return TINY_DEFAULT


@pytest.fixture(scope="session")
def tiny_configs() -> dict:
    """LR and SR configs of the tiny default sample (crop 32, LR 16, SR x2)."""
    return build_stage_configs(load_sample(TINY_DEFAULT))


@pytest.fixture(scope="session")
def tiny_height_configs() -> dict:
    """The tiny sample with a depth gradient: height and profile conditions."""
    sample = load_sample(TINY_DEFAULT)
    sample["structure"] = {
        "isotropic": False,
        "depth_gradient": True,
        "mirror_symmetric": False,
    }
    return build_stage_configs(sample)


@pytest.fixture(scope="session")
def random_export(tmp_path_factory):
    """Write a seeded random-init LR export of a config; memory and shapes do
    not depend on the weight values."""

    def make(config, name: str, lr=None) -> Path:
        """lr: for SR, the LR config whose random export it carries."""
        torch.manual_seed(0)
        model = build_denoiser(config)
        inner = None
        if lr is not None:
            inner = export_payload(lr, 0, build_denoiser(lr).state_dict())
        path = tmp_path_factory.mktemp("exports") / f"{name}.pt"
        payload = export_payload(config, 0, model.state_dict(), lr=inner)
        return write_export(path, payload)

    return make


@pytest.fixture(scope="session")
def tiny_lr_run(tiny_configs, tmp_path_factory) -> Path:
    """A finished one-step tiny LR run on the CPU."""
    from src.train.run import train_stage

    torch.manual_seed(0)
    run_dir = tmp_path_factory.mktemp("runs") / "tiny-lr"
    return train_stage(tiny_configs["lr"], torch.device("cpu"), run_dir)


@pytest.fixture(scope="session")
def tiny_sr_run(tiny_configs, tiny_lr_run, tmp_path_factory) -> Path:
    """A finished one-step tiny SR run on top of tiny_lr_run."""
    from src.train.run import train_stage

    torch.manual_seed(0)
    run_dir = tmp_path_factory.mktemp("runs") / "tiny-sr"
    return train_stage(
        tiny_configs["sr"], torch.device("cpu"), run_dir, lr_dir=tiny_lr_run
    )


@pytest.fixture(scope="session")
def tiny_anode(tmp_path_factory) -> Path:
    """Two 64^3 simulated anodes (scripts/simul) and their tiny sample file:
    crop 32, LR 16, SR x2, height and depth profile conditioned. The paper
    evaluation splits the real volumes, so there are two."""
    from scripts.simul.src.export import generate
    from src.config.io import load_yaml

    root = tmp_path_factory.mktemp("anode")
    config = load_yaml(ROOT / "scripts" / "simul" / "config_anode.yaml")
    config["output"] |= {"count": 2, "data_dir": str(root / "data")}
    config["geometry"]["size"] = 64
    generate(config)
    sample = yaml.safe_load(TINY_DEFAULT.read_text(encoding="utf-8"))
    sample["images"] = {
        plane: str(root / "data" / "slices" / str(axis))
        for axis, plane in enumerate(("xy", "xz", "yz"))
    }
    sample["structure"] = {
        "isotropic": False,
        "depth_gradient": True,
        "mirror_symmetric": False,
    }
    path = root / "tiny_anode.yaml"
    path.write_text(yaml.safe_dump(sample), encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def tiny_anode_lr_run(tiny_anode, tmp_path_factory) -> Path:
    """A finished one-step LR run of tiny_anode on the CPU."""
    from src.train.run import train_stage

    torch.manual_seed(0)
    configs = build_stage_configs(load_sample(tiny_anode))
    run_dir = tmp_path_factory.mktemp("runs") / "tiny-anode-lr"
    return train_stage(configs["lr"], torch.device("cpu"), run_dir)


@pytest.fixture(scope="session")
def tiny_anode_sr_run(tiny_anode, tiny_anode_lr_run, tmp_path_factory) -> Path:
    """A finished one-step SR run of tiny_anode on top of tiny_anode_lr_run."""
    from src.train.run import train_stage

    torch.manual_seed(0)
    configs = build_stage_configs(load_sample(tiny_anode))
    run_dir = tmp_path_factory.mktemp("runs") / "tiny-anode-sr"
    return train_stage(
        configs["sr"], torch.device("cpu"), run_dir, lr_dir=tiny_anode_lr_run
    )
