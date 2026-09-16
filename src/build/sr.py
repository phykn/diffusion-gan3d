import torch
from torch import nn

from src.build.data import build_augmentation, build_datasets, resolve_height_metadata
from src.build.model import build_sr_model
from src.config import (
    get_domains,
    get_plane_groups,
    get_sr_sizes,
    normalize_train_config,
)
from src.data.sr import SliceStream, validate_bank
from src.model.sr_critic import SliceCritic
from src.train.sr import SRTrainer, validate_sr_config


def build_sr_trainer(
    cfg: dict, bank: dict, device: torch.device, bank_origins=None
) -> SRTrainer:
    cfg = normalize_train_config(cfg, "sr")
    resolve_height_metadata(cfg)
    validate_sr_config(cfg)
    if cfg["conditioning"]["height_enabled"]:
        if bank_origins is None or set(bank_origins) != set(bank):
            raise ValueError("height-conditioned SR requires bank crop origins.")
        for domain, volumes in bank.items():
            origins = bank_origins[domain]
            if (
                origins.shape != (len(volumes),)
                or not torch.isfinite(origins).all()
                or (origins < 0).any()
                or (
                    origins + cfg["data"]["crop_size"]
                    > cfg["data"]["height_extents"][domain]
                ).any()
            ):
                raise ValueError("invalid LR bank crop origins.")
    domains = get_domains(cfg["data"])
    groups = get_plane_groups(cfg)
    axis_groups = {axis: group for group, axes in groups.items() for axis in axes}
    augment = build_augmentation(cfg)
    validate_bank(bank, domains, get_sr_sizes(cfg)[1], cfg["data"]["num_phases"])
    datasets = build_datasets(cfg, high=True)
    streams = {
        domain: {
            axis: SliceStream(dataset, cfg["train"]["slices_per_plane"])
            for axis, dataset in axes.items()
        }
        for domain, axes in datasets.items()
    }
    critics = nn.ModuleDict(
        {
            f"{domain}_{group}": SliceCritic(
                cfg["data"]["num_phases"],
                cfg["model"]["critic"]["channels"],
                cfg["model"]["critic"]["pyramid_min_size"],
                cfg["conditioning"]["height_enabled"],
            )
            for domain, axes in domains.items()
            for group in dict.fromkeys(axis_groups[axis] for axis in axes)
        }
    )
    return SRTrainer(
        build_sr_model(cfg), critics, streams, bank, cfg, device, augment, bank_origins
    )
