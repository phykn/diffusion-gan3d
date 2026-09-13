import torch
from torch import nn

from src.build.data import build_augmentation, build_datasets
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


def build_sr_trainer(cfg: dict, bank: dict, device: torch.device) -> SRTrainer:
    cfg = normalize_train_config(cfg, "sr")
    validate_sr_config(cfg)
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
                cfg["data"]["num_phases"], cfg["model"]["critic"]["channels"]
            )
            for domain, axes in domains.items()
            for group in dict.fromkeys(axis_groups[axis] for axis in axes)
        }
    )
    return SRTrainer(build_sr_model(cfg), critics, streams, bank, cfg, device, augment)
