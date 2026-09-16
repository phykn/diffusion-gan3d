import torch

from src.build.data import resolve_height_metadata
from src.build.trainer import build_trainer
from src.config import (
    get_domains,
    get_sr_sizes,
    normalize_train_config,
    validate_sr_config,
)
from src.data.bank import validate_bank
from src.train.trainer import Trainer


def build_sr_trainer(
    cfg: dict, bank: dict, device: torch.device, bank_origins=None
) -> Trainer:
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
    validate_bank(bank, domains, get_sr_sizes(cfg)[1], cfg["data"]["num_phases"])
    return build_trainer(cfg, device, bank, bank_origins)
