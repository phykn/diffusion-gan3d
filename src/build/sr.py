import torch

from src.build.trainer import build_trainer
from src.config.data import get_domains
from src.config.train import get_sr_sizes, validate_sr_config
from src.data.bank import validate_bank
from src.data.source import infer_height_extents
from src.train.trainer import Trainer


def build_sr_trainer(
    cfg: dict, bank: dict, device: torch.device, bank_origins=None, bank_extents=None
) -> Trainer:
    cfg = validate_sr_config(cfg)
    data = cfg["data"]
    if cfg["conditioning"]["height_enabled"]:
        data["height_extents"] = infer_height_extents(data)
        if bank_origins is None or set(bank_origins) != set(bank):
            raise ValueError("height-conditioned SR requires bank crop origins.")
        for domain, volumes in bank.items():
            origins = bank_origins[domain]
            extent = (
                data["height_extents"][domain]
                if bank_extents is None
                else bank_extents[domain]
            )
            if extent is None:
                raise ValueError("SR bank requires per-sample height extents.")
            extent = torch.as_tensor(extent)
            if (
                extent.shape not in ((), (len(volumes),))
                or not torch.isfinite(extent).all()
                or (extent <= 0).any()
            ):
                raise ValueError("invalid LR bank height extents.")
            if (
                origins.shape != (len(volumes),)
                or not torch.isfinite(origins).all()
                or (origins < 0).any()
                or (origins + data["crop_size"] > extent).any()
            ):
                raise ValueError("invalid LR bank crop origins.")
    domains = get_domains(data)
    _, low_size, _ = get_sr_sizes(cfg)
    bank = validate_bank(bank, domains, low_size, data["num_phases"])
    return build_trainer(cfg, device, bank, bank_origins, bank_extents)
