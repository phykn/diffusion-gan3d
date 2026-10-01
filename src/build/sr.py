import torch

from src.build.trainer import build_trainer
from src.config.data import get_domains
from src.config.train import get_sr_resolution, validate_sr_config
from src.data.bank import validate_bank, validate_bank_height
from src.data.source import infer_height_extents
from src.train.trainer import Trainer


def build_sr_trainer(
    cfg: dict, bank: dict, device: torch.device, bank_origins=None, bank_extents=None
) -> Trainer:
    cfg = validate_sr_config(cfg)
    data = cfg["data"]
    domains = get_domains(data)
    resolution = get_sr_resolution(cfg)
    bank = validate_bank(bank, domains, resolution.low_res_voxels, data["num_phases"])
    if cfg["conditioning"]["height_enabled"]:
        data["height_extents"] = infer_height_extents(data)
        bank_extents = validate_bank_height(
            bank,
            bank_origins,
            data["height_extents"] if bank_extents is None else bank_extents,
            data["crop_size"],
        )
    return build_trainer(cfg, device, bank, bank_origins, bank_extents)
