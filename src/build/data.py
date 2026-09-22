from torch.utils.data import DataLoader

from src.config.data import get_domains, get_sizes
from src.config.train import get_sr_sizes, normalize_train_config
from src.data.augment import CriticAugment
from src.data.dataset import RealDataset
from src.data.loader import BatchStream, FolderBatchSampler
from src.data.source import collect_image_groups
from src.plane import PLANES


def build_augmentation(cfg: dict) -> CriticAugment:
    settings = cfg.get("augmentation", {})
    planes = settings.get("planes")
    augment = CriticAugment(
        prob=settings.get("probability", 0.5),
        planes=planes,
        preserve_height=cfg["conditioning"]["height_enabled"],
    )
    active = {axis for axes in get_domains(cfg["data"]).values() for axis in axes}
    if planes is not None and any(PLANES[axis] not in planes for axis in active):
        raise ValueError("augmentation.planes must define every observed plane.")
    return augment


def build_datasets(cfg: dict, high: bool = False) -> dict[int, dict[int, RealDataset]]:
    cfg = normalize_train_config(cfg, cfg.get("stage", "sr" if high else "low_res"))
    data = cfg["data"]
    if high:
        crop, low, high_size = get_sr_sizes(cfg)
    else:
        crop, low, high_size = get_sizes(data)
    datasets = {}
    for domain_id, grouped in collect_image_groups(data).items():
        datasets[domain_id] = {
            axis: RealDataset(
                path_groups,
                crop,
                high_size if high else low,
                data["num_phases"],
                plane=axis,
                validation_regions=data.get("split", {}).get("validation_regions"),
            )
            for axis, path_groups in grouped.items()
        }
    return datasets


def build_stream(
    dataset: RealDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> BatchStream:
    loader = DataLoader(
        dataset,
        batch_sampler=FolderBatchSampler(dataset, batch_size),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return BatchStream(loader)
