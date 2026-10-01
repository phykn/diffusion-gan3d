from torch.utils.data import DataLoader

from src.config.data import get_domains, get_resolution
from src.config.train import get_sr_resolution, normalize_train_config
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


def build_datasets(
    cfg: dict, high_resolution: bool = False
) -> dict[int, dict[int, RealDataset]]:
    cfg = normalize_train_config(
        cfg, cfg.get("stage", "sr" if high_resolution else "low_res")
    )
    data = cfg["data"]
    resolution = get_sr_resolution(cfg) if high_resolution else get_resolution(data)
    output_size = (
        resolution.high_res_voxels if high_resolution else resolution.low_res_voxels
    )
    datasets = {}
    for domain_id, grouped in collect_image_groups(data).items():
        datasets[domain_id] = {
            axis: RealDataset(
                path_groups,
                crop_size=resolution.crop_pixels,
                patch_size=output_size,
                num_phases=data["num_phases"],
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
