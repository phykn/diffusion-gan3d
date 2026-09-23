from pathlib import Path

import torch
from tqdm import trange

from src.build.data import build_datasets
from src.build.predict import build_generator
from src.config.data import get_domains
from src.config.files import save_yaml
from src.prepare.profile import image_profile
from src.storage import atomic_torch_save
from src.train.run import source as _source


def save_bank(run_dir: Path, step: int, payload: dict) -> dict:
    path = run_dir / "lr_bank" / f"step_{step:08d}.pt"
    # Published banks may be referenced by older runs; never overwrite them.
    atomic_torch_save(payload, path, overwrite=False)
    return {"bank": str(path), "bank_sha256": _source.file_hash(path)}


def publish_bank(payload: dict, cfg: dict, run_dir: Path, step: int) -> None:
    payload["data"] = cfg["data"]
    payload["source"] = {
        key: value
        for key, value in cfg["source"].items()
        if key not in {"bank", "bank_sha256"}
    }
    cfg["source"].update(save_bank(run_dir, step, payload))
    save_yaml(run_dir / "train.yaml", cfg)


def create_bank(cfg: dict, device: torch.device) -> dict:
    source = Path(cfg["source"]["weights"])
    base_cfg = _source.load_frozen_source(cfg["source"])
    generator = build_generator(source, base_cfg, device)
    height_enabled = cfg["conditioning"]["height_enabled"]
    datasets = build_datasets(base_cfg) if height_enabled else None
    bank, origins, extents, conditions = {}, {}, {}, {}
    for domain in get_domains(cfg["data"]):
        conditions[domain] = [
            sample_bank_condition(base_cfg, domain, datasets)
            for _ in range(cfg["lr_bank"]["samples_per_domain"])
        ]
        origins[domain] = torch.tensor(
            [c.get("height_origin", 0.0) for c in conditions[domain]]
        )
        extents[domain] = torch.tensor(
            [c.get("height_extent", 1.0) for c in conditions[domain]]
        )
        bank[domain] = torch.stack(
            [
                generator.generate_probs(
                    domain=domain,
                    guidance=cfg["lr_bank"]["guidance"],
                    **model_bank_condition(conditions[domain][i]),
                )
                for i in trange(
                    cfg["lr_bank"]["samples_per_domain"],
                    desc=f"LR bank domain {domain}",
                )
            ]
        )
    del generator
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "format": "diffusion-gan3d.lr-bank",
        "volumes": bank,
        "height_origins": origins if height_enabled else None,
        "height_extents": extents,
        "conditions": conditions,
        "data": cfg["data"],
        "source": dict(cfg["source"]),
    }


def refresh_bank(
    payload: dict,
    cfg: dict,
    step: int,
    device: torch.device,
    path_maps: list | None = None,
) -> None:
    source = Path(cfg["source"]["weights"])
    base_cfg = _source.load_frozen_source(cfg["source"], path_maps)
    height_enabled = cfg["conditioning"]["height_enabled"]
    generator = build_generator(source, base_cfg, device)
    datasets = build_datasets(base_cfg) if height_enabled else None
    interval = cfg["lr_bank"]["refresh_every_steps"]
    for domain, volumes in payload["volumes"].items():
        index = (step // interval - 1) % len(volumes)
        recorded = sample_bank_condition(base_cfg, domain, datasets)
        conditions = model_bank_condition(recorded)
        volume = generator.generate_probs(
            domain=domain, guidance=cfg["lr_bank"]["guidance"], **conditions
        )
        volumes[index] = volume
        if height_enabled:
            payload["height_origins"][domain][index] = recorded["height_origin"]
            payload["height_extents"][domain][index] = recorded["height_extent"]
        if payload.get("conditions") is not None:
            payload["conditions"][domain][index] = recorded
    del generator
    if device.type == "cuda":
        torch.cuda.empty_cache()


def model_bank_condition(record):
    return {
        key: value
        for key, value in record.items()
        if key in ("height_origin", "height_extent", "vf_profile")
    }


def sample_bank_condition(cfg, domain, datasets):
    if datasets is None:
        return {}
    choices = [
        dataset
        for dataset in datasets[domain].values()
        if dataset.height_direction is not None
    ]
    dataset = choices[int(torch.randint(len(choices), ()))]
    paths = [p for group in dataset.path_groups for p in group]
    item = dataset[paths[int(torch.randint(len(paths), ()))]]
    record = {key: item[key] for key in ("image_id", "height_origin", "height_extent")}
    record["image_sha256"] = _source.file_hash(Path(item["image_id"]))
    record["crop_origin"] = item["crop_origin"].tolist()
    record["source_shape"] = item["source_shape"].tolist()
    settings = cfg["conditioning"].get("spatial_profile", {})
    if settings.get("enabled", False):
        profile = image_profile(
            item["image"].unsqueeze(0), dataset.height_direction, settings["num_bins"]
        )[0].T.tolist()
        start = record["height_origin"] / record["height_extent"]
        length = cfg["data"]["crop_size"] / record["height_extent"]
        points = [[0.0, profile[0]]] if start > 0 else []
        points.extend(
            [start + i * length / len(profile), value]
            for i, value in enumerate(profile)
        )
        points.append([1.0, profile[-1]])
        record["vf_profile"] = {
            "axis": "z",
            "points": points,
            "interpolation": "constant",
        }
    return record
