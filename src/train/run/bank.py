import hashlib
from pathlib import Path

import torch
from tqdm import trange

from src.build.data import build_datasets
from src.build.predict import load_generator
from src.config.data import get_domains
from src.config.files import find_train_config, load_yaml, save_yaml
from src.config.train import normalize_train_config
from src.prepare.profile import image_profile
from src.storage import atomic_torch_save
from src.train.relocate import PathRemapper


def file_hash(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def validate_frozen_source(source: Path, recorded: dict) -> None:
    for key, path, label in (
        ("weights_sha256", source, "weights"),
        ("config_sha256", find_train_config(source), "configuration"),
    ):
        if not recorded.get(key):
            raise ValueError(f"frozen LR source has no saved {label} hash.")
        if file_hash(path) != recorded[key]:
            raise ValueError(f"frozen LR source {label} changed since SR training.")


def save_bank(run_dir: Path, step: int, payload: dict) -> dict:
    path = run_dir / "lr_bank" / f"step_{step:08d}.pt"
    # Published banks may be referenced by older runs; never overwrite them.
    atomic_torch_save(payload, path, overwrite=False)
    return {"bank": str(path), "bank_sha256": file_hash(path)}


def publish_bank(payload: dict, cfg: dict, run_dir: Path, step: int) -> None:
    payload["data"] = cfg["data"]
    payload["source"] = dict(cfg["source"])
    cfg["source"].update(save_bank(run_dir, step, payload))
    save_yaml(run_dir / "train.yaml", cfg)


def create_bank(cfg: dict, base_cfg: dict, device: torch.device) -> dict:
    generator = load_generator(Path(cfg["source"]["weights"]), device)
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
    validate_frozen_source(source, cfg["source"])
    generator = load_generator(source, device)
    height_enabled = cfg["conditioning"]["height_enabled"]
    base_cfg = (
        normalize_train_config(
            PathRemapper(path_maps or []).config(load_yaml(find_train_config(source)))
        )
        if height_enabled
        else None
    )
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
    record["image_sha256"] = file_hash(Path(item["image_id"]))
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
