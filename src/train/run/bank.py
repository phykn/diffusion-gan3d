import hashlib
from pathlib import Path

import torch

from src.build.data import build_datasets
from src.build.predict import load_generator
from src.config.files import find_train_config, save_yaml
from src.config.train import load_train_config
from src.prepare.profile import image_profile


def file_hash(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def save_bank(run_dir: Path, step: int, payload: dict) -> dict:
    path = run_dir / "lr_bank" / f"step_{step:08d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Published banks may be referenced by older runs; never overwrite them.
    with path.open("xb") as file:
        torch.save(payload, file)
    return {"bank": str(path), "bank_sha256": file_hash(path)}


def refresh_bank(trainer, run_dir: Path) -> None:
    cfg = trainer.cfg
    source = Path(cfg["source"]["weights"])
    if file_hash(source) != cfg["source"]["weights_sha256"]:
        raise ValueError("frozen LR source weights changed before bank refresh.")
    if file_hash(find_train_config(source)) != cfg["source"]["config_sha256"]:
        raise ValueError("frozen LR source configuration changed before bank refresh.")
    generator = load_generator(source, trainer.device)
    height_enabled = cfg["conditioning"]["height_enabled"]
    base_cfg = load_train_config(find_train_config(source)) if height_enabled else None
    datasets = build_datasets(base_cfg) if height_enabled else None
    interval = cfg["lr_bank"]["refresh_every_steps"]
    for domain, volumes in trainer.bank.items():
        index = (trainer.completed_steps // interval - 1) % len(volumes)
        recorded = sample_bank_condition(base_cfg, domain, datasets)
        conditions = model_bank_condition(recorded)
        if height_enabled:
            trainer.bank_origins[domain][index] = recorded["height_origin"]
            trainer.bank_extents[domain][index] = recorded["height_extent"]
        if trainer.bank_conditions is not None:
            trainer.bank_conditions[domain][index] = recorded
        volumes[index] = generator.generate_probs(
            domain=domain, guidance=cfg["lr_bank"]["guidance"], **conditions
        )
    del generator
    if trainer.device.type == "cuda":
        torch.cuda.empty_cache()
    bank_source = save_bank(
        run_dir,
        trainer.completed_steps,
        {
            "format": "diffusion-gan3d.lr-bank",
            "volumes": trainer.bank,
            "height_extents": trainer.bank_extents,
            "conditions": trainer.bank_conditions,
            "height_origins": trainer.bank_origins,
            "data": cfg["data"],
            "source": dict(cfg["source"]),
        },
    )
    cfg["source"].update(bank_source)
    save_yaml(run_dir / "train.yaml", cfg)


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
