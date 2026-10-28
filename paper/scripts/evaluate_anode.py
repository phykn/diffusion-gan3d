"""Evaluate the simulated-anode runs for the paper against the real 3D volumes.

LR volumes are generated over the whole depth and width of the real volumes,
conditioned on the training depth profile, and refined by SR. Both are compared
with the real volumes on their own grid: LR with the real volumes
area-downsampled to the LR grid, SR with them at source resolution.

Writes paper/assets/anode-results.json. Volumes are cached under
checks/paper-anode/<SR run>/; --reuse recomputes the metrics from them.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from paper.scripts.evaluate_lr import cache_dir, cached, interface_density, mean_std
from paper.scripts.evaluate_sr import lr_agreement, upsample_linear, upsample_nearest
from scripts.common.metrics import wall_minutes
from src.config.files import load_yaml
from src.config.train import load_train_config
from src.device import default_device
from src.evaluate.connectivity import spanning_fractions
from src.evaluate.structure import (
    compare_sections,
    label_sections,
    section_statistics,
)
from src.evaluate.tilt import component_slopes, tilt_balance
from src.evaluate.tortuosity import tortuosity
from src.predict.inference import LowResolutionAPI
from src.predict.sr.inference import SuperResolutionAPI, load_sr_export
from src.prepare.resize import downsample, phase_channels
from src.storage import load_volume
from src.train.run.source import frozen_source_weights
from src.train.selection import pick_sections

OUTPUT = PROJECT_ROOT / "paper" / "assets" / "anode-results.json"
CACHE = PROJECT_ROOT / "checks" / "paper-anode"
SAMPLES = 8
HR_SAMPLES = 4
GRAPHITE = 1
PLANES = ("xy", "xz", "yz")
AXES = "zyx"
BANDS = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="SR run directory.")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    run = args.run.resolve()
    device = torch.device(args.device)
    cfg = load_sr_export(run / "generator.pt")["config"]
    lr_weights = frozen_source_weights(cfg["source"])
    lr_run = lr_weights.parent
    lr_cfg = load_train_config(lr_run / "train.yaml")
    phases = lr_cfg["data"]["num_phases"]
    scale = lr_cfg["data"]["crop_size"] // lr_cfg["data"]["lo_res_size"]
    cache = cache_dir(CACHE, run)

    real = [
        load_volume(path).to(torch.uint8) for path in real_volume_paths(lr_cfg["data"])
    ]
    real_low = [lr_grid(volume, scale, phases) for volume in real]
    shape = tuple(real_low[0].shape)

    def sample_lr():
        api = LowResolutionAPI(lr_weights, device)
        return [api.generate_probs(shape=shape, seed=seed) for seed in range(SAMPLES)]

    probs = cached(cache / "lr.pt", args.reuse, sample_lr)
    low = [p.argmax(0).to(torch.uint8) for p in probs]
    print(f"LR: {len(low)} volumes of {shape}")

    def refine():
        api = SuperResolutionAPI(run / "generator.pt", device)
        tile_size, overlap = api.default_tiling()
        return [
            api.refine(p, seed=seed, tile_size=tile_size, overlap=overlap)
            for seed, p in enumerate(probs[:HR_SAMPLES])
        ]

    high = cached(cache / "hr.pt", args.reuse, refine)
    print(f"SR: {len(high)} volumes of {tuple(high[0].shape)}")

    results = {
        "run": str(run.relative_to(PROJECT_ROOT)),
        "lr_run": str(lr_run.relative_to(PROJECT_ROOT)),
        "best_step": load_yaml(run / "selection" / "scores.yaml")["best_step"],
        "lr_best_step": load_yaml(lr_run / "selection" / "scores.yaml")["best_step"],
        "training_minutes": wall_minutes(run / "tensorboard"),
        "lr_training_minutes": wall_minutes(lr_run / "tensorboard"),
        "real_volumes": len(real),
        "source_pixels_per_lr_voxel": scale,
        "lr": summary(low, real_low, phases),
        "sr": summary(high, real, phases)
        | {"lr_agreement": mean_std([lr_agreement(h, p) for h, p in zip(high, probs)])},
        "baselines": {},
    }
    for name, upsample in (("nearest", upsample_nearest), ("linear", upsample_linear)):
        volumes = [upsample(p) for p in probs[:HR_SAMPLES]]
        results["baselines"][name] = summary(
            volumes, real, phases, full=name == "nearest"
        )
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Results: {OUTPUT}")


def real_volume_paths(data: dict) -> list[Path]:
    """Ground-truth volumes saved beside the slices the sample trains on."""
    slices = PROJECT_ROOT / data["domains"][0]["xy"][0]
    paths = sorted((slices.parents[1] / "volumes").glob("*.tiff"))
    if not paths:
        raise FileNotFoundError(f"no real volumes beside {slices}.")
    return paths


def lr_grid(volume: torch.Tensor, scale: int, phases: int) -> torch.Tensor:
    """Real labels area-downsampled like training crops, ties broken at random."""
    shape = tuple(n // scale for n in volume.shape)
    probs = downsample(phase_channels(volume.long()[None], phases), shape)[0]
    return label_sections(probs.movedim(0, 1)).argmax(1).to(torch.uint8)


def summary(volumes, real, phases: int, full: bool = True) -> dict:
    result = {
        "samples": len(volumes),
        "shape": list(volumes[0].shape),
        "sections": section_summary(volumes, real, phases),
        "interface_density": pair(
            [interface_density(v, (0, 1, 2)) for v in volumes],
            [interface_density(v, (0, 1, 2)) for v in real],
        ),
        "phase_fractions": {
            str(phase): pair(
                [(v == phase).float().mean() for v in volumes],
                [(v == phase).float().mean() for v in real],
            )
            for phase in range(phases)
        },
        "profile": profile_summary(volumes, real, phases),
    }
    if not full:
        return result
    result["percolation"] = {}
    for phase in range(phases):
        fake, true = (
            [spanning_fractions((v == phase).numpy()) for v in items]
            for items in (volumes, real)
        )
        result["percolation"][str(phase)] = {
            axis: pair([f[index] for f in fake], [t[index] for t in true])
            for index, axis in enumerate(AXES)
        }
    result["tortuosity_phase0"] = {
        axis: pair(
            [tortuosity(v, 0, index, device="cpu") for v in volumes],
            [tortuosity(v, 0, index, device="cpu") for v in real],
        )
        for index, axis in enumerate(AXES)
    }
    result["chord_mean"] = {
        str(phase): {
            axis: pair(
                [chord_mean(v, phase, index) for v in volumes],
                [chord_mean(v, phase, index) for v in real],
            )
            for index, axis in enumerate(AXES)
        }
        for phase in range(phases)
    }
    result["tilt_phase1"] = {
        "generated": tilt_summary(volumes),
        "real": tilt_summary(real),
    }
    return result


def pair(generated, real) -> dict:
    return {
        "generated": mean_std([float(value) for value in generated]),
        "real": mean_std([float(value) for value in real]),
        "real_values": [float(value) for value in real],
    }


def section_summary(volumes, real, phases: int) -> dict:
    """Section errors per plane against all real sections, on the scale of the
    errors between the real volumes themselves."""
    planes = {}
    for axis, name in enumerate(PLANES):
        reference = section_statistics(pick_sections(real, axis, phases))
        fake = section_statistics(pick_sections(volumes, axis, phases))
        split = compare_sections(
            section_statistics(pick_sections(real[:1], axis, phases)),
            section_statistics(pick_sections(real[1:], axis, phases)),
        )
        errors = compare_sections(reference, fake)
        planes[name] = {
            "errors": errors._asdict(),
            "real_split": split._asdict(),
            "structure": float(
                np.mean([e / max(s, 1e-6) for e, s in zip(errors, split)])
            ),
        }
    side = [planes[name]["structure"] for name in ("xz", "yz")]
    return {
        "planes": planes,
        "structure": float(np.mean([p["structure"] for p in planes.values()])),
        "structure_xy": planes["xy"]["structure"],
        "structure_side": float(np.mean(side)),
    }


def profile_summary(volumes, real, phases: int) -> dict:
    """Phase fractions per depth slice and in equal depth bands."""

    def profiles(items):
        return np.stack(
            [
                [(v == phase).float().mean((1, 2)).numpy() for phase in range(phases)]
                for v in items
            ]
        )

    fake, true = profiles(volumes), profiles(real)
    depth = fake.shape[-1]
    bands = fake.reshape(*fake.shape[:2], BANDS, depth // BANDS).mean(-1)
    real_bands = true.reshape(*true.shape[:2], BANDS, depth // BANDS).mean(-1)
    return {
        "generated": fake.mean(0).tolist(),
        "generated_std": fake.std(0, ddof=1).tolist(),
        "real": true.mean(0).tolist(),
        "mean_abs_error": np.abs(fake.mean(0) - true.mean(0)).mean(1).tolist(),
        "bands": {
            "generated": bands.mean(0).tolist(),
            "generated_std": bands.std(0, ddof=1).tolist(),
            "real": real_bands.mean(0).tolist(),
        },
    }


def chord_mean(volume: torch.Tensor, phase: int, axis: int) -> float:
    """Mean length of runs of phase along axis, in voxels."""
    lines = np.moveaxis(volume.numpy() == phase, axis, -1)
    lines = lines.reshape(-1, lines.shape[-1])
    edges = np.diff(np.pad(lines, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    runs = int((edges == 1).sum())
    return float(lines.sum() / max(runs, 1))


def tilt_summary(volumes) -> dict:
    labels = [v.numpy() for v in volumes]
    balance = tilt_balance(labels, GRAPHITE)
    slopes = [component_slopes(v, GRAPHITE) for v in labels]
    result = {}
    for name, stats in balance.items():
        values = np.concatenate([s[name] for s in slopes])
        result[name] = {
            key: stats[key] for key in ("count", "balance", "mean", "sem")
        } | {"mean_abs": float(np.abs(values).mean())}
    return result


if __name__ == "__main__":
    main()
