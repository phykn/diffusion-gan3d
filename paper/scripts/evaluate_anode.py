import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from paper.scripts.common import (
    TILT_KEYS,
    cache_dir,
    cached,
    mean_std,
    parse_args,
    wall_minutes,
)
from paper.scripts.evaluate_sr import lr_agreement, upsample_linear, upsample_nearest
from src.api import SuperResolutionAPI
from src.config.io import load_yaml, project_path, stored_path
from src.data.crops import downsample, phase_channels
from src.stats.sections import compare_sections, label_sections, section_statistics
from src.stats.volume import (
    chord_mean,
    component_slopes,
    interface_density,
    spanning_fractions,
    tilt_balance,
    tortuosity,
)
from src.storage import load_volume
from src.train.selection import pick_sections

DESCRIPTION = (
    "Evaluate the simulated-anode runs for the paper against the real 3D volumes. "
    "LR volumes are generated over the whole depth and width of the real volumes, "
    "conditioned on the training depth profile, and refined by SR. Both are "
    "compared with the real volumes on their own grid: LR with the real volumes "
    "area-downsampled to the LR grid, SR with them at source resolution. Writes "
    "anode-results.json to --out and caches the volumes in its cache/<SR run>/; "
    "--reuse recomputes the metrics from them."
)
SAMPLES = 8
HR_SAMPLES = 4
SMOKE_SAMPLES = 2
SMOKE_HR_SAMPLES = 2
GRAPHITE = 1
PLANES = ("xy", "xz", "yz")
AXES = "zyx"
BANDS = 8


def main(argv=None) -> None:
    args = parse_args(DESCRIPTION, "SR run directory.", argv)
    run = args.run
    samples, hr_samples = (
        (SMOKE_SAMPLES, SMOKE_HR_SAMPLES) if args.smoke else (SAMPLES, HR_SAMPLES)
    )
    api = SuperResolutionAPI(run, args.device)
    lr_run = project_path(api.config.source.lr_run)
    data = api.lr.config.data
    phases = data.num_phases
    scale = data.crop_size // data.lo_res_size
    cache = cache_dir(args.out, run, args.smoke)

    real = [load_volume(path).to(torch.uint8) for path in real_volume_paths(data)]
    if len(real) < 2:
        raise ValueError(
            f"need two or more real volumes to split, found {len(real)} beside "
            "the training slices."
        )
    real_low = [lr_grid(volume, scale, phases) for volume in real]
    shape = tuple(real_low[0].shape)

    def sample_lr():
        return [
            api.lr.generate_probs(shape=shape, seed=seed) for seed in range(samples)
        ]

    probs = cached(cache / "lr.pt", args.reuse, sample_lr)
    low = [p.argmax(0).to(torch.uint8) for p in probs]
    print(f"LR: {len(low)} volumes of {shape}")

    def refine():
        tile_size, overlap = api.default_tiling()
        return [
            api.refine(p, seed=seed, tile_size=tile_size, overlap=overlap)
            for seed, p in enumerate(probs[:hr_samples])
        ]

    high = cached(cache / "hr.pt", args.reuse, refine)
    print(f"SR: {len(high)} volumes of {tuple(high[0].shape)}")

    results = {
        "run": stored_path(run),
        "lr_run": api.config.source.lr_run,
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
        volumes = [upsample(p, scale) for p in probs[:hr_samples]]
        results["baselines"][name] = summary(
            volumes, real, phases, full=name == "nearest"
        )
    output = args.out / "anode-results.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Results: {output}")


def real_volume_paths(data) -> list[Path]:
    """Ground-truth volumes saved beside the slices the sample trains on."""
    slices = project_path(data.images["xy"][0])
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


def tilt_summary(volumes) -> dict:
    labels = [v.numpy() for v in volumes]
    balance = tilt_balance(labels, GRAPHITE)
    slopes = [component_slopes(v, GRAPHITE) for v in labels]
    result = {}
    for name, stats in balance.items():
        values = np.concatenate([s[name] for s in slopes])
        result[name] = {key: stats[key] for key in TILT_KEYS} | {
            "mean_abs": float(np.abs(values).mean())
        }
    return result


if __name__ == "__main__":
    main()
