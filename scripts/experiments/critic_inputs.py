"""Paired LR critic-input ablations; training diagnostics are not quality scores."""

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.build.trainer import build_trainer
from src.config.train import load_train_config
from src.train.run.loop import run_train


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def compare(cfg, root, seeds, steps, device):
    if cfg["stage"] != "low_res" or cfg["train"].get("initial_weights"):
        raise ValueError("comparison requires a fresh LR configuration.")
    if steps < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("use positive steps and distinct seeds.")
    if cfg["train"]["num_workers"] != 0:
        raise ValueError("paired sampling requires train.num_workers=0.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; choose --device cpu explicitly.")
    root.mkdir(parents=True, exist_ok=False)
    report = {
        "purpose": "training diagnostics, not held-out quality evaluation",
        "runs": [],
    }
    for seed in seeds:
        initial = None
        for mode in ("pair", "single"):
            config = copy.deepcopy(cfg)
            config["model"]["critic"]["input_mode"] = mode
            config["train"]["total_steps"] = steps
            seed_all(seed)
            trainer = build_trainer(config, device)
            digest = model_hash(trainer.denoiser)
            if initial is not None and digest != initial:
                raise RuntimeError("generator initializations differ between modes.")
            initial = digest
            out = root / f"seed_{seed}" / mode
            # Recreate loader iterators after model initialization consumed RNG.
            seed_all(seed + 1)
            for streams in trainer.streams.values():
                for stream in streams.values():
                    stream.iterator = iter(stream.loader)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            started = time.monotonic()
            run_train(
                trainer,
                steps,
                steps,
                out,
                before_step=lambda step: seed_all(seed + 2 + step),
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.monotonic() - started
            rows = [
                json.loads(row)
                for row in (out / "metrics.jsonl").read_text().splitlines()
            ]
            sensitivities = {}
            for row in rows:
                for key, value in row["diagnostics"].items():
                    if key.startswith("generator_input_gradient/"):
                        sensitivities.setdefault(key, []).append(value)
            report["runs"].append(
                {
                    "seed": seed,
                    "mode": mode,
                    "steps": trainer.completed_steps,
                    "generator_initial_sha256": digest,
                    "seconds": elapsed,
                    "cuda_peak_bytes": torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else None,
                    "mean_input_gradients": {
                        k: float(np.mean(v)) for k, v in sensitivities.items()
                    },
                    "metrics": str((out / "metrics.jsonl").resolve()),
                }
            )
            (root / "report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            del trainer
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config/train/low_res.yaml")
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    compare(
        load_train_config(args.config),
        args.run_dir,
        args.seeds,
        args.steps,
        torch.device(args.device),
    )


if __name__ == "__main__":
    main()
