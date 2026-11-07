"""Time training steps at a fixed transition (SPEC T15).

    PYTHONPATH=. .venv/bin/python scripts/parity/bench.py --stage lr

Builds the default sample's trainer, takes warmup + steps steps at
transition 5 and records the median of the last steps. Results merge into
--out under "v2/<stage>"; the "v1/<stage>" entries already in
checks/rebuild/bench.json were measured on v1 at commit 0608ab8.
"""

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter

import torch

TRANSITION = 5


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def build(stage: str, device: torch.device, lr_run: Path):
    from src.config.sample import build_stage_configs, load_sample
    from src.export import read_export
    from src.train.run import build_trainer

    configs = build_stage_configs(load_sample("default"))
    if stage == "lr":
        return build_trainer(configs["lr"], device)
    # Step time does not depend on the LR weights the SR stage starts from.
    return build_trainer(configs["sr"], device, lr=read_export(lr_run, "lr"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=("lr", "sr"), required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--lr-run",
        type=Path,
        default=Path("run/t11-lr"),
        help="Default-sample LR run the SR stage starts from (any step).",
    )
    parser.add_argument("--out", type=Path, default=Path("checks/rebuild/bench.json"))
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)
    trainer = build(args.stage, device, args.lr_run)
    times = []
    for index in range(args.warmup + args.steps):
        synchronize(device)
        started = perf_counter()
        trainer.step(index + 1, transition=TRANSITION)
        synchronize(device)
        times.append(perf_counter() - started)
    measured = times[args.warmup :]
    result = {
        "median_s": statistics.median(measured),
        "mean_s": statistics.mean(measured),
        "steps": len(measured),
        "transition": TRANSITION,
        "device": str(device),
        "torch": torch.__version__,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.out.read_text()) if args.out.is_file() else {}
    data[f"v2/{args.stage}"] = result
    args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print(json.dumps({f"v2/{args.stage}": result}))


if __name__ == "__main__":
    main()
