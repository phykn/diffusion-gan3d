"""Measure one generation's peak memory; run as a script.

    python tests/sampling/memory_probe.py EXPORT DEVICE JSON_OPTIONS

LR options are generate's plus anchors (that many xy anchors). SR options are
refine's plus low, the edge of a random LR label cube; tile_size "default" is
default_tiling()'s. Prints JSON: estimate fields, device_peak (bytes above
the start, MPS/CUDA) and host_peak (ru_maxrss growth after the model loads).
"""

import json
import resource
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api import LowResolutionAPI, PlaneAnchor, SuperResolutionAPI  # noqa: E402


def rss() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def allocated(device: torch.device) -> int:
    if device.type == "mps":
        torch.mps.synchronize()
        return torch.mps.driver_allocated_memory()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device)
    return 0


def prepare(export: str, device: str, options: dict):
    """The API, its estimate and the generation to measure."""
    probabilities = options.pop("probabilities", False)
    if "low" in options:
        api = SuperResolutionAPI(export, device=device)
        low = torch.randint(api.num_phases, (options.pop("low"),) * 3)
        if options.get("tile_size") == "default":
            options["tile_size"] = api.default_tiling()[0]
        estimate = api.estimate_memory(
            low.shape, probabilities=probabilities, **options
        )
        run = api.refine_probs if probabilities else api.refine
        return api, estimate, lambda: run(low, seed=0, **options)
    api = LowResolutionAPI(export, device=device)
    patch = api.input_size
    anchors = [
        PlaneAnchor(torch.randint(api.num_phases, (patch, patch)), 0, index)
        for index in range(options.pop("anchors", 0))
    ]
    estimate = api.estimate_memory(
        probabilities=probabilities, anchors=anchors, **options
    )
    run = api.generate_probs if probabilities else api.generate
    return api, estimate, lambda: run(anchors, seed=0, **options)


def main() -> None:
    export, device, options = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    api, estimate, generate = prepare(export, device, options)
    storage = api.check_memory(estimate)
    if api.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(api.device)
    start_device = allocated(api.device)
    start_host = rss()
    peak = [start_device]

    from src.sampling import sampler as module

    original = module.TileSampler.sample

    def sample(self, plan, **kwargs):
        kwargs["step_hook"] = lambda _: peak.append(allocated(api.device))
        return original(self, plan, **kwargs)

    module.TileSampler.sample = sample
    generate()
    peak.append(allocated(api.device))
    print(
        json.dumps(
            {
                "device_bytes": estimate.device_total(storage),
                "host_bytes": estimate.host_total(storage),
                "tiles": estimate.tiles,
                "storage": storage,
                "device_peak": max(peak) - start_device,
                "host_peak": rss() - start_host,
            }
        )
    )


if __name__ == "__main__":
    main()
