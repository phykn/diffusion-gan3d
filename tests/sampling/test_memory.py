import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.config.sample import build_stage_configs, load_sample
from src.sampling.memory import MemoryEstimate, check_memory

PROBE = Path(__file__).with_name("memory_probe.py")
ROOT = Path(__file__).resolve().parents[2]
GPU = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else None
)


def probe(export, device, options):
    """Peak memory of one generation, measured in a fresh process."""
    result = subprocess.run(
        [sys.executable, str(PROBE), str(export), device, json.dumps(options)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def default_configs():
    return build_stage_configs(load_sample("default"))


@pytest.fixture(scope="module")
def default_lr_export(random_export, default_configs):
    return random_export(default_configs["lr"], "lr")


@pytest.fixture(scope="module")
def default_sr_export(random_export, default_configs):
    return random_export(default_configs["sr"], "sr", lr=default_configs["lr"])


@pytest.mark.gpu_memory
@pytest.mark.skipif(GPU is None, reason="needs CUDA or MPS")
@pytest.mark.parametrize(
    "options",
    [{"size": 64}, {"blocks": [2, 2, 2]}, {"blocks": [2, 2, 2], "anchors": 1}],
)
def test_measured_gpu_memory_stays_under_the_estimate(default_lr_export, options):
    measured = probe(default_lr_export, GPU, options)
    assert measured["device_peak"] <= measured["device_bytes"], measured
    assert measured["host_peak"] <= measured["host_bytes"], measured


@pytest.mark.gpu_memory
@pytest.mark.skipif(GPU is None, reason="needs CUDA or MPS")
@pytest.mark.parametrize(
    "options",
    [{"low": 64}, {"low": 128, "tile_size": "default"}],
    ids=["128 one tile", "256 default tiling"],
)
def test_measured_gpu_sr_memory_stays_under_the_estimate(default_sr_export, options):
    measured = probe(default_sr_export, GPU, options)
    assert measured["device_peak"] <= measured["device_bytes"], measured
    assert measured["host_peak"] <= measured["host_bytes"], measured


def test_a_generation_over_the_ram_budget_stops_before_sampling(monkeypatch):
    monkeypatch.setattr("src.sampling.memory.ram_budget", lambda: 1024**3)
    estimate = MemoryEstimate(512 * 1024**2, 256 * 1024**2, 8, (96, 96, 96), 1024**3)
    with pytest.raises(MemoryError, match=r"96x96x96 output in 8 tiles"):
        check_memory(estimate, torch.device("cpu"), "auto")


def test_storage_cuda_needs_a_cuda_device():
    estimate = MemoryEstimate(0, 0, 1, (64, 64, 64))
    with pytest.raises(ValueError, match="requires a CUDA device"):
        check_memory(estimate, torch.device("cpu"), "cuda")
    with pytest.raises(ValueError, match="storage must be one of"):
        check_memory(estimate, torch.device("cpu"), "disk")
