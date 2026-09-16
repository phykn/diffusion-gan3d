from types import SimpleNamespace

import pytest
import torch

from src.predict import sr_memory


def test_sr_budget_counts_shared_states_slab_and_coarse_label_conversion():
    estimate = sr_memory.estimate_sr_memory(
        (16, 20, 24),
        (32, 40, 48),
        3,
        tile_size=16,
        margin=4,
        label_input=True,
        input_element_size=1,
    )
    assert estimate.expanded_shape == (24, 24, 24)
    assert estimate.accumulation_bytes == 4 * 4 * 24 * 48 * 56
    assert estimate.state_bytes == 4 * 3 * 40 * 48 * 56
    assert estimate.coarse_bytes == (1 + 8 + 4 * 3) * 16 * 20 * 24
    assert estimate.output_bytes == 4 * 3 * 32 * 40 * 48
    # Interpolation requires the expanded coarse tile plus one voxel per side.
    assert estimate.cpu_tile_bytes >= 4 * 3 * (28**3 + 14**3 + 24**3)
    larger_margin = sr_memory.estimate_sr_memory(
        (16, 20, 24), (32, 40, 48), 3, tile_size=16, margin=8
    )
    assert larger_margin.model_tile_bytes > estimate.model_tile_bytes
    assert larger_margin.cpu_tile_bytes > estimate.cpu_tile_bytes


def test_single_block_sr_budget_includes_cropped_output_and_fractional_staging():
    estimate = sr_memory.estimate_sr_memory(
        (8, 10, 12), (12, 15, 18), 3, margin=2, label_input=False
    )
    assert estimate.expanded_shape == (16, 19, 22)
    assert estimate.accumulation_bytes == 0
    assert estimate.output_bytes == 4 * 3 * 12 * 15 * 18
    assert estimate.coarse_bytes >= 4 * 3 * 8 * 10 * 12


def test_sr_checks_device_workspace_even_when_accumulation_is_cpu(monkeypatch):
    estimate = sr_memory.estimate_sr_memory(
        (8, 8, 8), (16, 16, 16), 3, tile_size=8, margin=2
    )
    monkeypatch.setattr(sr_memory, "cuda_memory_budget", lambda device: 1024)
    generator = SimpleNamespace(device=torch.device("cuda"), model=None, num_phases=3)
    with pytest.raises(MemoryError, match="SR CUDA"):
        sr_memory.check_sr_memory(estimate, generator)


def test_cuda_fraction_validation_is_budgeted_even_for_cpu_sr(monkeypatch):
    estimate = sr_memory.estimate_sr_memory(
        (128, 128, 128),
        (256, 256, 256),
        3,
        tile_size=16,
        margin=2,
        label_input=False,
        input_on_cuda=True,
        input_element_size=8,
    )
    assert estimate.cuda_input_bytes >= 8 * 128**3
    monkeypatch.setattr(sr_memory, "cuda_memory_budget", lambda device: 1024)
    generator = SimpleNamespace(device=torch.device("cpu"), model=None, num_phases=3)
    with pytest.raises(MemoryError, match="SR input CUDA"):
        sr_memory.check_sr_memory(
            estimate, generator, input_device=torch.device("cuda")
        )


@pytest.mark.parametrize("tile_size", [None, 64])
def test_sr_labels_budget_final_bytes_and_bounded_conversion(tile_size):
    labels = sr_memory.estimate_sr_memory(
        (256,) * 3, (512,) * 3, 3, tile_size=tile_size, output_kind="labels"
    )
    probabilities = sr_memory.estimate_sr_memory(
        (256,) * 3, (512,) * 3, 3, tile_size=tile_size, output_kind="probabilities"
    )
    assert labels.output_bytes == 512**3
    assert labels.cpu_label_bytes + labels.model_label_bytes == 9 * 1024**2
    assert probabilities.cpu_label_bytes == probabilities.model_label_bytes == 0
    if tile_size:
        assert labels.cpu_label_bytes > 0 and labels.model_label_bytes == 0
        assert labels.accumulation_bytes == probabilities.accumulation_bytes
    else:
        assert labels.cpu_label_bytes == 0 and labels.model_label_bytes > 0
        assert labels.cpu_tile_bytes < probabilities.cpu_tile_bytes
