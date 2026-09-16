from types import SimpleNamespace

import pytest
import torch

from src.predict import memory


@pytest.fixture(autouse=True)
def isolated_cuda_allocator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)


def generator(device="cuda"):
    return SimpleNamespace(device=torch.device(device), num_phases=2, model=None)


def test_estimate_counts_states_but_bounds_fusion_by_tile_depth():
    small = memory.estimate_memory((64, 100, 90), 3, tile_size=32, margin=4)
    tall = memory.estimate_memory((640, 100, 90), 3, tile_size=32, margin=4)
    assert small.generation_shape == (72, 108, 98)
    assert small.state_bytes == 2 * 2 * 3 * 72 * 108 * 98
    assert small.fusion_bytes == 4 * 4 * 32 * 108 * 98
    assert tall.fusion_bytes == small.fusion_bytes
    assert tall.state_bytes > small.state_bytes
    assert (
        memory.estimate_memory(32, 3, probabilities=True).output_bytes == 4 * 3 * 32**3
    )


def test_estimate_counts_margin_and_overlap_in_resolved_tiles():
    from src.predict.tiled import TiledGenerator

    gen = SimpleNamespace(
        device=torch.device("cpu"),
        num_phases=2,
        model=None,
        patch_size=32,
        default_margin=4,
    )
    for overlap in (0, 8, 15):
        estimate = memory.estimate_memory(
            (75, 64, 37), 2, tile_size=32, margin=4, overlap=overlap
        )
        plan = TiledGenerator(gen)._generation_plan((75, 64, 37), overlap, 4)
        assert estimate.tile_count == plan.tile_count


def test_auto_uses_actual_free_cuda_memory_and_falls_back_to_cpu(monkeypatch):
    estimate = memory.estimate_memory(256, 2, tile_size=32)
    monkeypatch.setattr(
        memory.psutil, "virtual_memory", lambda: SimpleNamespace(available=8 * 1024**3)
    )
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device: (2 * 1024**3, 8 * 1024**3)
    )
    assert memory.select_storage("auto", estimate, generator()) == "cuda"
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device: (400 * 1024**2, 8 * 1024**3)
    )
    assert memory.select_storage("auto", estimate, generator()) == "cpu"
    with pytest.raises(MemoryError, match="CUDA allocation"):
        memory.select_storage("cuda", estimate, generator())


def test_auto_rejects_when_even_one_model_tile_does_not_fit(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device: (128 * 1024**2, 8 * 1024**3)
    )
    with pytest.raises(MemoryError, match="CUDA allocation"):
        memory.select_storage(
            "auto", memory.estimate_memory(32, 2, tile_size=32), generator()
        )


def test_auto_includes_reusable_allocator_cache(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device: (64 * 1024**2, 8 * 1024**3)
    )
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 2 * 1024**3)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 1024**3)
    assert (
        memory.select_storage(
            "auto", memory.estimate_memory(32, 2, tile_size=32), generator()
        )
        == "cuda"
    )


def test_cpu_budget_checked_before_state_allocation(monkeypatch):
    monkeypatch.setattr(
        memory.psutil, "virtual_memory", lambda: SimpleNamespace(available=1024)
    )
    with pytest.raises(MemoryError, match="RAM allocation"):
        memory.select_storage(
            "cpu", memory.estimate_memory(256, 2, tile_size=32), generator("cpu")
        )
