import argparse

import torch


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between zero and one")
    return parsed


def select_indices(size: int, count: int) -> tuple[int, ...]:
    if size < 1:
        raise ValueError("volume depth must be positive.")
    if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= size:
        raise ValueError("count must be between 0 and the volume depth.")
    if count == 0:
        return ()
    return tuple((2 * idx + 1) * size // (2 * count) for idx in range(count))


def select_display_index(size: int, indices: tuple[int, ...]) -> int:
    center = size // 2
    if not indices:
        return center
    return min(indices, key=lambda idx: (abs(idx - center), idx))


def format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


def show_napari(volume: torch.Tensor, name: str = "generated phases") -> None:
    import napari

    viewer = napari.Viewer()
    viewer.add_labels(volume.detach().cpu().numpy(), name=name)
    viewer.dims.ndisplay = 3
    napari.run()
