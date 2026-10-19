import torch

from src.evaluate.label import voxel_accuracy
from src.random import preserved_rng


def select_indices(size: int, count: int) -> tuple[int, ...]:
    if count < 0 or count > size:
        raise ValueError("count must be between 0 and the volume depth.")
    if count == 0:
        return ()
    return tuple((2 * idx + 1) * size // (2 * count) for idx in range(count))


def select_display_index(size: int, indices: tuple[int, ...]) -> int:
    center = size // 2
    if not indices:
        return center
    return min(indices, key=lambda idx: (abs(idx - center), idx))


def anchor_accuracy(
    vol: torch.Tensor, target: torch.Tensor, indices: tuple[int, ...], axis: int
) -> float | None:
    """Voxel accuracy on the anchored planes only."""
    if not indices:
        return None
    idx = torch.tensor(indices, dtype=torch.long)
    actual = vol.movedim(axis, 0).index_select(0, idx)
    expected = target.movedim(axis, 0).index_select(0, idx)
    return voxel_accuracy(actual, expected)


def generate_with_baseline(generator, anchors, anchor_strength, **options):
    """Generate with anchors, then without them from the same noise."""
    # The conditioned sample runs first inside preserved_rng, so the baseline
    # replays its RNG stream.
    with preserved_rng(generator.device):
        conditioned = generator.generate(
            anchors=anchors, anchor_strength=anchor_strength, **options
        )
    baseline = generator.generate(anchors=(), anchor_strength=0.0, **options)
    return conditioned, baseline


def format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def format_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}x"


def show_napari(volume: torch.Tensor, name: str = "Generated output") -> None:
    import napari

    viewer = napari.Viewer()
    viewer.add_labels(volume.detach().cpu().numpy(), name=name)
    viewer.dims.ndisplay = 3
    napari.run()
