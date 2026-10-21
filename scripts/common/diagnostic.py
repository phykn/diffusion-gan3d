import matplotlib.pyplot as plt
import torch

from src.evaluate.label import voxel_accuracy
from src.plane import PLANES
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


def format_profile(values: tuple[float | None, ...], near_label: str) -> str:
    valid = [
        (distance, value) for distance, value in enumerate(values) if value is not None
    ]
    if not valid:
        return "n/a"
    mean = sum(value for _, value in valid) / len(valid)
    distance, farthest = valid[-1]
    return (
        f"{near_label} {format_percent(valid[0][1])}, mean {format_percent(mean)}, "
        f"farthest {format_percent(farthest)} at distance {distance}"
    )


def middle_slices(volume: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Center xy, xz and yz sections."""
    middle = tuple(size // 2 for size in volume.shape)
    return volume[middle[0]], volume[:, middle[1]], volume[:, :, middle[2]]


def show_slices(volume: torch.Tensor, num_phases: int, title: str) -> None:
    fig, panels = plt.subplots(1, 3, figsize=(10, 4))
    for panel, plane, image in zip(panels, PLANES, middle_slices(volume), strict=True):
        panel.imshow(
            image.numpy(),
            cmap="gray",
            vmin=-0.5,
            vmax=num_phases - 0.5,
            interpolation="nearest",
        )
        panel.set_title(plane)
        panel.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def show_napari(volume: torch.Tensor, name: str = "Generated output") -> None:
    import napari

    viewer = napari.Viewer()
    viewer.add_labels(volume.detach().cpu().numpy(), name=name)
    viewer.dims.ndisplay = 3
    napari.run()
