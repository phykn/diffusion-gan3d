from pathlib import Path

AXES = (0, 1, 2)
_EXTENSIONS = {".png", ".tif", ".tiff"}


def load_axis_paths(folders: dict[int, Path]) -> dict[int, tuple[Path, ...]]:
    if set(folders) != set(AXES):
        raise ValueError("axis folders must contain exactly axes 0, 1, and 2.")
    result = {}
    for axis in AXES:
        folder = Path(folders[axis])
        if not folder.is_dir():
            raise FileNotFoundError(f"axis {axis} folder does not exist: {folder}")
        paths = tuple(
            sorted(
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in _EXTENSIONS
            )
        )
        if not paths:
            raise ValueError(f"axis {axis} folder contains no label images: {folder}")
        result[axis] = paths
    return result
