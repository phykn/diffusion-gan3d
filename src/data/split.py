from pathlib import Path

import numpy as np


def resolve_split(settings, root):
    files = settings.get("validation_files", [])
    regions = settings.get("validation_regions", {})
    if not isinstance(files, list) or not isinstance(regions, dict):
        raise ValueError(
            "split requires validation_files list and validation_regions mapping."
        )

    def resolve_image_path(value):
        if not isinstance(value, str) or not value:
            raise ValueError("split image paths must be non-empty strings.")
        return str((Path(root) / value).resolve())

    files = [resolve_image_path(value) for value in files]
    if len(set(files)) != len(files):
        raise ValueError("validation_files must not contain duplicates.")
    resolved = {}
    for source, region in regions.items():
        source = resolve_image_path(source)
        if source in resolved:
            raise ValueError(
                "validation_regions must not contain duplicate image paths."
            )
        if source in files or not isinstance(region, list) or len(region) != 4:
            raise ValueError(
                "a holdout must be a file or one [top,left,height,width] region."
            )
        if any(
            type(value) is not int or value < (0 if i < 2 else 1)
            for i, value in enumerate(region)
        ):
            raise ValueError("invalid validation region coordinates.")
        resolved[source] = region
    return {"validation_files": files, "validation_regions": resolved}


def training_origin(shape, crop, excluded):
    height, width = shape
    top, left, rows, cols = excluded
    if top + rows > height or left + cols > width:
        raise ValueError("validation region must fit inside its original image.")
    max_y, max_x = height - crop + 1, width - crop + 1
    # Disjoint rectangles of valid integer crop origins: above, below, left, right.
    overlap_start, overlap_end = max(0, top - crop + 1), min(max_y, top + rows)
    boxes = [
        (0, min(max_y, max(0, top - crop + 1)), 0, max_x),
        (min(max_y, top + rows), max_y, 0, max_x),
        (overlap_start, overlap_end, 0, min(max_x, max(0, left - crop + 1))),
        (overlap_start, overlap_end, min(max_x, left + cols), max_x),
    ]
    areas = [max(0, b - a) * max(0, d - c) for a, b, c, d in boxes]
    if sum(areas) == 0:
        raise ValueError("no training crop fits outside the validation region.")
    choice = int(np.random.randint(sum(areas)))
    for (a, b, c, d), area in zip(boxes, areas, strict=True):
        if choice < area:
            return a + choice // (d - c), c + choice % (d - c)
        choice -= area
    raise AssertionError("unreachable crop origin")
