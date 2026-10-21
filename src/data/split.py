import numpy as np


def training_origin(shape, crop, excluded):
    height, width = shape
    top, left, rows, cols = excluded
    if top + rows > height or left + cols > width:
        raise ValueError("validation region must fit inside its original image.")
    max_y, max_x = height - crop + 1, width - crop + 1
    # Disjoint rectangles of valid integer crop origins: above, below, left, right.
    overlap_start, overlap_end = max(0, top - crop + 1), min(max_y, top + rows)
    boxes = [
        (0, min(max_y, overlap_start), 0, max_x),
        (overlap_end, max_y, 0, max_x),
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
