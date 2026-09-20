AXES = (0, 1, 2)

PLANES = ("xy", "xz", "yz")
PLANE_AXES = {plane: axis for axis, plane in enumerate(PLANES)}
PLANE_DIRECTIONS = {"xy": ("y", "x"), "xz": ("z", "x"), "yz": ("z", "y")}


def get_axis(plane: str | int) -> int:
    if isinstance(plane, str) and plane in PLANE_AXES:
        return PLANE_AXES[plane]
    # Internal tensor axes and the anchor API remain numeric; configs use names.
    if isinstance(plane, int) and not isinstance(plane, bool) and 0 <= plane < 3:
        return plane
    raise ValueError(f"invalid plane {plane!r}; use xy, xz or yz.")
