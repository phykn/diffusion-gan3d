"""Plane names for volumes stored as D,H,W = z,y,x."""

PLANES = ("xy", "xz", "yz")
PLANE_AXES = {plane: axis for axis, plane in enumerate(PLANES)}
PLANE_DIRECTIONS = {"xy": ("y", "x"), "xz": ("z", "x"), "yz": ("z", "y")}


def get_axis(plane: str | int) -> int:
    if isinstance(plane, str) and plane in PLANE_AXES:
        return PLANE_AXES[plane]
    # Numeric axes remain readable in saved configurations and the anchor API.
    if isinstance(plane, int) and not isinstance(plane, bool) and 0 <= plane < 3:
        return plane
    raise ValueError(f"invalid plane {plane!r}; use xy, xz or yz.")
