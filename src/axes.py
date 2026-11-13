AXES = (0, 1, 2)
AXIS_NAMES = ("z", "y", "x")
PLANES = ("xy", "xz", "yz")
PLANE_DIRECTIONS = {"xy": ("y", "x"), "xz": ("z", "x"), "yz": ("z", "y")}
PORE_PHASE = 0


def validate_num_phases(value):
    if type(value) is not int or not 1 <= value <= 256:
        raise ValueError("num_phases must be an integer in [1, 256].")
    return value
