def validate_num_phases(value: int) -> int:
    """Validate the phase count supported by uint8 labels and raw/TIFF output."""
    if type(value) is not int or not 1 <= value <= 256:
        raise ValueError("num_phases must be an integer in [1, 256].")
    return value
