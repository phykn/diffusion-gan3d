from src.device import adaptive_pool


def area_pyramid(images, min_size=16):
    if type(min_size) is not int or min_size < 2:
        raise ValueError("pyramid_min_size must be an integer of at least two.")
    yield images
    while min(images.shape[-2:]) >= 2 * min_size:
        images = adaptive_pool(images, tuple(n // 2 for n in images.shape[-2:]))
        yield images
