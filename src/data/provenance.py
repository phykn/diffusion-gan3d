import hashlib

from PIL import Image

from src.plane import PLANES


def fingerprint_data(streams: dict) -> dict:
    paths = {
        path.resolve()
        for axes in streams.values()
        for stream in axes.values()
        for group in stream.loader.dataset.path_groups
        for path in group
    }
    result = {}
    for path in sorted(paths):
        with path.open("rb") as file:
            result[str(path)] = hashlib.file_digest(file, "sha256").hexdigest()
    return result


def describe_sources(streams: dict, fingerprints: dict, split: dict) -> list[dict]:
    records = []
    for domain, axes in streams.items():
        for axis, stream in axes.items():
            dataset = stream.loader.dataset
            for group in dataset.path_groups:
                for path in group:
                    with Image.open(path) as image:
                        shape = [image.height, image.width]
                    records.append(
                        {
                            "image_id": str(path.resolve()),
                            "domain": domain,
                            "plane": PLANES[axis],
                            "source_shape": shape,
                            "sha256": fingerprints[str(path.resolve())],
                            "validation_region": split.get(
                                "validation_regions", {}
                            ).get(str(path.resolve())),
                        }
                    )
    return records
