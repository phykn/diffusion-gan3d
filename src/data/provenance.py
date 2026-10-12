import hashlib


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
