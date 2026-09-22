import copy
from pathlib import Path, PurePosixPath, PureWindowsPath


def _pure_path(value):
    text = str(value).replace("\\", "/")
    return PureWindowsPath(text) if PureWindowsPath(text).drive else PurePosixPath(text)


class PathRemapper:
    """Apply explicit prefix maps, longest first, across successive relocations."""

    def __init__(self, groups):
        self.groups = [
            sorted(
                [(_pure_path(old), _pure_path(new)) for old, new in group],
                key=lambda pair: len(pair[0].parts),
                reverse=True,
            )
            for group in groups
        ]

    def __call__(self, value):
        original = value
        for group in self.groups:
            for old, new in group:
                try:
                    relative = _pure_path(value).relative_to(old)
                except ValueError:
                    continue
                value = new.joinpath(*relative.parts).as_posix()
                break
        return str(Path(value)) if value != original else value

    def keys(self, values):
        result = {}
        for path, value in values.items():
            target = self(path)
            if target in result:
                raise ValueError(f"path-map merges distinct source paths: {target}")
            result[target] = value
        return result

    def config(self, cfg):
        cfg = copy.deepcopy(cfg)
        data = cfg.get("data", {})
        for planes in data.get("domains", {}).values():
            for plane, folders in planes.items():
                planes[plane] = [self(folder) for folder in folders]
        split = data.get("split", {})
        if "validation_files" in split:
            split["validation_files"] = [
                self(path) for path in split["validation_files"]
            ]
        if "validation_regions" in split:
            split["validation_regions"] = self.keys(split["validation_regions"])
        train = cfg.get("train", {})
        if train.get("initial_weights") is not None:
            train["initial_weights"] = self(train["initial_weights"])
        source = cfg.get("source", {})
        for key in ("weights", "bank"):
            if key in source:
                source[key] = self(source[key])
        return cfg


def relocate_checkpoint(payload, path_map):
    """Remap path fields only; retain all hashes, tensors and training settings."""
    if not path_map:
        return payload
    group = []
    seen = set()
    for old, new in path_map:
        if not str(old).strip() or not str(new).strip():
            raise ValueError("path-map requires non-empty OLD and NEW prefixes.")
        old = _pure_path(old)
        if old in seen or ".." in old.parts:
            raise ValueError("path-map OLD prefixes must be distinct and normalized.")
        seen.add(old)
        group.append([old.as_posix(), Path(new).expanduser().resolve().as_posix()])
    mapper = PathRemapper([group])
    result = dict(payload)
    result["config"] = mapper.config(payload["config"])
    result["data_fingerprint"] = mapper.keys(payload["data_fingerprint"])
    # Keep the map history for frozen LR train.yaml files, which must remain
    # byte-identical to preserve the recorded source configuration hash.
    result["path_maps"] = [*payload.get("path_maps", []), group]
    if "anchor_bank" in payload:
        result["anchor_bank"] = {
            domain: [
                {
                    **entry,
                    "geometry": {
                        **entry["geometry"],
                        "image_id": mapper(entry["geometry"]["image_id"]),
                    }
                    if entry.get("geometry") is not None
                    else None,
                }
                for entry in entries
            ]
            for domain, entries in payload["anchor_bank"].items()
        }
    return result
