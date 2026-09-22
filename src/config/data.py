from collections.abc import Mapping, Sequence
from pathlib import Path

from src.plane import PLANES, get_axis


def validate_sr_source(data: Mapping, base_data: Mapping) -> Mapping:
    if "lo_res_size" not in base_data:
        raise ValueError("SR requires stage-1 weights trained with lo_res_size.")
    if (data["crop_size"], data["lo_res_size"]) != (
        base_data["crop_size"],
        base_data["lo_res_size"],
    ):
        raise ValueError("SR data resolution must match the stage-1 crop/LR sizes.")
    if data["num_phases"] != base_data["num_phases"]:
        raise ValueError("SR data.num_phases must match the stage-1 model.")
    if set(get_domains(data)) != set(get_domains(base_data)):
        raise ValueError("SR data domain IDs must match the stage-1 model.")
    if data.get("height_extents") != base_data.get("height_extents"):
        raise ValueError("SR height coordinates must match the stage-1 model.")
    return data


def get_sizes(data: Mapping[str, object]) -> tuple[int, int, int]:
    if any(
        key in data
        for key in (
            "input_size",
            "scale_factor",
            "allow_part",
            "allow_partial_crops",
        )
    ):
        raise ValueError(
            "use crop_size, lo_res_size and hi_res_size; scale_factor is derived."
        )
    crop, low = data["crop_size"], data["lo_res_size"]
    for name, size in (("crop_size", crop), ("lo_res_size", low)):
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"{name} must be a positive integer.")
    high = data.get("hi_res_size", low)
    if type(high) is not int or high < low:
        raise ValueError("hi_res_size must be an integer at least lo_res_size.")
    return crop, low, high


def get_plane_groups(cfg: Mapping) -> dict[str, tuple[int, ...]]:
    active = {axis for axes in get_domains(cfg["data"]).values() for axis in axes}
    groups = cfg["model"]["critic"].get("plane_groups")
    if not isinstance(groups, (list, tuple)) or not groups:
        raise ValueError("model.critic.plane_groups must be a non-empty list of lists.")
    result = {}
    seen = set()
    for group in groups:
        if not isinstance(group, (list, tuple)) or not group:
            raise ValueError("each critic plane group must be a non-empty list.")
        if any(not isinstance(plane, str) or plane not in PLANES for plane in group):
            raise ValueError("critic plane groups must use xy, xz and yz names.")
        axes = tuple(sorted(get_axis(plane) for plane in group))
        if len(set(axes)) != len(axes) or seen.intersection(axes):
            raise ValueError("each plane must occur in exactly one critic group.")
        seen.update(axes)
        result["_".join(PLANES[axis] for axis in axes)] = axes
    if seen != active:
        raise ValueError("critic plane groups must cover exactly the observed planes.")
    return dict(sorted(result.items(), key=lambda item: item[1]))


def get_domains(
    data: Mapping[str, object],
) -> dict[int, dict[int, Sequence[str | Path]]]:
    domains = data["domains"]
    if not isinstance(domains, Mapping):
        raise TypeError("data.domains must be a mapping.")
    if not domains:
        raise ValueError("data.domains must not be empty.")
    if any(type(domain) is not int for domain in domains) or set(domains) != set(
        range(len(domains))
    ):
        raise ValueError("domain IDs must be contiguous and start at zero.")
    parsed = {}
    for domain, folders in domains.items():
        if not isinstance(folders, Mapping):
            raise TypeError(f"domain {domain} must map axes to folders.")
        if not folders:
            raise ValueError(f"domain {domain} must contain at least one axis.")
        axes = {}
        for plane, paths in folders.items():
            if plane not in PLANES:
                raise ValueError("data.domains must use plane names xy, xz or yz.")
            axis = get_axis(plane)
            if axis in axes:
                raise ValueError(f"domain {domain} contains duplicate plane {plane!r}.")
            axes[axis] = paths
        parsed[domain] = axes
    return parsed
