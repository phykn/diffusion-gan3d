from collections.abc import Mapping

from pydantic import ValidationError
from pydantic_core import to_jsonable_python

from src.config.schema import CONFIG_TYPES, STAGES, GenerationSettings


def build_config(values):
    stage = values.get("stage") if isinstance(values, Mapping) else None
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {', '.join(STAGES)}.")
    config = parse(CONFIG_TYPES[stage], values)
    _validate_model(config)
    _validate_plane_groups(config)
    _validate_conditioning(config)
    return config


def build_generation_settings(values):
    return parse(GenerationSettings, values)


def parse(settings_type, values):
    try:
        return settings_type.model_validate(freeze(values))
    except ValidationError as error:
        problems = (
            f"{'.'.join(map(str, problem['loc']))}: {problem['msg']}"
            for problem in error.errors()
        )
        raise ValueError("; ".join(problems)) from None


def change_setting(config, path, value):
    values = to_dict(config)
    section = values
    *section_names, key = path.split(".")
    for name in section_names:
        section = section.get(name) if isinstance(section, dict) else None
    if not isinstance(section, dict) or key not in section:
        raise ValueError(f"unknown setting: {path}")
    section[key] = value
    return build_config(values)


def to_dict(value):
    return to_jsonable_python(value)


def freeze(value):
    if isinstance(value, Mapping):
        return {key: freeze(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    return value


def _validate_model(config):
    widths = config.model.generator.channels
    if any(width % widths[0] for width in widths):
        raise ValueError(
            "model.generator.channels must be integer multiples of its first value."
        )
    if config.patch_size % 2 ** (len(widths) - 1):
        raise ValueError(
            "the patch size must be divisible by the generator's downsampling."
        )


def _validate_plane_groups(config):
    grouped = [plane for group in config.model.critic.plane_groups for plane in group]
    if sorted(grouped) != sorted(config.data.images):
        raise ValueError(
            "model.critic.plane_groups must hold each plane in data.images once."
        )


def _validate_conditioning(config):
    conditioning = config.conditioning
    if conditioning.height_enabled:
        if config.train.volume_batch_size > config.train.real_batch_size:
            raise ValueError(
                "height conditioning requires train.real_batch_size >= "
                "train.volume_batch_size."
            )
        if not set(config.data.images) - {"xy"}:
            raise ValueError("height conditioning requires xz or yz images.")
    elif config.profile_enabled:
        raise ValueError(
            "conditioning.spatial_profile requires conditioning.height_enabled."
        )
