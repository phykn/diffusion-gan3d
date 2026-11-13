import math
from collections.abc import Mapping
from dataclasses import MISSING, fields, is_dataclass
from types import MappingProxyType, UnionType
from typing import get_args, get_origin

from src.axes import PLANES, validate_num_phases
from src.config.ranges import check_range
from src.config.schema import CONFIG_TYPES, STAGES, GenerationSettings, SRConfig

FORBIDDEN_NAME_CHARACTERS = '<>:"/\\|?*'


def build_config(values):
    if not isinstance(values, Mapping):
        raise ValueError("config must be a mapping.")
    stage = values.get("stage")
    if not isinstance(stage, str) or stage not in CONFIG_TYPES:
        raise ValueError(f"stage must be one of {', '.join(STAGES)}.")
    config = _build(CONFIG_TYPES[stage], values, "")
    _validate(config)
    return config


def build_generation_settings(values):
    return _build(GenerationSettings, values, "")


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
    if is_dataclass(value):
        return {item.name: to_dict(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {key: to_dict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_dict(item) for item in value]
    return value


def freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    return value


def _build(cls, values, path):
    if not isinstance(values, Mapping):
        raise ValueError(f"{path or 'config'} must be a mapping.")
    settings = {setting.name: setting for setting in fields(cls)}
    for key in values.keys() - settings.keys():
        raise ValueError(f"unknown setting: {_join(path, key)}")
    kwargs = {}
    for name, setting in settings.items():
        key = _join(path, name)
        if name in values:
            kwargs[name] = _convert(setting.type, values[name], key)
            check_range(setting, kwargs[name], key)
        elif setting.default is MISSING and setting.default_factory is MISSING:
            raise ValueError(f"missing setting: {key}")
    return cls(**kwargs)


def _convert(kind, value, path):
    origin = get_origin(kind)
    if origin is UnionType:
        return None if value is None else _convert(get_args(kind)[0], value, path)
    if is_dataclass(kind):
        return _build(kind, value, path)
    if kind is bool:
        # YAML reads a quoted "false" as text, which would act as true.
        if type(value) is not bool:
            raise ValueError(f"{path} must be true or false, got {value!r}.")
        return value
    if kind is int:
        if type(value) is not int:
            raise ValueError(f"{path} must be an integer, got {value!r}.")
        return value
    if kind is float:
        if type(value) not in (int, float):
            # PyYAML reads exponents without a decimal point, such as 1e-4, as text.
            hint = " (write 1.0e-4, not 1e-4)" if isinstance(value, str) else ""
            raise ValueError(f"{path} must be a number, got {value!r}{hint}.")
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite.")
        return float(value)
    if kind is str:
        if not isinstance(value, str):
            raise ValueError(f"{path} must be text, got {value!r}.")
        return value
    if origin is tuple:
        return _convert_tuple(get_args(kind), value, path)
    if origin is Mapping or kind is Mapping:
        return _convert_mapping(get_args(kind), value, path)
    raise TypeError(f"unsupported setting type for {path}: {kind}")


def _convert_tuple(item_kinds, value, path):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{path} must be a list.")
    if item_kinds[-1] is Ellipsis:
        item_kinds = (item_kinds[0],) * len(value)
    elif len(value) != len(item_kinds):
        raise ValueError(f"{path} must list {len(item_kinds)} values.")
    return tuple(
        _convert(kind, item, f"{path}[{index}]")
        for index, (kind, item) in enumerate(zip(item_kinds, value, strict=True))
    )


def _convert_mapping(kinds, value, path):
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping.")
    if not kinds:
        return freeze(value)
    key_kind, item_kind = kinds
    return MappingProxyType(
        {
            _convert(key_kind, key, path): _convert(item_kind, item, _join(path, key))
            for key, item in value.items()
        }
    )


def _join(path, key):
    return f"{path}.{key}" if path else str(key)


def _validate(config):
    _validate_nickname(config.nickname)
    _validate_data(config)
    _validate_model(config)
    for beta in config.optim.adam_betas:
        if not 0 <= beta < 1:
            raise ValueError("optim.adam_betas must contain two numbers in [0, 1).")
    _validate_conditioning(config)


def _validate_nickname(nickname):
    is_folder_name = (
        nickname == nickname.strip()
        and not nickname.endswith(".")
        and not any(
            char in FORBIDDEN_NAME_CHARACTERS or ord(char) < 32 for char in nickname
        )
    )
    if not is_folder_name:
        raise ValueError("nickname must be a valid folder-name component.")


def _validate_data(config):
    data = config.data
    validate_num_phases(data.num_phases)
    if set(data.images) - set(PLANES) or not data.images:
        raise ValueError("data.images must map xy, xz or yz to folders.")
    for plane, folders in data.images.items():
        if not folders:
            raise ValueError(f"data.images.{plane} must list at least one folder.")
    if data.hi_res_size < data.lo_res_size:
        raise ValueError("data.hi_res_size must be at least data.lo_res_size.")
    if isinstance(config, SRConfig) and data.hi_res_size % data.lo_res_size:
        raise ValueError("data.hi_res_size must be a multiple of data.lo_res_size.")


def _validate_model(config):
    model = config.model
    widths = model.generator.channels
    if not widths or any(width < 2 for width in widths):
        raise ValueError("model.generator.channels must list integers of at least 2.")
    if any(width % widths[0] for width in widths):
        raise ValueError(
            "model.generator.channels must be integer multiples of its first value."
        )
    if config.patch_size % 2 ** (len(widths) - 1):
        raise ValueError(
            "the patch size must be divisible by the generator's downsampling."
        )
    critic_widths = model.critic.channels
    if len(critic_widths) < 2 or any(width < 1 for width in critic_widths):
        raise ValueError("model.critic.channels must list at least two levels.")
    _validate_plane_groups(model.critic.plane_groups, set(config.data.images))
    if model.diffusion.beta_min >= model.diffusion.beta_max:
        raise ValueError("model.diffusion.beta_min must be below beta_max.")


def _validate_plane_groups(groups, image_planes):
    grouped = []
    for group in groups:
        if not group:
            raise ValueError("each model.critic.plane_groups entry must be non-empty.")
        for plane in group:
            if plane not in PLANES:
                raise ValueError("model.critic.plane_groups must use xy, xz and yz.")
            grouped.append(plane)
    if len(grouped) != len(set(grouped)):
        raise ValueError("each plane must occur in one critic group only.")
    if set(grouped) != image_planes:
        raise ValueError(
            "model.critic.plane_groups must cover exactly the planes in data.images."
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
