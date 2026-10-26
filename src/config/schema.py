from collections.abc import Mapping

from src.config.data import get_domains
from src.plane import PLANES


def validate_config_keys(cfg: Mapping, stage: str) -> Mapping:
    common = {
        "stage": None,
        "nickname": None,
        "data": dict.fromkeys(
            (
                "domains",
                "num_phases",
                "crop_size",
                "lo_res_size",
                "hi_res_size",
                "height_extents",
                "depth_profiles",
            )
        ),
        "augmentation": {
            "probability": None,
            "auto_planes": None,
            "real_only": None,
            "planes": {
                plane: dict.fromkeys(("flip_axes", "rotate_90")) for plane in PLANES
            },
        },
        "optim": dict.fromkeys(
            ("generator_lr", "critic_lr", "adam_betas", "ema_decay")
        ),
        "model": {
            "gradient_checkpointing": None,
            "generator": dict.fromkeys(
                ("channels", "embedding_channels", "latent_channels")
            ),
            "critic": dict.fromkeys(
                ("channels", "plane_groups", "pyramid_min_size", "input_mode")
            ),
            "diffusion": dict.fromkeys(
                ("num_steps", "beta_min", "beta_max", "time_embedding")
            ),
        },
        "train": dict.fromkeys(
            (
                "total_steps",
                "mixed_precision",
                "volume_batch_size",
                "real_batch_size",
                "slice_pairs_per_plane",
                "num_workers",
                "weights_every_steps",
                "archive_every_steps",
                "structure_every_steps",
            )
        ),
        "loss": dict.fromkeys(("r1_weight", "r1_every_steps", "r2_weight"))
        | {
            "group_statistics": dict.fromkeys(
                ("weight", "max_gap", "tolerance", "start_step", "ramp_steps")
            ),
            "mirror_balance": dict.fromkeys(("weight", "max_gap", "tolerance")),
        },
    }
    common["data"]["split"] = dict.fromkeys(("validation_files", "validation_regions"))
    if stage == "low_res":
        schema = common | {
            "conditioning": {
                "height_enabled": None,
                "spatial_profile": dict.fromkeys(
                    ("enabled", "num_bins", "critic_enabled")
                ),
                "domain_keep_probability": None,
                "cfg_dropout_probability": None,
            },
            "loss": common["loss"]
            | (
                dict.fromkeys(
                    (
                        "volume_fraction_weight",
                        "spatial_profile_weight",
                        "spatial_profile_gradient_weight",
                    )
                )
                | {
                    "real_transition": dict.fromkeys(
                        ("weight", "max_gap", "start_step", "ramp_steps")
                    )
                }
            ),
            "train": common["train"] | {"initial_weights": None},
            "selection": dict.fromkeys(("enabled", "samples", "structure_tolerance")),
        }
    else:
        schema = common | {
            "conditioning": dict.fromkeys(
                (
                    "coarse_corruption_probability",
                    "coarse_corruption_strength",
                    "height_enabled",
                    "domain_keep_probability",
                )
            ),
            "loss": common["loss"]
            | dict.fromkeys(
                ("downsample_consistency_weight", "downsample_mse_tolerance")
            ),
            "lr_bank": dict.fromkeys(
                ("samples_per_domain", "guidance", "refresh_every_steps")
            ),
            "source": dict.fromkeys(
                (
                    "weights",
                    "weights_sha256",
                    "config_sha256",
                    "bank",
                    "bank_sha256",
                )
            ),
        }

    def check(values, allowed, path=""):
        if not isinstance(values, Mapping):
            raise TypeError(f"{path or 'config'} must be a mapping.")
        for key, value in values.items():
            name = f"{path}.{key}" if path else str(key)
            if key not in allowed:
                raise ValueError(f"unknown training setting: {name}")
            if allowed[key] is not None:
                check(value, allowed[key], name)

    check(cfg, schema)
    if "domains" in cfg.get("data", {}):
        get_domains(cfg["data"])
    return cfg
