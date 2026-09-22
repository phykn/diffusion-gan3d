from collections.abc import Mapping

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
                "thickness_axis",
                "height_extents",
            )
        ),
        "augmentation": {
            "probability": None,
            "auto_planes": None,
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
            "critic": dict.fromkeys(("channels", "plane_groups", "pyramid_min_size")),
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
        "loss": dict.fromkeys(
            ("critic_local_weight", "r1_weight", "r1_every_steps", "r2_weight")
        ),
    }
    common["data"]["split"] = dict.fromkeys(("validation_files", "validation_regions"))
    if stage == "low_res":
        common["model"]["generator"]["anchor_multiscale_input"] = None
        schema = common | {
            "conditioning": {
                "height_enabled": None,
                "spatial_profile": dict.fromkeys(
                    ("enabled", "num_bins", "critic_enabled")
                ),
                "domain_keep_probability": None,
                "dropout_probability_per_case": None,
                "anchor": dict.fromkeys(
                    (
                        "probability",
                        "start_step",
                        "ramp_steps",
                        "borrowed_plane_probability",
                        "bank_capacity",
                        "plane_spacing",
                    )
                ),
            },
            "loss": common["loss"]
            | (
                dict.fromkeys(
                    (
                        "anchor_pixel_weight",
                        "volume_fraction_weight",
                        "spatial_profile_weight",
                        "spatial_profile_gradient_weight",
                    )
                )
                | {
                    "connectivity": dict.fromkeys(
                        (
                            "max_slice_gap",
                            "adversarial_weight",
                            "normal_transition_weight",
                            "start_step",
                            "ramp_steps",
                            "windows_per_plane",
                        )
                    )
                }
            ),
            "train": common["train"] | {"initial_weights": None},
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
                ("weights", "weights_sha256", "config_sha256", "bank", "bank_sha256")
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
    domains = cfg.get("data", {}).get("domains", {})
    if not isinstance(domains, Mapping):
        raise TypeError("data.domains must be a mapping.")
    for planes in domains.values():
        if not isinstance(planes, Mapping):
            raise TypeError("data.domains must map domains to plane folders.")
        if any(plane not in PLANES for plane in planes):
            raise ValueError("data.domains must use plane names xy, xz or yz.")
    return cfg
