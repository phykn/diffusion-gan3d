import math

from src.phase import validate_num_phases


def _number(value, name, maximum=None, *, exclusive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative.")
    if maximum is not None and (value >= maximum if exclusive else value > maximum):
        end = ")" if exclusive else "]"
        raise ValueError(f"{name} must be in [0, {maximum}{end}.")


def validate_training_values(cfg):
    """Check supplied numeric settings for both stages, including sparse configs."""
    data = cfg.get("data", {})
    if "num_phases" in data:
        validate_num_phases(data["num_phases"])

    optim = cfg["optim"]
    for name in ("generator_lr", "critic_lr"):
        if name in optim:
            _number(optim[name], f"optim.{name}")
    _number(optim["ema_decay"], "optim.ema_decay", 1, exclusive=True)
    betas = optim["adam_betas"]
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("optim.adam_betas must contain two numbers in [0, 1).")
    for beta in betas:
        _number(beta, "optim.adam_betas", 1, exclusive=True)

    conditioning = cfg["conditioning"]
    for name in (
        "domain_keep_probability",
        "coarse_corruption_probability",
        "coarse_corruption_strength",
    ):
        if name in conditioning:
            _number(conditioning[name], f"conditioning.{name}", 1)
    if "dropout_probability_per_case" in conditioning:
        # The sampler reserves three disjoint intervals of this probability.
        _number(
            conditioning["dropout_probability_per_case"],
            "conditioning.dropout_probability_per_case",
            1 / 3,
        )
    for name in ("probability", "borrowed_plane_probability"):
        anchor = conditioning.get("anchor", {})
        if name in anchor:
            _number(anchor[name], f"conditioning.anchor.{name}", 1)
    augmentation = cfg.get("augmentation", {})
    if "probability" in augmentation:
        _number(augmentation["probability"], "augmentation.probability", 1)

    for section, values in (
        ("loss", cfg["loss"]),
        ("loss.connectivity", cfg["loss"].get("connectivity", {})),
    ):
        for name, value in values.items():
            if name.endswith("_weight") or name == "downsample_mse_tolerance":
                _number(value, f"{section}.{name}")

    for name in (
        "total_steps",
        "volume_batch_size",
        "real_batch_size",
        "slice_pairs_per_plane",
        "weights_every_steps",
        "archive_every_steps",
        "num_workers",
    ):
        if name not in cfg["train"]:
            continue
        value = cfg["train"][name]
        if name == "archive_every_steps" and value is None:
            continue
        minimum = 0 if name == "num_workers" else 1
        if type(value) is not int or value < minimum:
            qualifier = "non-negative" if minimum == 0 else "positive"
            raise ValueError(f"train.{name} must be a {qualifier} integer.")
