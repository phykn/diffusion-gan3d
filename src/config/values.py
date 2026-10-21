import math

from src.phase import validate_num_phases


def _number(value, name, maximum=None, *, exclusive=False):
    if type(value) not in (int, float):
        # PyYAML reads exponents without a decimal point, such as 1e-4, as text.
        hint = " (write 1.0e-4, not 1e-4)" if isinstance(value, str) else ""
        raise ValueError(
            f"{name} must be a number, got {type(value).__name__} {value!r}{hint}."
        )
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative.")
    if maximum is not None and (value >= maximum if exclusive else value > maximum):
        end = ")" if exclusive else "]"
        raise ValueError(f"{name} must be in [0, {maximum}{end}.")


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        bound = {0: "a non-negative integer", 1: "a positive integer"}.get(
            minimum, f"an integer of at least {minimum}"
        )
        raise ValueError(f"{name} must be {bound}.")


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
    anchor = conditioning.get("anchor", {})
    for name in ("probability", "borrowed_plane_probability"):
        if name in anchor:
            _number(anchor[name], f"conditioning.anchor.{name}", 1)
    _number(cfg["augmentation"]["probability"], "augmentation.probability", 1)

    connectivity = cfg["loss"].get("connectivity", {})
    for section, values in (
        ("loss", cfg["loss"]),
        ("loss.connectivity", connectivity),
    ):
        for name, value in values.items():
            if name.endswith("_weight") or name == "downsample_mse_tolerance":
                _number(value, f"{section}.{name}")
    _integer(cfg["loss"]["r1_every_steps"], "loss.r1_every_steps")
    for name, value in (
        ("conditioning.anchor.bank_capacity", anchor.get("bank_capacity", 1)),
        ("conditioning.anchor.plane_spacing", anchor.get("plane_spacing", 1)),
        (
            "loss.connectivity.windows_per_plane",
            connectivity.get("windows_per_plane", 1),
        ),
    ):
        _integer(value, name)
    diffusion = cfg["model"]["diffusion"]
    beta_min, beta_max = diffusion["beta_min"], diffusion["beta_max"]
    _number(beta_min, "model.diffusion.beta_min")
    _number(beta_max, "model.diffusion.beta_max")
    if beta_min >= beta_max:
        raise ValueError("model.diffusion.beta_min must be below beta_max.")
    statistics = cfg["loss"]["group_statistics"]
    _number(statistics["weight"], "loss.group_statistics.weight")
    _number(statistics["tolerance"], "loss.group_statistics.tolerance", 1)

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
        _integer(value, f"train.{name}", 0 if name == "num_workers" else 1)
