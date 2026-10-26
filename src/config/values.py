import math

from src.phase import validate_num_phases


def check_number(value, name, maximum=None, *, exclusive=False):
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


def check_boolean(value, name):
    # YAML reads a quoted "false" as text, which would act as true.
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean.")


def check_integer(value, name, minimum=1):
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
            check_number(optim[name], f"optim.{name}")
    check_number(optim["ema_decay"], "optim.ema_decay", 1, exclusive=True)
    betas = optim["adam_betas"]
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("optim.adam_betas must contain two numbers in [0, 1).")
    for beta in betas:
        check_number(beta, "optim.adam_betas", 1, exclusive=True)

    conditioning = cfg["conditioning"]
    for name in (
        "domain_keep_probability",
        "cfg_dropout_probability",
        "coarse_corruption_probability",
        "coarse_corruption_strength",
    ):
        if name in conditioning:
            check_number(conditioning[name], f"conditioning.{name}", 1)
    check_number(cfg["augmentation"]["probability"], "augmentation.probability", 1)
    if "selection" in cfg:
        check_number(
            cfg["selection"]["structure_tolerance"], "selection.structure_tolerance"
        )

    for name, value in cfg["loss"].items():
        if name.endswith("_weight") or name == "downsample_mse_tolerance":
            check_number(value, f"loss.{name}")
    if "real_transition" in cfg["loss"]:
        check_number(
            cfg["loss"]["real_transition"]["weight"], "loss.real_transition.weight"
        )
    check_integer(cfg["loss"]["r1_every_steps"], "loss.r1_every_steps")
    diffusion = cfg["model"]["diffusion"]
    beta_min, beta_max = diffusion["beta_min"], diffusion["beta_max"]
    check_number(beta_min, "model.diffusion.beta_min")
    check_number(beta_max, "model.diffusion.beta_max")
    if beta_min >= beta_max:
        raise ValueError("model.diffusion.beta_min must be below beta_max.")
    statistics = cfg["loss"]["group_statistics"]
    check_number(statistics["weight"], "loss.group_statistics.weight")
    check_number(statistics["tolerance"], "loss.group_statistics.tolerance", 1)
    mirror = cfg["loss"]["mirror_balance"]
    check_number(mirror["weight"], "loss.mirror_balance.weight")
    check_number(mirror["tolerance"], "loss.mirror_balance.tolerance", 1)

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
        check_integer(value, f"train.{name}", 0 if name == "num_workers" else 1)
