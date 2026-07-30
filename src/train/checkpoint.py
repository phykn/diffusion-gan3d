import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch import nn

from ..misc import atomic_torch_save, require_int

FORMAT_VERSION = 1
CRITIC_COUNT = 3
_TOP_LEVEL_KEYS = {
    "format_version",
    "step",
    "config_signature",
    "models",
    "optimizers",
    "scaler",
    "rng",
}


class _Stateful(Protocol):
    def state_dict(self) -> dict: ...

    def load_state_dict(self, state_dict: dict) -> object: ...


def save_checkpoint(
    run_dir: str | Path,
    *,
    step: int,
    denoiser: nn.Module,
    ema_denoiser: nn.Module,
    critics: Sequence[nn.Module],
    denoiser_optimizer: torch.optim.Optimizer,
    critic_optimizers: Mapping[str, torch.optim.Optimizer],
    scaler: _Stateful | None,
    config_signature: Mapping[object, object],
) -> Path:
    require_int("step", step, minimum=0)
    critic_models = _require_critics(critics)
    critic_opts = _require_critic_optimizers(critic_optimizers)
    signature = _plain_signature(config_signature)

    checkpoint = {
        "format_version": FORMAT_VERSION,
        "step": step,
        "config_signature": signature,
        "models": {
            "denoiser": denoiser.state_dict(),
            "ema_denoiser": ema_denoiser.state_dict(),
            "critics": [critic.state_dict() for critic in critic_models],
        },
        "optimizers": {
            "denoiser": denoiser_optimizer.state_dict(),
            "critics": {
                name: optimizer.state_dict() for name, optimizer in critic_opts.items()
            },
        },
        "scaler": {} if scaler is None else scaler.state_dict(),
        "rng": _capture_rng(),
    }
    path = Path(run_dir) / "last.pt"
    atomic_torch_save(checkpoint, path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    denoiser: nn.Module,
    ema_denoiser: nn.Module,
    critics: Sequence[nn.Module],
    denoiser_optimizer: torch.optim.Optimizer,
    critic_optimizers: Mapping[str, torch.optim.Optimizer],
    scaler: _Stateful | None,
    config_signature: Mapping[object, object],
) -> int:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"training checkpoint is required: {checkpoint_path}")

    critic_models = _require_critics(critics)
    critic_opts = _require_critic_optimizers(critic_optimizers)
    expected_signature = _plain_signature(config_signature)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ValueError(
            f"training checkpoint could not be loaded: {checkpoint_path}"
        ) from exc

    _require_mapping_keys("checkpoint", checkpoint, _TOP_LEVEL_KEYS)
    if checkpoint["format_version"] != FORMAT_VERSION:
        raise ValueError(
            "checkpoint format version does not match this implementation."
        )
    step = checkpoint["step"]
    require_int("checkpoint step", step, minimum=0)
    if checkpoint["config_signature"] != expected_signature:
        raise ValueError("checkpoint config signature does not match training config.")

    models = checkpoint["models"]
    _require_mapping_keys(
        "checkpoint models",
        models,
        {"denoiser", "ema_denoiser", "critics"},
    )
    model_critics = _require_state_list(
        "checkpoint critic models",
        models["critics"],
    )

    optimizers = checkpoint["optimizers"]
    _require_mapping_keys(
        "checkpoint optimizers",
        optimizers,
        {"denoiser", "critics"},
    )
    optimizer_critics = _require_optimizer_states(
        "checkpoint critic optimizers",
        optimizers["critics"],
        set(critic_opts),
    )

    scaler_state = checkpoint["scaler"]
    if not isinstance(scaler_state, dict):
        raise TypeError("checkpoint scaler state must be a mapping.")
    runtime_scaler_state = {} if scaler is None else scaler.state_dict()
    if bool(scaler_state) != bool(runtime_scaler_state):
        raise ValueError("checkpoint scaler state does not match training config.")

    rng = _validate_rng(checkpoint["rng"])

    try:
        denoiser.load_state_dict(models["denoiser"], strict=True)
        ema_denoiser.load_state_dict(models["ema_denoiser"], strict=True)
        for critic, state in zip(critic_models, model_critics, strict=True):
            critic.load_state_dict(state, strict=True)
        denoiser_optimizer.load_state_dict(optimizers["denoiser"])
        for name, optimizer in critic_opts.items():
            optimizer.load_state_dict(optimizer_critics[name])
        if scaler is not None:
            scaler.load_state_dict(scaler_state)
    except Exception as exc:
        raise ValueError(
            f"training checkpoint state does not match runtime objects: "
            f"{checkpoint_path}"
        ) from exc

    _restore_rng(rng)
    return step


def _require_critics(critics: Sequence[nn.Module]) -> tuple[nn.Module, ...]:
    if isinstance(critics, (str, bytes)):
        raise TypeError("critics must contain exactly three torch modules.")
    values = tuple(critics)
    if len(values) != CRITIC_COUNT or not all(
        isinstance(critic, nn.Module) for critic in values
    ):
        raise ValueError("critics must contain exactly three torch modules.")
    return values


def _require_critic_optimizers(
    optimizers: Mapping[str, torch.optim.Optimizer],
) -> dict[str, torch.optim.Optimizer]:
    if not isinstance(optimizers, Mapping) or not optimizers:
        raise TypeError("critic_optimizers must be a non-empty mapping.")
    values = {}
    for name, optimizer in optimizers.items():
        if not isinstance(name, str) or not name:
            raise ValueError("critic optimizer names must be non-empty strings.")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("critic optimizer values must be torch optimizers.")
        values[name] = optimizer
    return values


def _require_mapping_keys(
    label: str,
    value: object,
    expected: set[str],
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    if set(value) != expected:
        raise ValueError(f"{label} has missing or unknown values.")


def _require_state_list(label: str, value: object) -> tuple[dict, ...]:
    if not isinstance(value, list) or len(value) != CRITIC_COUNT:
        raise ValueError(f"{label} must contain exactly three states.")
    if not all(isinstance(state, dict) for state in value):
        raise ValueError(f"{label} entries must be mappings.")
    return tuple(value)


def _require_optimizer_states(
    label: str,
    value: object,
    expected: set[str],
) -> dict[str, dict]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} must match the runtime optimizer names.")
    if not all(
        isinstance(name, str) and isinstance(state, dict)
        for name, state in value.items()
    ):
        raise ValueError(f"{label} entries must be named mappings.")
    return dict(value)


def _plain_signature(value: Mapping[object, object]) -> dict:
    if not isinstance(value, Mapping):
        raise TypeError("config_signature must be a mapping.")
    result = _plain_value(value)
    if not isinstance(result, dict):
        raise TypeError("config_signature must be a mapping.")
    return result


def _plain_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise TypeError("config signature keys must be strings or integers.")
            result[key] = _plain_value(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_plain_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(
        f"config signature contains unsupported value: {type(value).__name__}."
    )


def _capture_rng() -> dict[str, object]:
    numpy_state = np.random.get_state()
    return {
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].astype(np.int64)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "python": random.getstate(),
    }


def _validate_rng(value: object) -> dict[str, object]:
    _require_mapping_keys(
        "checkpoint RNG state",
        value,
        {"torch_cpu", "torch_cuda", "numpy", "python"},
    )
    cpu = value["torch_cpu"]
    if not isinstance(cpu, torch.Tensor) or cpu.device.type != "cpu":
        raise ValueError("checkpoint CPU RNG state must be a CPU tensor.")
    cuda = value["torch_cuda"]
    if not isinstance(cuda, list) or not all(
        isinstance(state, torch.Tensor) and state.device.type == "cpu" for state in cuda
    ):
        raise ValueError("checkpoint CUDA RNG states must be CPU tensors.")
    expected_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if len(cuda) != expected_cuda:
        raise ValueError(
            "checkpoint CUDA RNG device count does not match this runtime."
        )

    numpy_state = value["numpy"]
    _require_mapping_keys(
        "checkpoint NumPy RNG state",
        numpy_state,
        {
            "bit_generator",
            "state",
            "position",
            "has_gauss",
            "cached_gaussian",
        },
    )
    if not isinstance(numpy_state["bit_generator"], str):
        raise TypeError("checkpoint NumPy bit generator must be a string.")
    if not isinstance(numpy_state["state"], torch.Tensor):
        raise TypeError("checkpoint NumPy RNG state must be a tensor.")
    require_int("checkpoint NumPy RNG position", numpy_state["position"], minimum=0)
    require_int(
        "checkpoint NumPy RNG has_gauss",
        numpy_state["has_gauss"],
        minimum=0,
    )
    if not isinstance(numpy_state["cached_gaussian"], float):
        raise TypeError("checkpoint NumPy cached Gaussian must be a float.")
    if not isinstance(value["python"], tuple):
        raise TypeError("checkpoint Python RNG state must be a tuple.")
    return dict(value)


def _restore_rng(value: dict[str, object]) -> None:
    torch.set_rng_state(value["torch_cpu"])
    if value["torch_cuda"]:
        torch.cuda.set_rng_state_all(value["torch_cuda"])

    numpy_state = value["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["state"].numpy().astype(np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    random.setstate(value["python"])
