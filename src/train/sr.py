import math

import torch

from src.config import get_plane_groups, get_sr_sizes, normalize_train_config
from src.data.augment import CriticAugment
from src.plane import PLANES
from src.prepare.resize import phase_channels
from src.train.ema import build_ema, update_ema
from src.train.sr_loss import consistency_loss, gradient_penalty, sample_slices
from src.train.step import check_loss, step_optimizer


class SRTrainer:
    def __init__(self, model, critics, streams, bank, cfg, device, augment=None):
        cfg = normalize_train_config(cfg, "sr")
        self.model = model.to(device)
        self.critics = critics.to(device)
        self.ema = build_ema(self.model)
        self.streams = streams
        groups = get_plane_groups(cfg)
        axis_groups = {axis: group for group, axes in groups.items() for axis in axes}
        self.critic_groups = {
            domain: {
                f"{domain}_{group}": tuple(
                    axis for axis in axes if axis_groups[axis] == group
                )
                for group in dict.fromkeys(axis_groups[axis] for axis in axes)
            }
            for domain, axes in streams.items()
        }
        self.axis_critics = {
            domain: {axis: group for group, axes in groups.items() for axis in axes}
            for domain, groups in self.critic_groups.items()
        }
        self.bank = bank
        self.cfg = cfg
        self.device = device
        self.step = 0
        train = cfg["train"]
        self.amp = bool(train["mixed_precision"] and device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        optim = cfg["optim"]
        self.generator_optim = torch.optim.Adam(
            self.model.parameters(),
            lr=optim["generator_lr"],
            betas=tuple(optim["adam_betas"]),
        )
        self.critic_optims = {
            group: torch.optim.Adam(
                critic.parameters(),
                lr=optim["critic_lr"],
                betas=tuple(optim["adam_betas"]),
            )
            for group, critic in self.critics.items()
        }
        self.augment = CriticAugment(False) if augment is None else augment

    def train_step(self) -> dict[str, float | int | None]:
        diagnostics = {}
        train = self.cfg["train"]
        losses = self.cfg["loss"]
        domain = int(torch.randint(len(self.bank), ()).item())
        volumes = self.bank[domain]
        indices = torch.randint(len(volumes), (train["volume_batch_size"],))
        low = phase_channels(volumes[indices].to(self.device), self.model.num_phases)
        domain_ids = torch.full(
            (len(low),), domain, device=self.device, dtype=torch.long
        )
        self.model.train()
        self.critics.train().requires_grad_(True)
        d_value = 0.0
        groups = self.critic_groups[domain]
        for _ in range(train["critic_updates_per_step"]):
            with torch.no_grad(), torch.autocast(self.device.type, enabled=self.amp):
                fake = self.model(low, self.noise(low), domain_ids).float().softmax(1)
            for group, axes in groups.items():
                optimizer = self.critic_optims[group]
                optimizer.zero_grad(set_to_none=True)
                critic = self.critics[group]
                total = torch.zeros((), device=self.device)
                for axis in axes:
                    real = phase_channels(
                        self.streams[domain][axis].next().to(self.device),
                        self.model.num_phases,
                    )
                    slices = sample_slices(fake, axis, real.shape[0])
                    real, slices = self.augment.apply_together(
                        (real, slices), plane=axis
                    )
                    # GP stays in float32. Average planes within each group, then groups.
                    fake_score, real_score = critic(slices).mean(), critic(real).mean()
                    gp = gradient_penalty(critic, real, slices)
                    diagnostics[f"score/{group}/{axis}/real"] = float(
                        real_score.detach()
                    )
                    diagnostics[f"score/{group}/{axis}/fake"] = float(
                        fake_score.detach()
                    )
                    diagnostics[f"gp/{group}/{axis}"] = float(gp.detach())
                    total = total + (
                        fake_score - real_score + losses["gradient_penalty_weight"] * gp
                    ) / (len(axes) * len(groups))
                check_loss(total, "SR critic")
                total.backward()
                step_optimizer(optimizer, None, diagnostics, group)
                d_value += float(total.detach()) / train["critic_updates_per_step"]

        self.critics.requires_grad_(False)
        self.generator_optim.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, enabled=self.amp):
            high = self.model(low, self.noise(low), domain_ids).float().softmax(1)
            adversarial = torch.zeros((), device=self.device)
            for axis in self.streams[domain]:
                slices = sample_slices(high, axis, train["slices_per_plane"])
                (slices,) = self.augment.apply_together((slices,), plane=axis)
                group = self.axis_critics[domain][axis]
                adversarial = adversarial - self.critics[group](slices).mean() / (
                    len(groups[group]) * len(groups)
                )
            consistency, error = consistency_loss(
                high,
                low,
                losses["downsample_temperature"],
                losses["downsample_mse_tolerance"],
            )
            loss = adversarial + losses["downsample_consistency_weight"] * consistency
        check_loss(loss, "SR generator")
        self.scaler.scale(loss).backward()
        updated = step_optimizer(
            self.generator_optim, self.scaler, diagnostics, "generator"
        )
        self.scaler.update()
        self.critics.requires_grad_(True)
        if updated:
            update_ema(self.ema, self.model, self.cfg["optim"]["ema_decay"])
        self.step += 1
        with torch.no_grad():
            accuracy = (
                (
                    torch.nn.functional.interpolate(
                        high, size=low.shape[2:], mode="area"
                    ).argmax(1)
                    == low.argmax(1)
                )
                .float()
                .mean()
            )
            fractions = high.mean(dim=(0, 2, 3, 4))
        return {
            "step": self.step,
            "domain": domain,
            "generator": float(loss.detach()),
            "critic": d_value,
            "adversarial": float(adversarial.detach()),
            "consistency": float(consistency.detach()),
            "amp/scale": self.scaler.get_scale(),
            "lr_mse": float(error.detach()),
            "lr_accuracy": float(accuracy),
            **{f"phase_{i}": float(value) for i, value in enumerate(fractions)},
            **diagnostics,
        }

    def noise(self, low: torch.Tensor) -> torch.Tensor:
        return torch.randn(
            low.shape[0], self.model.noise_channels, *low.shape[2:], device=self.device
        )

    def save(self, path) -> None:
        torch.save(
            {
                "format": "diffusion-gan3d.sr.train.v4",
                "config": self.cfg,
                "step": self.step,
                "model": self.model.state_dict(),
                "ema": self.ema.state_dict(),
                "critics": self.critics.state_dict(),
                "generator_optim": self.generator_optim.state_dict(),
                "critic_optims": {
                    group: optim.state_dict()
                    for group, optim in self.critic_optims.items()
                },
                "scaler": self.scaler.state_dict(),
            },
            path,
        )

    def resume(self, payload: dict) -> None:
        saved_cfg = normalize_train_config(payload["config"], "sr")
        if get_plane_groups(saved_cfg) != get_plane_groups(self.cfg):
            raise ValueError("critic plane_groups cannot change on resume.")
        critics = payload["critics"]
        if payload.get("format") == "diffusion-gan3d.sr.train.v1":
            names = {
                f"{domain}_{axis}": f"{domain}_{PLANES[axis]}"
                for domain, axes in self.streams.items()
                for axis in axes
            }
            renamed = {}
            for key, value in critics.items():
                name, _, param = key.partition(".")
                renamed[f"{names[name]}.{param}"] = value
            critics = renamed
        self.model.load_state_dict(payload["model"])
        self.ema.load_state_dict(payload["ema"])
        self.critics.load_state_dict(critics)
        self.generator_optim.load_state_dict(payload["generator_optim"])
        if "critic_optims" in payload:
            if set(payload["critic_optims"]) != set(self.critic_optims):
                raise ValueError("critic optimizer groups do not match the checkpoint.")
            for group, optimizer in self.critic_optims.items():
                optimizer.load_state_dict(payload["critic_optims"][group])
        else:
            # v1/v2 used one Adam over all independent critics in module order.
            old = payload["critic_optim"]
            if len(old["param_groups"]) != 1:
                raise ValueError(
                    "legacy SR checkpoint must contain one critic parameter group."
                )
            saved_group = old["param_groups"][0]
            params = saved_group["params"]
            offset = 0
            for group, critic in self.critics.items():
                count = len(list(critic.parameters()))
                ids = params[offset : offset + count]
                self.critic_optims[group].load_state_dict(
                    {
                        "param_groups": [{**saved_group, "params": ids}],
                        "state": {
                            key: old["state"][key] for key in ids if key in old["state"]
                        },
                    }
                )
                offset += count
            if offset != len(params):
                raise ValueError("legacy critic parameter count does not match.")
        self.scaler.load_state_dict(payload["scaler"])
        self.step = payload["step"]

    def export(self, path) -> None:
        torch.save(
            {
                "format": "diffusion-gan3d.sr.v1",
                "config": self.cfg,
                "step": self.step,
                "model": self.ema.state_dict(),
            },
            path,
        )


def validate_sr_config(cfg: dict) -> None:
    cfg = normalize_train_config(cfg, "sr")
    get_sr_sizes(cfg)
    get_plane_groups(cfg)
    for name in (
        "total_steps",
        "volume_batch_size",
        "slices_per_plane",
        "critic_updates_per_step",
        "checkpoint_every_steps",
    ):
        value = cfg["train"][name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"train.{name} must be a positive integer.")
    count = cfg["lr_bank"]["samples_per_domain"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("lr_bank.samples_per_domain must be a positive integer.")
    guidance = cfg["lr_bank"].get(
        "guidance", cfg.get("source", {}).get("guidance", 1.0)
    )
    if (
        isinstance(guidance, bool)
        or not isinstance(guidance, (int, float))
        or not math.isfinite(guidance)
    ):
        raise ValueError("lr_bank.guidance must be finite.")
    for name in (
        "gradient_penalty_weight",
        "downsample_consistency_weight",
        "downsample_mse_tolerance",
        "downsample_temperature",
    ):
        value = cfg["loss"][name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"loss.{name} must be non-negative and finite.")
    if cfg["loss"]["downsample_temperature"] == 0:
        raise ValueError("loss.downsample_temperature must be positive.")
    if not 0 <= cfg["optim"]["ema_decay"] < 1:
        raise ValueError("optim.ema_decay must be in [0, 1).")
