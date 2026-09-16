import torch
import torch.nn.functional as F

from src.config import get_plane_groups, normalize_train_config
from src.data.augment import CriticAugment
from src.data.slice import sample_slices
from src.evaluate.structure import structure_metrics
from src.prepare.height import height_field
from src.prepare.resize import phase_channels
from src.train.ema import build_ema, update_ema
from src.train.loss.sr import consistency_loss, gradient_penalty
from src.train.step import check_loss, materialize_metrics, step_optimizer


class SRTrainer:
    def __init__(
        self,
        model,
        critics,
        streams,
        bank,
        cfg,
        device,
        augment=None,
        bank_origins=None,
    ):
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
        self.bank_origins = bank_origins
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
        low = volumes[indices].to(self.device)
        conditioned, corruption_level = self.corrupt_coarse(low)
        height = (
            None
            if self.bank_origins is None
            else self.volume_height(low, self.bank_origins[domain][indices], domain)
        )
        conditions = {} if height is None else {"height": height}
        domain_ids = torch.full(
            (len(low),), domain, device=self.device, dtype=torch.long
        )
        self.model.train()
        self.critics.train().requires_grad_(True)
        d_value = 0.0
        groups = self.critic_groups[domain]
        real_slices = {}
        for _ in range(train["critic_updates_per_step"]):
            with torch.no_grad(), torch.autocast(self.device.type, enabled=self.amp):
                fake = (
                    self.model(
                        conditioned,
                        self.noise(low),
                        domain_ids,
                        corruption_level,
                        **conditions,
                    )
                    .float()
                    .softmax(1)
                )
            for group, axes in groups.items():
                optimizer = self.critic_optims[group]
                optimizer.zero_grad(set_to_none=True)
                critic = self.critics[group]
                total = torch.zeros((), device=self.device)
                for axis in axes:
                    batch = self.streams[domain][axis].next()
                    real_height = None
                    if isinstance(batch, dict):
                        real = batch["image"].to(self.device)
                        thickness = {"z": 0, "y": 1, "x": 2}[
                            self.cfg["data"]["thickness_axis"]
                        ]
                        if axis != thickness:
                            direction = [a for a in range(3) if a != axis].index(
                                thickness
                            )
                            real_height = height_field(
                                real.shape[-2:],
                                direction,
                                batch["height_origin"],
                                self.cfg["data"]["crop_size"] / real.shape[-1],
                                self.cfg["data"]["height_extents"][domain],
                                self.device,
                            )
                    else:
                        real = batch.to(self.device)
                    if real.ndim == 3:
                        real = phase_channels(real, self.model.num_phases)
                    real_slices[axis] = real
                    slices, fake_height = self.slices_with_height(
                        fake, height, axis, real.shape[0], domain
                    )
                    if real_height is None:
                        real, slices = self.augment.apply_together(
                            (real, slices), plane=axis
                        )
                    else:
                        real, slices, real_height, fake_height = (
                            self.augment.apply_together(
                                (real, slices, real_height, fake_height), plane=axis
                            )
                        )
                    real_conditions = (
                        {} if real_height is None else {"height": real_height}
                    )
                    fake_conditions = (
                        {} if fake_height is None else {"height": fake_height}
                    )
                    # Keep WGAN-GP in float32 for stable gradient norms.
                    fake_score, real_score = (
                        critic(slices, **fake_conditions).mean(),
                        critic(real, **real_conditions).mean(),
                    )
                    gp = gradient_penalty(
                        critic, real, slices, real_height, fake_height
                    )
                    diagnostics[f"score/{group}/{axis}/real"] = real_score.detach()
                    diagnostics[f"score/{group}/{axis}/fake"] = fake_score.detach()
                    diagnostics[f"gp/{group}/{axis}"] = gp.detach()
                    total = total + (
                        fake_score - real_score + losses["gradient_penalty_weight"] * gp
                    ) / (len(axes) * len(groups))
                check_loss(total, "SR critic")
                total.backward()
                step_optimizer(optimizer, None, diagnostics, group)
                d_value += total.detach() / train["critic_updates_per_step"]

        self.critics.requires_grad_(False)
        self.generator_optim.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type, enabled=self.amp):
            high = (
                self.model(
                    conditioned,
                    self.noise(low),
                    domain_ids,
                    corruption_level,
                    **conditions,
                )
                .float()
                .softmax(1)
            )
            adversarial = torch.zeros((), device=self.device)
            for axis in self.streams[domain]:
                slices, fake_height = self.slices_with_height(
                    high, height, axis, train["slices_per_plane"], domain
                )
                values = (slices,) if fake_height is None else (slices, fake_height)
                values = self.augment.apply_together(values, plane=axis)
                slices = values[0]
                fake_conditions = {} if fake_height is None else {"height": values[1]}
                group = self.axis_critics[domain][axis]
                adversarial = adversarial - self.critics[group](
                    slices, **fake_conditions
                ).mean() / (len(groups[group]) * len(groups))
            consistency, error = consistency_loss(
                high,
                low,
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
        interval = train["structure_every_steps"]
        if interval and self.step % interval == 0:
            diagnostics.update(structure_metrics(high, real_slices, groups))
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
        return materialize_metrics(
            {
                "step": self.step,
                "domain": domain,
                "generator": loss.detach(),
                "critic": d_value,
                "adversarial": adversarial.detach(),
                "consistency": consistency.detach(),
                "amp/scale": self.scaler.get_scale(),
                "lr_mse": error.detach(),
                "lr_accuracy": accuracy,
                **{f"phase_{i}": value for i, value in enumerate(fractions)},
                **diagnostics,
                "coarse_corruption_mse": (conditioned - low).square().mean().detach(),
                "coarse_corruption_level": corruption_level.mean().detach(),
            }
        )

    def volume_height(self, low, origins, domain):
        data = self.cfg["data"]
        axis = {"z": 0, "y": 1, "x": 2}[data["thickness_axis"]]
        return height_field(
            low.shape[2:],
            axis,
            origins,
            data["crop_size"] / data["lo_res_size"],
            data["height_extents"][domain],
            self.device,
        )

    def slices_with_height(self, volume, height, axis, count, domain):
        if height is None or axis == {"z": 0, "y": 1, "x": 2}.get(
            self.cfg["data"].get("thickness_axis")
        ):
            return sample_slices(volume, axis, count), None
        data = self.cfg["data"]
        thickness = {"z": 0, "y": 1, "x": 2}[data["thickness_axis"]]
        extent = data["height_extents"][domain]
        origins = (height[:, 0, 0, 0, 0] + 1) * extent / 2 - data["crop_size"] / data[
            "lo_res_size"
        ] / 2
        height = height_field(
            volume.shape[2:],
            thickness,
            origins,
            data["crop_size"] / volume.shape[thickness + 2],
            extent,
            volume.device,
        )
        slices = sample_slices(torch.cat((volume, height), 1), axis, count)
        return slices[:, :-1], slices[:, -1:].detach()

    def corrupt_coarse(self, low: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        settings = self.cfg["conditioning"]
        probability = settings["coarse_corruption_probability"]
        strength = settings["coarse_corruption_strength"]
        if probability == 0 or strength == 0:
            return low, low.new_zeros(len(low))
        shape = (low.shape[0], 1, *(max(1, n // 4) for n in low.shape[2:]))
        active = torch.rand((len(low), 1, 1, 1, 1), device=low.device) < probability
        level = (
            torch.rand((len(low), 1, 1, 1, 1), device=low.device) * strength * active
        )
        mask = (
            F.interpolate(
                (torch.rand(shape, device=low.device) < level).float(),
                size=low.shape[2:],
                mode="nearest",
            ).bool()
            & active
        )
        labels = torch.randint(low.shape[1], (shape[0], *shape[2:]), device=low.device)
        replacement = F.one_hot(labels, low.shape[1]).movedim(-1, 1).float()
        replacement = F.interpolate(replacement, size=low.shape[2:], mode="nearest")
        corrupted = torch.where(mask, replacement, low)
        # Condition on realized change; matching replacements contribute zero.
        level = (corrupted - low).abs().mean(dim=(2, 3, 4)).sum(1) * 0.5
        return corrupted, level

    def noise(self, low: torch.Tensor) -> torch.Tensor:
        return torch.randn(
            low.shape[0], self.model.noise_channels, *low.shape[2:], device=self.device
        )

    def save(self, path) -> None:
        torch.save(
            {
                "format": "diffusion-gan3d.sr.train.v6",
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
        if payload.get("format") != "diffusion-gan3d.sr.train.v6":
            raise ValueError("unsupported SR training checkpoint format.")
        self.model.load_state_dict(payload["model"])
        self.ema.load_state_dict(payload["ema"])
        self.critics.load_state_dict(payload["critics"])
        self.generator_optim.load_state_dict(payload["generator_optim"])
        if set(payload["critic_optims"]) != set(self.critic_optims):
            raise ValueError("critic optimizer groups do not match the checkpoint.")
        for group, optimizer in self.critic_optims.items():
            optimizer.load_state_dict(payload["critic_optims"][group])
        self.scaler.load_state_dict(payload["scaler"])
        self.step = payload["step"]

    def export(self, path) -> None:
        torch.save(
            {
                "format": "diffusion-gan3d.sr.v2",
                "config": self.cfg,
                "step": self.step,
                "model": self.ema.state_dict(),
            },
            path,
        )
