"""The GAN core shared by the LR and SR stages.

One step: burn-in values, the stage's conditions and real crops, a generated
pair (x_t, x_t+1) and its clean prediction, plane critic updates, then a
generator update. Stage trainers (src/train/lr.py, sr.py) supply the
conditions and the non-adversarial objective.
"""

import copy
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, fields, is_dataclass

import torch
from torch import nn

from src.config.schema import StageConfig, plane_groups
from src.data.crops import CropSampler
from src.data.height import grid_spacing, height_field
from src.data.images import fingerprint
from src.data.orientation import CriticAugment, plane_policies
from src.data.pairs import sample_pairs
from src.device import wide_dtype
from src.export import build_denoiser, build_diffusion, export_payload
from src.model.critic import PairCritic2D
from src.plane import AXES, PLANES
from src.rng import RunRandom, randn
from src.train.burn_in import BurnIn, BurnInValues
from src.train.losses.gan import critic_objective, relativistic_loss
from src.train.metrics import structure_metrics


@dataclass
class RealCrops:
    """Real sections of one plane for one step, on the training device."""

    images: torch.Tensor  # [B, K, p, p] phase fractions
    origins: torch.Tensor | None = None  # [B] source z of the crop top (side planes)
    extents: torch.Tensor | None = None  # [B] source image heights
    heights: torch.Tensor | None = None  # [B, 1, p, p] height field (side planes)


@dataclass
class Pairs:
    """Sections (x_t, x_t+1) of one plane, with the critic's height input."""

    previous: torch.Tensor
    current: torch.Tensor
    height: torch.Tensor | None = None

    def conditions(self) -> dict:
        return {} if self.height is None else {"height": self.height}


@dataclass
class StepInputs:
    transition: int
    real: dict[int, RealCrops]
    conditions: dict  # denoiser conditions for the generated volume
    extra: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


def build_ema(model: nn.Module) -> nn.Module:
    ema = copy.deepcopy(model)
    ema.eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(
        ema.parameters(), model.parameters(), strict=True
    ):
        ema_parameter.lerp_(parameter, 1.0 - decay)
    for ema_buffer, buffer in zip(ema.buffers(), model.buffers(), strict=True):
        ema_buffer.copy_(buffer)
    ema.eval()


def build_critics(config: StageConfig) -> nn.ModuleDict:
    critic = config.model.critic
    return nn.ModuleDict(
        {
            group: PairCritic2D(
                num_phases=config.data.num_phases,
                channels=critic.channels,
                embedding_channels=config.model.generator.embedding_channels,
                gradient_checkpointing=config.model.gradient_checkpointing,
                height_channels=int(config.conditioning.height_enabled),
                pyramid_min_size=critic.pyramid_min_size,
            )
            for group in plane_groups(config)
        }
    )


def validate_loss(loss: torch.Tensor, name: str) -> torch.Tensor:
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError(f"non-finite {name} loss; optimizer was not stepped.")
    return loss


def step_optimizer(optimizer, scaler, diagnostics: dict, name: str) -> bool:
    """Step unless the gradient is not finite; that is an error without AMP."""
    scaler.unscale_(optimizer)
    grads = [
        p.grad.detach().float()
        for group in optimizer.param_groups
        for p in group["params"]
        if p.grad is not None
    ]
    if not grads:
        raise RuntimeError(f"{name} has no gradients.")
    dtype = wide_dtype(grads[0].device)
    norm = torch.stack([torch.linalg.vector_norm(g, dtype=dtype) for g in grads]).norm()
    finite = bool(torch.isfinite(norm))
    diagnostics[f"gradient/{name}"] = norm if finite else None
    diagnostics[f"skipped/{name}"] = int(not finite)
    if not finite and not scaler.is_enabled():
        raise FloatingPointError(
            f"non-finite {name} gradient; optimizer was not stepped."
        )
    scaler.step(optimizer)
    return finite


def input_gradient_norms(loss: torch.Tensor, inputs: tuple[torch.Tensor, ...]):
    active = tuple(value for value in inputs if value.requires_grad)
    grads = iter(
        torch.autograd.grad(loss, active, retain_graph=True, allow_unused=True)
        if active and loss.requires_grad
        else (None,) * len(active)
    )
    norms = []
    for value in inputs:
        grad = next(grads) if value.requires_grad else None
        norms.append(
            loss.new_zeros(())
            if grad is None
            else grad.detach().float().flatten(1).norm(dim=1).mean()
        )
    return tuple(norms)


class Trainer:
    """State and update rules shared by both stages."""

    def __init__(self, config: StageConfig, device: torch.device, planes: dict):
        self.config = config
        self.device = torch.device(device)
        self.random = RunRandom(config.train.seed, device)
        self.groups = plane_groups(config)
        self.axes = tuple(
            sorted(axis for axes in self.groups.values() for axis in axes)
        )
        self.patch_size = config.patch_size
        self.num_phases = config.data.num_phases
        self.samplers = {
            PLANES.index(plane): CropSampler(
                images, config.data.crop_size, self.patch_size, self.num_phases
            )
            for plane, images in planes.items()
        }
        # Initial weights come from train.seed too, without touching the
        # process's global generator.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.train.seed)
            denoiser = build_denoiser(config)
            critics = build_critics(config)
        self.denoiser = denoiser.to(self.device)
        self.ema = build_ema(self.denoiser)
        self.critics = critics.to(self.device)
        self.diffusion = build_diffusion(config).to(self.device)
        betas = tuple(config.optim.adam_betas)
        self.optimizers = {
            "generator": self._adam(self.denoiser, config.optim.generator_lr, betas),
            **{
                group: self._adam(critic, config.optim.critic_lr, betas)
                for group, critic in self.critics.items()
            },
        }
        mixed = config.train.mixed_precision
        # fp16 denoiser under the GradScaler on CUDA and MPS; fp16 critics on
        # CUDA only: on MPS (torch 2.14) they fault the GPU.
        self.denoiser_amp = mixed and self.device.type in ("cuda", "mps")
        self.critic_amp = mixed and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.denoiser_amp)
        self.burn_in = BurnIn.from_config(config)
        self.augment = CriticAugment(
            plane_policies(config), config.augmentation.probability
        )
        self.spacing = grid_spacing(config.data.crop_size, self.patch_size)
        self.updates = {name: 0 for name in (*self.critics, "generator")}
        self.completed_steps = 0
        self.generator_updated = True
        self.selection = None
        self.data_fingerprint = fingerprint(planes)
        self.path_maps: list = []
        self.diagnostics: dict = {}
        self.scheduled = self.burn_in.values(0)

    @staticmethod
    def _adam(module, lr, betas):
        # Fused Adam runs one kernel per step instead of several per tensor.
        return torch.optim.Adam(module.parameters(), lr=lr, betas=betas, fused=True)

    # Stage hooks -----------------------------------------------------------

    def prepare(self, step: int, transition: int | None) -> StepInputs:
        raise NotImplementedError

    def objective(self, inputs: StepInputs, clean_probs, fake, real, diagnose):
        """Non-adversarial generator terms: (loss, metrics)."""
        raise NotImplementedError

    def stage_metrics(self, inputs: StepInputs) -> dict:
        return {}

    def export_payload(self, step: int, model) -> dict:
        """The stage's export of EMA weights model at step (SR adds its LR)."""
        return export_payload(self.config, step, model)

    def stage_state(self) -> dict:
        """Stage fields of the checkpoint (SR: lr export and bank)."""
        return {}

    def load_stage_state(self, payload) -> None:
        """Restore the fields stage_state saved."""

    # Step ------------------------------------------------------------------

    def step(self, step: int, transition: int | None = None) -> dict:
        self.diagnostics = {}
        self.scheduled = self.apply_burn_in(step)
        self.denoiser.train()
        self.critics.train()
        inputs = self.prepare(step, transition)
        self.diagnostics.update(inputs.diagnostics)
        previous, current, prediction = self.generate_pair(
            inputs.transition, inputs.conditions
        )
        clean_probs = (prediction + 1.0) * 0.5
        fake = self.fake_pairs(previous, current, inputs)
        real = self.real_pairs(inputs)
        critic_losses, r1 = self.update_critics(inputs.transition, fake, real)
        generator = self.update_generator(inputs, clean_probs, fake, real)
        self.scaler.update()
        self.after_step(step, inputs, clean_probs)
        self.completed_steps = step + 1
        self.diagnostics["amp/scale"] = self.scaler.get_scale()
        if self.generator_updated:
            update_ema(self.ema, self.denoiser, self.scheduled.ema_decay)
        scalars = {
            "loss/generator": generator["adversarial"],
            "loss/generator_total": generator["total"],
            "loss/critic": sum(value for value in critic_losses if value is not None),
            "loss/r1": r1,
            "sampling/transition": inputs.transition,
            f"timestep/{inputs.transition}/generator": generator["adversarial"],
            **{
                f"critic_plane/{PLANES[axis]}": value
                for axis, value in enumerate(critic_losses)
            },
            **self.stage_metrics(inputs),
            **self.diagnostics,
        }
        scalars[f"timestep/{inputs.transition}/critic"] = scalars["loss/critic"]
        return materialize(scalars)

    def after_step(self, step: int, inputs: StepInputs, clean_probs) -> None:
        """Structure metrics of the prediction every structure_every_steps."""
        every = self.config.train.structure_every_steps
        if every and (step + 1) % every == 0:
            real = {axis: crops.images for axis, crops in inputs.real.items()}
            self.diagnostics.update(structure_metrics(clean_probs, real, self.groups))

    def apply_burn_in(self, step: int) -> BurnInValues:
        """Set this step's learning rates and beta2; return its scheduled values."""
        values = self.burn_in.values(step)
        for name, optimizer in self.optimizers.items():
            lr = values.generator_lr if name == "generator" else values.critic_lr
            for group in optimizer.param_groups:
                group["lr"] = lr
                group["betas"] = values.betas
        self.diagnostics.update(
            {
                "burn_in/generator_lr": values.generator_lr,
                "burn_in/critic_lr": values.critic_lr,
                "burn_in/beta2": values.betas[1],
                "burn_in/regularization_scale": values.regularization_scale,
                "burn_in/ema_decay": values.ema_decay,
            }
        )
        return values

    def sample_transition(self) -> int:
        timesteps = self.diffusion.timesteps
        return int(torch.randint(timesteps, (), generator=self.random["train"]))

    def real_crops(self, side_heights: bool) -> dict[int, RealCrops]:
        """real_batch_size crops of every judged plane, with side-plane heights."""
        count = self.config.train.real_batch_size
        crops = {}
        for axis in self.axes:
            sample = self.samplers[axis].sample(count, self.random["data"])
            real = RealCrops(images=sample.images.to(self.device))
            if side_heights and axis != 0:
                real.origins, real.extents = sample.origins, sample.extents
                real.heights = height_field(
                    sample.images.shape[-2:],
                    sample.origins,
                    self.spacing,
                    sample.extents,
                    self.device,
                )
            crops[axis] = real
        return crops

    def volume_height(self, origins, extents) -> torch.Tensor:
        return height_field(
            (self.patch_size,) * 3, origins, self.spacing, extents, self.device
        )

    def make_time(self, transition: int, batch: int) -> torch.Tensor:
        return torch.full((batch,), transition, device=self.device, dtype=torch.long)

    def latent(self, batch: int) -> torch.Tensor:
        channels = self.config.model.generator.latent_channels
        return randn((batch, channels), self.random["train"], self.device)

    def noise(self, shape) -> torch.Tensor:
        return torch.randn(
            tuple(shape), generator=self.random["noise"], device=self.device
        )

    def autocast(self, enabled: bool) -> AbstractContextManager:
        return torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=enabled
        )

    def generate_pair(self, transition: int, conditions: dict):
        """Run the reverse chain from noise to x_t+1 without gradients, then
        take the transition t step with them. Returns (x_t, x_t+1, x0 pred)."""
        batch = self.config.train.volume_batch_size
        shape = (batch, self.num_phases, *(self.patch_size,) * 3)
        current = self.noise(shape)
        timesteps = self.diffusion.timesteps
        with torch.no_grad(), self.autocast(self.denoiser_amp):
            for index in reversed(range(transition + 1, timesteps)):
                prediction = self.denoiser(
                    current,
                    self.make_time(index, batch),
                    self.latent(batch),
                    **conditions,
                )
                current = self.diffusion.sample_posterior(
                    current, prediction, index, self.noise(shape)
                )
        current = current.detach()
        latent = self.latent(batch)
        noise = self.noise(shape) if transition else None
        with self.autocast(self.denoiser_amp):
            prediction = self.denoiser(
                current, self.make_time(transition, batch), latent, **conditions
            )
            previous = self.diffusion.sample_posterior(
                current, prediction, transition, noise
            )
        return previous, current, prediction

    def fake_pairs(self, previous, current, inputs: StepInputs) -> dict[int, Pairs]:
        volume_height = inputs.conditions.get("height")
        pairs = {}
        for axis in self.axes:
            real = inputs.real[axis]
            height = volume_height if real.heights is not None else None
            sampled = sample_pairs(
                previous,
                current,
                axis,
                self.config.train.slice_pairs_per_plane,
                tuple(real.images.shape[-2:]),
                self.random["pairs"],
                height=height,
            )
            pairs[axis] = Pairs(*self.augment(sampled, axis, self.random["augment"]))
        return pairs

    def real_pairs(self, inputs: StepInputs) -> dict[int, Pairs]:
        pairs = {}
        for axis in self.axes:
            real = inputs.real[axis]
            images = real.images.mul(2.0).sub(1.0)
            previous, current = self.diffusion.sample_pair(
                images,
                inputs.transition,
                self.noise(images.shape),
                self.noise(images.shape),
            )
            values = (previous, current)
            if real.heights is not None:
                values += (real.heights,)
            pairs[axis] = Pairs(*self.augment(values, axis, self.random["augment"]))
        return pairs

    def regularization(self, name: str) -> tuple[bool, float, float]:
        """Lazy R1/R2 schedule and weights for one critic's next update."""
        loss = self.config.loss
        regularize = (self.updates[name] + 1) % loss.r1_every_steps == 0
        scale = self.scheduled.regularization_scale if regularize else 0.0
        return regularize, loss.r1_weight * scale, loss.r2_weight * scale

    def update_critics(self, transition, fake, real):
        """Per-axis adversarial critic losses (R1/R2 excluded) and mean R1.

        Planes no critic judged and steps without R1 give None, not zero.
        """
        critic_losses = [None] * len(AXES)
        r1_sum = None
        interval = self.config.loss.r1_every_steps
        for group, axes in self.groups.items():
            regularize, r1_weight, r2_weight = self.regularization(group)
            weight = 1.0 / (len(axes) * len(self.groups))
            critic = self.critics[group]
            optimizer = self.optimizers[group]
            optimizer.zero_grad(set_to_none=True)
            for axis in axes:
                # Copies keep the shared real pairs free of gradients for the
                # generator update.
                real_prev = real[axis].previous.detach().requires_grad_(regularize)
                real_curr = real[axis].current.detach().requires_grad_(regularize)
                fake_prev = fake[axis].previous.detach().float()
                fake_prev = fake_prev.requires_grad_(r2_weight > 0)
                fake_curr = fake[axis].current.detach().float()
                with self.autocast(self.critic_amp and not regularize):
                    real_score = critic(
                        real_prev,
                        real_curr,
                        self.make_time(transition, len(real_prev)),
                        **real[axis].conditions(),
                    )
                    fake_score = critic(
                        fake_prev,
                        fake_curr,
                        self.make_time(transition, len(fake_prev)),
                        **fake[axis].conditions(),
                    )
                    # Fake volumes take their heights from real crops, so each
                    # pair compares (section, height) samples.
                    losses = critic_objective(
                        relativistic_loss(real_score, fake_score),
                        real_score,
                        fake_score,
                        (real_prev,),
                        (fake_prev,),
                        r1_weight=r1_weight,
                        r2_weight=r2_weight,
                        interval=interval,
                    )
                prefix = f"{group}/{axis}"
                self.diagnostics[f"score/{prefix}/real"] = (
                    real_score.logits_global.detach().mean()
                )
                self.diagnostics[f"score/{prefix}/fake"] = (
                    fake_score.logits_global.detach().mean()
                )
                if regularize:
                    norms = input_gradient_norms(
                        real_score.logits_global.sum(), (real_prev, real_curr)
                    )
                    for name, norm in zip(("previous", "current"), norms, strict=True):
                        self.diagnostics[f"input_gradient/{prefix}/{name}"] = norm
                if r1_weight > 0:
                    r1 = losses.r1.detach() * weight
                    r1_sum = r1 if r1_sum is None else r1_sum + r1
                if r2_weight > 0:
                    self.diagnostics[f"r2/{prefix}"] = losses.r2.detach()
                loss = validate_loss(losses.total, f"critic {prefix}")
                self.scaler.scale(loss * weight).backward()
                critic_losses[axis] = losses.adversarial.detach() * weight
            if step_optimizer(optimizer, self.scaler, self.diagnostics, group):
                self.updates[group] += 1
        return critic_losses, r1_sum

    def adversarial_loss(self, transition, fake, real, diagnose):
        losses = []
        for group, axes in self.groups.items():
            critic = self.critics[group]
            for axis in axes:
                fake_prev, fake_curr = fake[axis].previous, fake[axis].current
                if diagnose:
                    # The preceding reverse chain is already detached.
                    fake_curr = fake_curr.detach().requires_grad_(True)
                scores = critic(
                    fake_prev,
                    fake_curr,
                    self.make_time(transition, len(fake_prev)),
                    **fake[axis].conditions(),
                )
                real_scores = critic(
                    real[axis].previous,
                    real[axis].current,
                    self.make_time(transition, len(real[axis].previous)),
                    **real[axis].conditions(),
                )
                loss = relativistic_loss(scores, real_scores)
                if diagnose:
                    # Undo only the batch mean, retaining pyramid weights.
                    norms = input_gradient_norms(
                        loss * len(fake_prev), (fake_prev, fake_curr)
                    )
                    prefix = f"generator_input_gradient/{group}/{axis}/t{transition}"
                    for name, norm in zip(("previous", "current"), norms, strict=True):
                        self.diagnostics[f"{prefix}/{name}"] = norm
                losses.append(loss / (len(axes) * len(self.groups)))
        return torch.stack(losses).sum()

    def update_generator(self, inputs, clean_probs, fake, real) -> dict:
        optimizer = self.optimizers["generator"]
        optimizer.zero_grad(set_to_none=True)
        self.critics.requires_grad_(False)
        try:
            diagnose = (
                self.updates["generator"] + 1
            ) % self.config.loss.r1_every_steps == 0
            # The objective scores fake pairs with the critics.
            with self.autocast(self.critic_amp):
                adversarial = self.adversarial_loss(
                    inputs.transition, fake, real, diagnose
                )
                extra, values = self.objective(
                    inputs, clean_probs, fake, real, diagnose
                )
                total = adversarial + extra
            self.diagnostics.update(values)
            total = validate_loss(total, "generator")
            self.scaler.scale(total).backward()
            self.generator_updated = step_optimizer(
                optimizer, self.scaler, self.diagnostics, "generator"
            )
            if self.generator_updated:
                self.updates["generator"] += 1
        finally:
            self.critics.requires_grad_(True)
        return {"adversarial": adversarial.detach(), "total": total.detach()}


def materialize(value):
    """Turn every tensor in a nested structure into a Python number (one sync)."""
    tensors = []

    def collect(item):
        if isinstance(item, torch.Tensor):
            tensors.append(item.detach())
        elif is_dataclass(item):
            for entry in fields(item):
                collect(getattr(item, entry.name))
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                collect(child)

    collect(value)
    if not tensors:
        return value
    device = tensors[0].device
    dtype = wide_dtype(device)
    numbers = iter(
        torch.stack([t.to(device=device, dtype=dtype).reshape(()) for t in tensors])
        .cpu()
        .tolist()
    )

    def rebuild(item):
        if isinstance(item, torch.Tensor):
            number = next(numbers)
            if item.dtype == torch.bool:
                return bool(number)
            return float(number) if item.dtype.is_floating_point else int(number)
        if isinstance(item, dict):
            return {key: rebuild(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return type(item)(rebuild(child) for child in item)
        return item

    return rebuild(value)
