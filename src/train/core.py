import copy
from dataclasses import dataclass, field

import torch
from torch import nn

from src.axes import AXES, PLANES
from src.config.schema import group_planes
from src.data.augment import SectionAugment, build_plane_transforms
from src.data.crops import CropSampler
from src.data.height import build_heights
from src.device import gpu_amp, wide_dtype
from src.export import build_denoiser, build_diffusion, export_payload
from src.model.critic import PairCritic2D
from src.rng import RunRandom, randn
from src.train.burn_in import burn_in_values
from src.train.losses.gan import critic_objective, relativistic_loss
from src.train.losses.group_statistics import group_statistics_loss
from src.train.losses.mirror_balance import mirror_balance_loss
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

    def conditions(self):
        return {} if self.height is None else {"height": self.height}


@dataclass
class StepInputs:
    step: int
    transition: int
    real: dict[int, RealCrops]
    conditions: dict  # denoiser conditions for the generated volume
    extra: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


def build_ema(model):
    ema = copy.deepcopy(model)
    ema.eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(ema, model, decay):
    for ema_parameter, parameter in zip(
        ema.parameters(), model.parameters(), strict=True
    ):
        ema_parameter.lerp_(parameter, 1.0 - decay)
    for ema_buffer, buffer in zip(ema.buffers(), model.buffers(), strict=True):
        ema_buffer.copy_(buffer)


def build_critics(config):
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
            for group in group_planes(config)
        }
    )


def validate_loss(loss, name):
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError(f"non-finite {name} loss; optimizer was not stepped.")
    return loss


def step_optimizer(optimizer, scaler, name):
    """Step unless the gradient is not finite (an error without AMP); return
    whether it stepped and the gradient metrics."""
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
    if not finite and not scaler.is_enabled():
        raise FloatingPointError(
            f"non-finite {name} gradient; optimizer was not stepped."
        )
    scaler.step(optimizer)
    return finite, {
        f"gradient/{name}": norm if finite else None,
        f"skipped/{name}": int(not finite),
    }


def schedule_ramp(step, ramp):
    if ramp == 0:
        return 1.0
    return min((step + 1) / ramp, 1.0)


def sample_sections(volumes, axis, count, generator):
    dim = axis + 2
    batch, size = volumes[0].shape[0], volumes[0].shape[dim]
    choices = torch.randint(batch * size, (count,), generator=generator)
    choices = choices.to(volumes[0].device)
    indices = (choices // size, slice(None), choices % size)
    return tuple(volume.movedim(dim, 2)[indices] for volume in volumes)


class Trainer:
    """State and update rules shared by both stages.

    One step: burn-in values, the stage's conditions and real crops, a generated
    pair (x_t, x_t+1) and its clean prediction, plane critic updates, then a
    generator update. Stage trainers (src/train/lr.py, sr.py) supply the
    conditions and the non-adversarial objective.
    """

    def __init__(self, config, device, planes):
        self.config = config
        self.device = torch.device(device)
        self.random = RunRandom(config.train.seed, device)
        self.groups = group_planes(config)
        self.axes = tuple(
            sorted(axis for axes in self.groups.values() for axis in axes)
        )
        self.patch_size = config.patch_size
        self.num_phases = config.data.num_phases
        self.height = config.conditioning.height_enabled
        self.samplers = {
            PLANES.index(plane): CropSampler(
                images, config.data.crop_size, self.patch_size, self.num_phases
            )
            for plane, images in planes.items()
        }
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
        self.denoiser_amp = gpu_amp(mixed, self.device)
        self.critic_amp = mixed and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.denoiser_amp)
        self.augment = SectionAugment(
            build_plane_transforms(config), config.augmentation.probability
        )
        self.spacing = config.data.crop_size / self.patch_size
        self.updates = {name: 0 for name in (*self.critics, "generator")}
        self.completed_steps = 0

    @staticmethod
    def _adam(module, lr, betas):
        return torch.optim.Adam(module.parameters(), lr=lr, betas=betas, fused=True)

    def prepare(self, step, transition):
        raise NotImplementedError

    def objective(self, inputs, clean_probs):
        """Non-adversarial generator terms: (loss, metrics)."""
        raise NotImplementedError

    def export_payload(self, step, model):
        """The stage's export of EMA weights model at step (SR adds its LR)."""
        return export_payload(self.config, step, model)

    def stage_state(self):
        return {}

    def state_dict(self):
        return {
            "models": {
                "generator": self.denoiser.state_dict(),
                "ema": self.ema.state_dict(),
                "critics": {
                    name: model.state_dict() for name, model in self.critics.items()
                },
            },
            "optimizers": {
                name: opt.state_dict() for name, opt in self.optimizers.items()
            },
            "scaler": self.scaler.state_dict(),
            "updates": dict(self.updates),
            "rng": self.random.state(),
            **self.stage_state(),
        }

    def load_state_dict(self, state):
        models = state["models"]
        if (
            models["critics"].keys() != self.critics.keys()
            or state["optimizers"].keys() != self.optimizers.keys()
        ):
            raise ValueError("critic groups do not match the checkpoint.")
        self.denoiser.load_state_dict(models["generator"])
        self.ema.load_state_dict(models["ema"])
        for name, critic in self.critics.items():
            critic.load_state_dict(models["critics"][name])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(state["optimizers"][name])
        if state["scaler"]:
            self.scaler.load_state_dict(state["scaler"])
        self.updates = dict(state["updates"])
        self.random.load_state(state["rng"])
        self.completed_steps = state["step"]

    def step(self, step, transition=None):
        scheduled = self.apply_burn_in(step)
        self.denoiser.train()
        self.critics.train()
        inputs = self.prepare(step, transition)
        previous, current, prediction = self.generate_pair(
            inputs.transition, inputs.conditions
        )
        clean_probs = (prediction + 1.0) * 0.5
        fake = self.fake_pairs(previous, current, inputs)
        real = self.real_pairs(inputs)
        critic_metrics = self.update_critics(inputs.transition, fake, real, scheduled)
        updated, generator_metrics = self.update_generator(
            inputs, clean_probs, fake, real
        )
        self.scaler.update()
        structure = self.measure_structure(step, inputs, clean_probs)
        self.completed_steps = step + 1
        if updated:
            update_ema(self.ema, self.denoiser, scheduled.ema_decay)
        return materialize(
            {
                "sampling/transition": inputs.transition,
                **inputs.diagnostics,
                **critic_metrics,
                **generator_metrics,
                **structure,
                "amp/scale": self.scaler.get_scale(),
            }
        )

    def apply_burn_in(self, step):
        values = burn_in_values(self.config, step)
        for name, optimizer in self.optimizers.items():
            lr = values.generator_lr if name == "generator" else values.critic_lr
            for group in optimizer.param_groups:
                group["lr"] = lr
                group["betas"] = values.betas
        return values

    def sample_transition(self):
        timesteps = self.diffusion.timesteps
        return int(torch.randint(timesteps, (), generator=self.random["train"]))

    def real_crops(self):
        """real_batch_size crops of every judged plane, with side-plane heights."""
        count = self.config.train.real_batch_size
        crops = {}
        for axis in self.axes:
            sample = self.samplers[axis].sample(count, self.random["data"])
            real = RealCrops(images=sample.images.to(self.device))
            if self.height and axis != 0:
                real.origins = sample.height_origins
                real.extents = sample.height_extents
                real.heights = build_heights(
                    sample.images.shape[-2:],
                    sample.height_origins,
                    self.spacing,
                    sample.height_extents,
                    self.device,
                )
            crops[axis] = real
        return crops

    def volume_height(self, origins, extents):
        return build_heights(
            (self.patch_size,) * 3, origins, self.spacing, extents, self.device
        )

    def generate_pair(self, transition, conditions):
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

    def make_time(self, transition, batch):
        return torch.full((batch,), transition, device=self.device, dtype=torch.long)

    def latent(self, batch):
        channels = self.config.model.generator.latent_channels
        return randn((batch, channels), self.random["train"], self.device)

    def noise(self, shape):
        return torch.randn(shape, generator=self.random["noise"], device=self.device)

    def autocast(self, enabled):
        return torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=enabled
        )

    def fake_pairs(self, previous, current, inputs):
        volume_height = inputs.conditions.get("height")
        pairs = {}
        for axis in self.axes:
            volumes = (previous, current)
            if volume_height is not None and inputs.real[axis].heights is not None:
                volumes += (volume_height.detach(),)
            sampled = sample_sections(
                volumes,
                axis,
                self.config.train.slice_pairs_per_plane,
                self.random["pairs"],
            )
            pairs[axis] = Pairs(*self.augment(sampled, axis, self.random["augment"]))
        return pairs

    def real_pairs(self, inputs):
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

    def update_critics(self, transition, fake, real, scheduled):
        """Step every plane-group critic once; return its metrics.

        Planes no critic judged and steps without R1 give None, not zero.
        """
        plane_losses = [None] * len(AXES)
        r1_total = None
        metrics = {}
        for group, axes in self.groups.items():
            regularize, r1_weight, r2_weight = self.regularization(group, scheduled)
            weight = 1.0 / (len(axes) * len(self.groups))
            optimizer = self.optimizers[group]
            optimizer.zero_grad(set_to_none=True)
            for axis in axes:
                losses, real_score, fake_score = self.critic_losses(
                    group,
                    transition,
                    real[axis],
                    fake[axis],
                    regularize,
                    r1_weight,
                    r2_weight,
                )
                prefix = f"{group}/{axis}"
                metrics[f"score/{prefix}/real"] = real_score[0].detach().mean()
                metrics[f"score/{prefix}/fake"] = fake_score[0].detach().mean()
                if r1_weight > 0:
                    r1 = losses.r1.detach() * weight
                    r1_total = r1 if r1_total is None else r1_total + r1
                if r2_weight > 0:
                    metrics[f"r2/{prefix}"] = losses.r2.detach()
                loss = validate_loss(losses.total, f"critic {prefix}")
                self.scaler.scale(loss * weight).backward()
                plane_losses[axis] = losses.adversarial.detach() * weight
            stepped, gradient = step_optimizer(optimizer, self.scaler, group)
            metrics.update(gradient)
            if stepped:
                self.updates[group] += 1
        metrics.update(
            {
                "loss/critic": sum(loss for loss in plane_losses if loss is not None),
                "loss/r1": r1_total,
                **{
                    f"critic_plane/{PLANES[axis]}": loss
                    for axis, loss in enumerate(plane_losses)
                },
            }
        )
        return metrics

    def regularization(self, group, scheduled):
        """Whether this update of group's critic is a lazy R1/R2 one, and the
        R1 and R2 weights it uses."""
        loss = self.config.loss
        regularize = (self.updates[group] + 1) % loss.r1_every_steps == 0
        scale = scheduled.regularization_scale if regularize else 0.0
        return regularize, loss.r1_weight * scale, loss.r2_weight * scale

    def critic_losses(
        self, group, transition, real, fake, regularize, r1_weight, r2_weight
    ):
        """Adversarial and R1/R2 losses of one plane, with the real and fake scores."""
        # Detached copies keep the shared pairs gradient-free for the generator.
        real = Pairs(
            real.previous.detach().requires_grad_(r1_weight > 0),
            real.current.detach(),
            real.height,
        )
        fake = Pairs(
            fake.previous.detach().float().requires_grad_(r2_weight > 0),
            fake.current.detach().float(),
            fake.height,
        )
        critic = self.critics[group]
        with self.autocast(self.critic_amp and not regularize):
            real_score = self.score(critic, real, transition)
            fake_score = self.score(critic, fake, transition)
            losses = critic_objective(
                relativistic_loss(real_score, fake_score),
                real_score,
                fake_score,
                (real.previous,),
                (fake.previous,),
                r1_weight=r1_weight,
                r2_weight=r2_weight,
                interval=self.config.loss.r1_every_steps,
            )
        return losses, real_score, fake_score

    def score(self, critic, pairs, transition):
        return critic(
            pairs.previous,
            pairs.current,
            self.make_time(transition, len(pairs.previous)),
            **pairs.conditions(),
        )

    def update_generator(self, inputs, clean_probs, fake, real):
        """Step the generator once; return whether it stepped and its metrics."""
        optimizer = self.optimizers["generator"]
        optimizer.zero_grad(set_to_none=True)
        self.critics.requires_grad_(False)
        try:
            # The critics score the fake pairs, so the critics' precision applies.
            with self.autocast(self.critic_amp):
                adversarial = self.adversarial_loss(inputs.transition, fake, real)
                extra, metrics = self.objective(inputs, clean_probs)
                total = adversarial + extra
            total = validate_loss(total, "generator")
            self.scaler.scale(total).backward()
            stepped, gradient = step_optimizer(optimizer, self.scaler, "generator")
            if stepped:
                self.updates["generator"] += 1
        finally:
            self.critics.requires_grad_(True)
        return stepped, {
            "loss/generator": adversarial.detach(),
            "loss/generator_total": total.detach(),
            **metrics,
            **gradient,
        }

    def adversarial_loss(self, transition, fake, real):
        losses = []
        for group, axes in self.groups.items():
            critic = self.critics[group]
            for axis in axes:
                loss = relativistic_loss(
                    self.score(critic, fake[axis], transition),
                    self.score(critic, real[axis], transition),
                )
                losses.append(loss / (len(axes) * len(self.groups)))
        return torch.stack(losses).sum()

    def group_statistics_term(self, inputs, clean_probs, low=None):
        """The ramped group-statistics loss on transition 0 (None otherwise) and
        its metrics. SR passes its LR input low: differences low already has
        are allowed on top."""
        group_statistics = self.config.loss.group_statistics
        ramp = schedule_ramp(inputs.step, group_statistics.ramp_steps)
        if not (group_statistics.weight > 0 and inputs.transition == 0 and ramp > 0):
            return None, {}
        value, metrics = group_statistics_loss(
            clean_probs,
            self.groups,
            group_statistics.max_gap,
            group_statistics.tolerance,
            self.height,
            low,
        )
        metrics["loss/group_statistics"] = value.detach()
        metrics["group_statistics/ramp"] = value.new_tensor(ramp)
        return group_statistics.weight * ramp * value, metrics

    def mirror_balance_term(self, clean_probs):
        mirror_balance = self.config.loss.mirror_balance
        if not mirror_balance.weight > 0:
            return None, {}
        # The x0 prediction of every transition must keep the mirror symmetry.
        value, metrics = mirror_balance_loss(
            clean_probs, mirror_balance.max_gap, mirror_balance.tolerance
        )
        metrics["loss/mirror_balance"] = value.detach()
        return mirror_balance.weight * value, metrics

    def measure_structure(self, step, inputs, clean_probs):
        every = self.config.train.structure_every_steps
        if not every or (step + 1) % every:
            return {}
        real = {axis: crops.images for axis, crops in inputs.real.items()}
        return structure_metrics(clean_probs, real, self.groups)


def materialize(scalars):
    """The scalars with each tensor turned into a Python number (one sync)."""
    keys = [key for key, value in scalars.items() if isinstance(value, torch.Tensor)]
    result = dict(scalars)
    if not keys:
        return result
    device = scalars[keys[0]].device
    dtype = wide_dtype(device)
    numbers = (
        torch.stack(
            [
                scalars[key].detach().to(device=device, dtype=dtype).reshape(())
                for key in keys
            ]
        )
        .cpu()
        .tolist()
    )
    for key, number in zip(keys, numbers, strict=True):
        kind = scalars[key].dtype
        if kind == torch.bool:
            result[key] = bool(number)
        else:
            result[key] = float(number) if kind.is_floating_point else int(number)
    return result
