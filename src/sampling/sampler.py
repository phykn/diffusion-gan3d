import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from tqdm import tqdm

from src.device import gpu_amp
from src.model.denoiser import Denoiser3D
from src.model.diffusion import Diffusion
from src.rng import make_generator, randn
from src.sampling.plan import Tile, TilePlan, tile_window

LABEL_CHUNK_VOXELS = 1024**2
# The reverse-diffusion state between steps. An fp16 state scored the same
# but cut the measured peak by only 1-2%, the device heap dwarfing the state.
STATE_DTYPE = torch.float32

# conditions(index, tile) -> model conditions for that tile
TileConditions = Callable[[int, Tile], dict]
# guide(index, tile) -> a guide for that tile, or None
TileGuide = Callable[[int, Tile], object]


@torch.no_grad()
def labels_from_channels(channels: torch.Tensor) -> torch.Tensor:
    """Chunked argmax of [C, D, H, W] values into uint8 labels on the CPU."""
    if channels.ndim != 4 or channels.numel() == 0:
        raise ValueError("channels must be a non-empty C,D,H,W tensor.")
    shape = channels.shape[1:]
    labels = torch.empty(shape, dtype=torch.uint8, device="cpu")
    depth = min(shape[0], max(1, LABEL_CHUNK_VOXELS // math.prod(shape[1:])))
    for start in range(0, shape[0], depth):
        region = slice(start, start + depth)
        labels[region].copy_(channels[:, region].argmax(0).to(torch.uint8))
    return labels


@torch.no_grad()
def clean_to_probs_(clean: torch.Tensor) -> torch.Tensor:
    """Clean values in [-1, 1] to phase fractions, in place on fp32 storage."""
    probs = clean.float().add_(1).mul_(0.5).clamp_(0, 1)
    total = probs.sum(1, keepdim=True).clamp_min_(torch.finfo(probs.dtype).eps)
    return probs.div_(total)


@dataclass(frozen=True)
class Predictor:
    """A frozen denoiser on its device, with CFG and autocast."""

    model: Denoiser3D
    diffusion: Diffusion
    device: torch.device
    num_phases: int
    latent_channels: int
    amp: bool

    @classmethod
    def from_config(cls, model, diffusion, config, device) -> "Predictor":
        return cls(
            model=model,
            diffusion=diffusion,
            device=device,
            num_phases=config.data.num_phases,
            latent_channels=config.model.generator.latent_channels,
            amp=gpu_amp(config.train.mixed_precision, device),
        )

    def __call__(self, values, time, latent, guidance=1.0, **conditions):
        conditions = {k: v for k, v in conditions.items() if v is not None}
        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=self.amp
        ):
            logits = self.model.guided_logits(
                values, time, latent, guidance, **conditions
            )
        return Denoiser3D.decode(logits.float())


class _Slab:
    """Circular blend sums one tile deep along z."""

    def __init__(self, plan: TilePlan, phases: int, device, guided: bool):
        shape = (plan.tile_edge[0], *plan.shape[1:])
        self.weight = torch.zeros((1, 1, *shape), device=device)
        self.pred = torch.zeros((1, phases, *shape), device=device)
        self.step = torch.zeros_like(self.pred) if guided else None

    def regions(self, region):
        depth = self.pred.shape[2]
        start, stop = region[0].start, region[0].stop
        while start < stop:
            offset = start % depth
            end = min(stop, start + depth - offset)
            yield (
                (slice(start, end), *region[1:]),
                (slice(offset, offset + end - start), *region[1:]),
            )
            start = end

    def add(self, tile: Tile, window, pred, step=None):
        staged = [(self.pred, pred.float() * window), (self.weight, window)]
        if step is not None:
            staged.append((self.step, step.float() * window))
        staged = [(total, values.to(total.device)) for total, values in staged]
        for global_region, local in self.regions(tile.source):
            start = global_region[0].start - tile.source[0].start
            stop = global_region[0].stop - tile.source[0].start
            for total, values in staged:
                total[(slice(None), slice(None), *local)].add_(values[:, :, start:stop])


class TileSampler:
    """The one volume reverse-diffusion loop, for LR and SR, tiled or not.

    Tiles are visited in z/y/x order. Their clean predictions (and anchor guide
    steps) are blended with the plan's windows into a slab one tile deep; once
    the next z layer starts, no later tile reads the finished part of the slab,
    so its posterior step is taken there and then.
    """

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    @torch.no_grad()
    def sample(
        self,
        plan: TilePlan,
        *,
        seed: int | None,
        conditions: TileConditions,
        guidance: float = 1.0,
        guide: TileGuide | None = None,
        probabilities: bool = False,
        storage: str = "cpu",
        progress: bool = False,
        step_hook: Callable[[int], None] | None = None,
    ) -> torch.Tensor:
        """Labels uint8 (D, H, W) or fractions float32 (C, D, H, W), on the CPU."""
        predictor = self.predictor
        device = predictor.device
        phases = predictor.num_phases
        timesteps = predictor.diffusion.timesteps
        generator = make_generator(
            int(torch.randint(2**63 - 1, ())) if seed is None else seed
        )
        store = device if storage == "cuda" else torch.device("cpu")
        state = torch.empty((1, phases, *plan.shape), device=store, dtype=STATE_DTYPE)
        for tile in plan.tiles:
            shape = (1, phases, *(r.stop - r.start for r in tile.target))
            state[(slice(None), slice(None), *tile.target)] = randn(
                shape, generator, store, STATE_DTYPE
            )
        output = torch.empty(
            (phases, *plan.output_shape) if probabilities else plan.output_shape,
            dtype=torch.float32 if probabilities else torch.uint8,
        )
        slab = _Slab(plan, phases, store, guided=guide is not None)
        windows = {}
        with tqdm(total=timesteps, desc="Generate", disable=not progress) as bar:
            for transition in reversed(range(timesteps)):
                time = torch.full((1,), transition, device=device, dtype=torch.long)
                latent = randn((1, predictor.latent_channels), generator, device)
                final = transition == 0
                layer = 0
                for index, tile in enumerate(plan.tiles):
                    values = state[(slice(None), slice(None), *tile.source)].to(
                        device=device, dtype=torch.float32
                    )
                    tile_conditions = conditions(index, tile)
                    tile_guide = None if guide is None or final else guide(index, tile)
                    step = None
                    if tile_guide is None:
                        pred = predictor(
                            values, time, latent, guidance, **tile_conditions
                        )
                    else:
                        pred, step = tile_guide(
                            predictor, values, time, latent, guidance, tile_conditions
                        )
                    key = (tile.shape, tile.margins)
                    if key not in windows:
                        windows[key] = tile_window(tile, plan.overlap, device)
                    slab.add(tile, windows[key], pred, step)
                    next_z = (
                        plan.tiles[index + 1].source[0].start
                        if index + 1 < len(plan.tiles)
                        else plan.shape[0]
                    )
                    if next_z != tile.source[0].start:
                        self._flush(
                            state,
                            plan,
                            plan.tiles[layer : index + 1],
                            slab,
                            transition,
                            next_z,
                            output if final else None,
                            generator,
                        )
                        layer = index + 1
                if step_hook is not None:
                    step_hook(transition)
                bar.update()
        return output

    def _flush(self, state, plan, tiles, slab, transition, stop, output, generator):
        diffusion = self.predictor.diffusion
        for tile in tiles:
            target = (slice(tile.source[0].start, stop), *tile.target[1:])
            for global_region, local in slab.regions(target):
                region = (slice(None), slice(None), *local)
                weights = slab.weight[region]
                if not bool((weights > 0).all()):
                    raise RuntimeError("blend weights must cover the output slab.")
                clean = slab.pred[region] / weights
                where = (slice(None), slice(None), *global_region)
                if output is None:
                    previous = state[where].float()
                    noise = randn(previous.shape, generator, previous.device)
                    updated = diffusion.sample_posterior(
                        previous, clean, transition, noise
                    )
                    if slab.step is not None:
                        updated = updated - slab.step[region] / weights
                    state[where] = updated.to(state.dtype)
                else:
                    _write_output(output, global_region, clean, plan.margin)
                slab.pred[region].zero_()
                if slab.step is not None:
                    slab.step[region].zero_()
                weights.zero_()


def _write_output(output, target, clean, margin):
    source, destination = [], []
    for part, size in zip(target, output.shape[-3:], strict=True):
        left, right = max(part.start, margin), min(part.stop, margin + size)
        if left >= right:
            return
        source.append(slice(left - part.start, right - part.start))
        destination.append(slice(left - margin, right - margin))
    values = clean[(slice(None), slice(None), *source)]
    if output.ndim == 3:
        output[tuple(destination)] = labels_from_channels(values.squeeze(0))
    else:
        output[(slice(None), *destination)] = clean_to_probs_(values.clone()).squeeze(0)
