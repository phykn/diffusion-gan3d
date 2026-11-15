import math

import torch
from tqdm import tqdm

from src.device import gpu_amp
from src.model.denoiser import Denoiser3D
from src.rng import make_generator, randn
from src.sampling.plan import build_tile_window

LABEL_CHUNK_VOXELS = 1024**2
# An fp16 state scored the same but cut the measured peak memory by only 1-2%.
STATE_DTYPE = torch.float32
BATCH_AND_CHANNELS = (slice(None), slice(None))


class Predictor:
    def __init__(self, model, diffusion, config, device):
        self.model = model
        self.diffusion = diffusion
        self.device = device
        self.num_phases = config.data.num_phases
        self.latent_channels = config.model.generator.latent_channels
        self.amp = gpu_amp(config.train.mixed_precision, device)

    def __call__(self, noisy, time, latent, guidance, **conditions):
        conditions = {
            name: value for name, value in conditions.items() if value is not None
        }
        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=self.amp
        ):
            logits = self.model.predict_guided_logits(
                noisy, time, latent, guidance, **conditions
            )
        return Denoiser3D.decode(logits.float())


class _BlendSlab:
    # Blend sums of overlapping tiles. Only one tile deep along z: a finished
    # layer is cleared and its rows are reused for the next one.
    def __init__(self, plan, num_phases, device, guided):
        shape = (plan.tile_shape[0], *plan.region_shape[1:])
        self.weight = torch.zeros((1, 1, *shape), device=device)
        self.prediction = torch.zeros((1, num_phases, *shape), device=device)
        self.guide_step = torch.zeros_like(self.prediction) if guided else None

    def add(self, tile, window, prediction, guide_step=None):
        weighted = [
            (self.prediction, prediction.float() * window),
            (self.weight, window),
        ]
        if guide_step is not None:
            weighted.append((self.guide_step, guide_step.float() * window))
        weighted = [(total, values.to(total.device)) for total, values in weighted]
        tile_start = tile.source[0].start
        for region, slab_region in self.split(tile.source):
            start, stop = region[0].start - tile_start, region[0].stop - tile_start
            for total, values in weighted:
                total[(*BATCH_AND_CHANNELS, *slab_region)].add_(
                    values[:, :, start:stop]
                )

    def split(self, region):
        depth = self.prediction.shape[2]
        start, stop = region[0].start, region[0].stop
        while start < stop:
            slab_start = start % depth
            end = min(stop, start + depth - slab_start)
            yield (
                (slice(start, end), *region[1:]),
                (slice(slab_start, slab_start + end - start), *region[1:]),
            )
            start = end

    def clear(self, slab_region):
        self.prediction[slab_region].zero_()
        self.weight[slab_region].zero_()
        if self.guide_step is not None:
            self.guide_step[slab_region].zero_()


class TileSampler:
    def __init__(self, predictor):
        self.predictor = predictor

    @torch.no_grad()
    def sample(
        self,
        plan,
        *,
        seed,
        conditions,
        guidance=1.0,
        guide=None,
        probabilities=False,
        storage="cpu",
        progress=False,
        step_hook=None,
    ):
        predictor = self.predictor
        device = predictor.device
        num_phases = predictor.num_phases
        timesteps = predictor.diffusion.timesteps
        generator = make_generator(
            int(torch.randint(2**63 - 1, ())) if seed is None else seed
        )
        storage_device = device if storage == "cuda" else torch.device("cpu")
        tiles = plan.tiles
        state = torch.empty(
            (1, num_phases, *plan.region_shape),
            device=storage_device,
            dtype=STATE_DTYPE,
        )
        for tile in tiles:
            shape = (1, num_phases, *(part.stop - part.start for part in tile.target))
            state[(*BATCH_AND_CHANNELS, *tile.target)] = randn(
                shape, generator, storage_device, STATE_DTYPE
            )
        output = torch.empty(
            (num_phases, *plan.output_shape) if probabilities else plan.output_shape,
            dtype=torch.float32 if probabilities else torch.uint8,
        )
        slab = _BlendSlab(plan, num_phases, storage_device, guided=guide is not None)
        windows = {}
        with tqdm(total=timesteps, desc="Generate", disable=not progress) as bar:
            for transition in reversed(range(timesteps)):
                time = torch.full((1,), transition, device=device, dtype=torch.long)
                latent = randn((1, predictor.latent_channels), generator, device)
                is_last_step = transition == 0
                layer_start = 0
                for index, tile in enumerate(tiles):
                    noisy = state[(*BATCH_AND_CHANNELS, *tile.source)].to(
                        device=device, dtype=torch.float32
                    )
                    tile_conditions = conditions(tile)
                    tile_guide = None
                    if guide is not None and not is_last_step:
                        tile_guide = guide(index, tile)
                    guide_step = None
                    if tile_guide is None:
                        prediction = predictor(
                            noisy, time, latent, guidance, **tile_conditions
                        )
                    else:
                        prediction, guide_step = tile_guide(
                            predictor, noisy, time, latent, guidance, tile_conditions
                        )
                    window_key = (tile.shape, tile.blend_widths)
                    if window_key not in windows:
                        windows[window_key] = build_tile_window(
                            tile, plan.overlap, device
                        )
                    slab.add(tile, windows[window_key], prediction, guide_step)
                    # Once the next z layer starts, no later tile reads this one.
                    next_z = (
                        tiles[index + 1].source[0].start
                        if index + 1 < len(tiles)
                        else plan.region_shape[0]
                    )
                    if next_z != tile.source[0].start:
                        self._finish_layer(
                            state,
                            plan,
                            tiles[layer_start : index + 1],
                            slab,
                            transition,
                            next_z,
                            output if is_last_step else None,
                            generator,
                        )
                        layer_start = index + 1
                if step_hook is not None:
                    step_hook(transition)
                bar.update()
        return output

    def _finish_layer(
        self, state, plan, tiles, slab, transition, stop, output, generator
    ):
        diffusion = self.predictor.diffusion
        for tile in tiles:
            finished = (slice(tile.source[0].start, stop), *tile.target[1:])
            for region, slab_region in slab.split(finished):
                slab_region = (*BATCH_AND_CHANNELS, *slab_region)
                weights = slab.weight[slab_region]
                clean = slab.prediction[slab_region] / weights
                if output is None:
                    state_region = (*BATCH_AND_CHANNELS, *region)
                    current = state[state_region].float()
                    noise = randn(current.shape, generator, current.device)
                    previous = diffusion.sample_posterior(
                        current, clean, transition, noise
                    )
                    if slab.guide_step is not None:
                        previous = previous - slab.guide_step[slab_region] / weights
                    state[state_region] = previous.to(state.dtype)
                else:
                    _write_output(output, region, clean, plan.offsets)
                slab.clear(slab_region)


def _write_output(output, region, clean, offsets):
    source, destination = [], []
    for part, size, offset in zip(region, output.shape[-3:], offsets, strict=True):
        first, last = max(part.start, offset), min(part.stop, offset + size)
        if first >= last:
            return
        source.append(slice(first - part.start, last - part.start))
        destination.append(slice(first - offset, last - offset))
    values = clean[(*BATCH_AND_CHANNELS, *source)]
    if output.ndim == 3:
        output[tuple(destination)] = _pick_labels(values.squeeze(0))
    else:
        fractions = _normalize_fractions(values.clone()).squeeze(0)
        output[(slice(None), *destination)] = fractions


def _pick_labels(channels):
    shape = channels.shape[1:]
    labels = torch.empty(shape, dtype=torch.uint8, device="cpu")
    depth = min(shape[0], max(1, LABEL_CHUNK_VOXELS // math.prod(shape[1:])))
    for start in range(0, shape[0], depth):
        layers = slice(start, start + depth)
        labels[layers].copy_(channels[:, layers].argmax(0).to(torch.uint8))
    return labels


def _normalize_fractions(clean):
    fractions = clean.float().add_(1).mul_(0.5).clamp_(0, 1)
    total = fractions.sum(1, keepdim=True).clamp_min_(torch.finfo(fractions.dtype).eps)
    return fractions.div_(total)
