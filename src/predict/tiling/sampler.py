from functools import partial

import torch
from tqdm import tqdm

from src.predict.generator import Generator
from src.predict.tiling.fusion import Fusion, add_prediction, make_fusion
from src.predict.tiling.layout import Tile, TilePlan
from src.predict.tiling.state import TileBuffer, VolumeState, write_output
from src.predict.volume_condition import VolumeCondition


class TileSampler:
    """Own LR/SR global states, diffusion steps, fusion, and output buffers."""

    def __init__(self, generator: Generator) -> None:
        self.generator = generator

    def sample(
        self,
        tiles: tuple[Tile, ...],
        plan: TilePlan,
        base: VolumeCondition | None,
        domain: torch.Tensor,
        probabilities: bool = False,
        storage: str = "cpu",
        progress: bool = False,
        guidance: float = 1.0,
        tile_conditions=None,
        tile_guidance=None,
    ) -> torch.Tensor:
        """tile_guidance(tile, index) returns the tile's AnchorGuidance or None."""
        generator = self.generator
        device = generator.device if storage == "cuda" else torch.device("cpu")
        # One state suffices: a z-layer is updated only after every tile that
        # reads it, and later layers read only z at or above their own start.
        current = VolumeState(generator.num_phases, plan.shape, device)
        shape = tuple(n - 2 * plan.margin for n in plan.shape)
        output = torch.empty(
            (generator.num_phases, *shape) if probabilities else shape,
            dtype=torch.float32 if probabilities else torch.uint8,
        )
        self.fill_noise(current, tiles)
        tile_buffer = TileBuffer(
            generator.num_phases,
            tuple(min(plan.tile_size, size) for size in plan.shape),
            current.values.device.type == "cpu" and generator.device.type == "cuda",
        )
        fusion = make_fusion(
            plan,
            generator.num_phases,
            current.values.device,
            guided=tile_guidance is not None,
        )
        with tqdm(
            total=generator.diffusion.timesteps,
            desc="Scale up",
            disable=not progress,
        ) as bar:
            for transition in reversed(range(generator.diffusion.timesteps)):
                time = torch.full(
                    (1,),
                    transition,
                    device=generator.device,
                    dtype=torch.long,
                )
                latent = torch.randn(
                    1,
                    generator.latent_channels,
                    device=generator.device,
                    dtype=torch.float32,
                )
                # A preserved base is already re-noised at the end of each step.
                first = transition == generator.diffusion.timesteps - 1
                if base is not None and (first or not base.preserve):
                    base.condition(
                        current, generator.diffusion, transition + 1, plan.tile_size
                    )
                final_output = output if transition == 0 else None
                self.step(
                    current,
                    tiles,
                    time,
                    latent,
                    domain,
                    transition,
                    plan,
                    final_output,
                    fusion,
                    tile_buffer,
                    guidance=guidance,
                    tile_conditions=tile_conditions,
                    tile_guidance=tile_guidance,
                    base=base,
                )
                bar.update()
        return output

    def fill_noise(
        self,
        state: VolumeState,
        tiles: tuple[Tile, ...],
    ) -> None:
        generator = self.generator
        for tile in tiles:
            shape = tuple(region.stop - region.start for region in tile.target)
            # Draw on the model device so a seed does not depend on where
            # storage="auto" placed the state.
            noise = torch.randn(
                1,
                generator.num_phases,
                *shape,
                device=generator.device,
                dtype=torch.float32,
            )
            state.write(tile.target, noise)

    def step(
        self,
        current: VolumeState,
        tiles: tuple[Tile, ...],
        time: torch.Tensor,
        latent: torch.Tensor,
        domain: torch.Tensor,
        transition: int,
        plan: TilePlan,
        output: torch.Tensor | None,
        fusion: Fusion,
        tile_buffer: TileBuffer,
        guidance: float = 1.0,
        tile_conditions=None,
        tile_guidance=None,
        base: VolumeCondition | None = None,
    ) -> None:
        generator = self.generator
        model = partial(generator.predict, guidance=guidance)
        layer_start = 0
        for index, tile in enumerate(tiles):
            values = tile_buffer.read(
                current,
                tile.source,
                generator.device,
            )
            pred = (
                None if base is None else base.prediction(tile.source, generator.device)
            )
            step = None
            if pred is None:
                conditions = (
                    {} if tile_conditions is None else tile_conditions(tile, index)
                )
                guide = None if tile_guidance is None else tile_guidance(tile, index)
                if guide is None:
                    pred = model(values, time, latent, domain=domain, **conditions)
                else:
                    pred, step = guide(
                        model, values, time, latent, domain=domain, **conditions
                    )
            expected = (1, generator.num_phases, *values.shape[-3:])
            if pred.shape != expected:
                raise ValueError(f"model prediction must have shape {expected}.")
            add_prediction(fusion, tile, pred, tile_buffer, plan.overlap, step)
            next_z = (
                tiles[index + 1].source[0].start
                if index + 1 < len(tiles)
                else plan.shape[0]
            )
            if next_z != tile.source[0].start:
                self.flush_slab(
                    current,
                    tiles[layer_start : index + 1],
                    fusion,
                    transition,
                    output,
                    next_z,
                    base=base,
                    margin=plan.margin,
                )
                layer_start = index + 1

    def flush_slab(
        self,
        current: VolumeState,
        tiles: tuple[Tile, ...],
        fusion: Fusion,
        transition: int,
        output: torch.Tensor | None,
        stop: int,
        base: VolumeCondition | None = None,
        margin: int = 0,
    ) -> None:
        generator = self.generator
        for tile in tiles:
            target = (slice(tile.source[0].start, stop), *tile.target[1:])
            for global_region, local_region in fusion.regions(target):
                region = (slice(None), slice(None), *local_region)
                weights = fusion.weight_sum[region]
                if not bool((weights > 0).all()):
                    raise RuntimeError("blend weights must cover the output slab.")
                clean = fusion.pred_sum[region] / weights
                if base is not None:
                    base.constrain(clean, global_region)
                if output is None:
                    previous = current.read(global_region).float()
                    noise = torch.randn(
                        previous.shape, device=generator.device, dtype=torch.float32
                    )
                    updated = generator.diffusion.sample_posterior(
                        previous,
                        clean,
                        transition,
                        noise=noise.to(previous.device),
                    )
                    if fusion.step_sum is not None:
                        updated = updated - fusion.step_sum[region] / weights
                    if base is not None:
                        base.constrain(
                            updated, global_region, generator.diffusion, transition
                        )
                    current.write(global_region, updated)
                else:
                    write_output(output, global_region, clean, margin)
                fusion.pred_sum[region].zero_()
                if fusion.step_sum is not None:
                    fusion.step_sum[region].zero_()
                weights.zero_()
