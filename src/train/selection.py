import numpy as np
import torch

from src.config.io import save_yaml
from src.data.resize import downsample
from src.export import write_export
from src.rng import make_generator
from src.sampling.conditions import LRConditions, SRConditions, make_height_plan
from src.sampling.plan import choose_sr_margin
from src.sampling.sampler import Predictor, TileSampler
from src.stats.sections import (
    SectionErrors,
    compare_sections,
    half_split_errors,
    label_sections,
    section_statistics,
)

# Fixed crops and seeds, so every step meets the same data and noise.
REFERENCE_CROPS = 64


def build_reference(samplers):
    """Real crops per plane axis, identical for every run on this data."""
    generator = make_generator(0)
    reference = {}
    for axis, sampler in sorted(samplers.items()):
        crops = sampler.sample(REFERENCE_CROPS, generator)
        reference[axis] = {
            "images": crops.images.cpu(),
            "height_origins": crops.height_origins,
            "height_extents": crops.height_extents,
        }
    return reference


class CheckpointSelection:
    """Export the step whose sections best match fixed real crops, not the last.

    Only 2D sections of the material exist, so a step is scored on two-point
    correlation, chord lengths and per-phase cluster splits of its sections.
    """

    def __init__(self, reference, samples, height_enabled):
        self.reference = reference
        self.samples = samples
        self.height_enabled = height_enabled
        self.history = []
        # Best EMA weights travel in the checkpoint, so any resume restores them.
        self.best_state = None
        self.real = {}
        self.noise = {}
        for axis, plane in reference.items():
            images = label_sections(plane["images"])
            self.real[axis] = section_statistics(images)
            self.noise[axis] = half_split_errors(images)

    def heights(self):
        """Height condition of each scored volume, cycling over side crops."""
        if not self.height_enabled:
            return [{}] * self.samples
        axes = [axis for axis in self.reference if axis != 0]
        heights = []
        for index in range(self.samples):
            plane = self.reference[axes[index % len(axes)]]
            crop = index // len(axes) % REFERENCE_CROPS
            heights.append(
                {
                    "height_origin": float(plane["height_origins"][crop]),
                    "height_extent": float(plane["height_extents"][crop]),
                }
            )
        return heights

    @torch.no_grad()
    def evaluate(self, trainer):
        config = trainer.config
        patch = config.patch_size
        margin = trainer.ema.downsample_factor
        sampler = TileSampler(ema_predictor(trainer))
        volumes = []
        for index, height in enumerate(self.heights()):
            plan = height_plan(config, patch, margin, height)
            conditions = LRConditions(config, plan, trainer.device, **height)
            volumes.append(sampler.sample(plan, seed=index, conditions=conditions))
        return self.score(volumes)

    def score(self, volumes):
        """Section errors of uint8 label volumes against the reference."""
        errors = {name: [] for name in SectionErrors._fields}
        structure = []
        for axis, plane in self.reference.items():
            fake = pick_sections(volumes, axis, plane["images"].shape[1])
            values = compare_sections(self.real[axis], section_statistics(fake))
            for name, value in values._asdict().items():
                errors[name].append(value)
            structure.append(
                np.mean(
                    [
                        value / scale
                        for value, scale in zip(values, self.noise[axis], strict=True)
                    ]
                )
            )
        return {
            **{name: float(np.mean(values)) for name, values in errors.items()},
            "structure": float(np.mean(structure)),
        }

    def best(self):
        if not self.history:
            return None
        return min(self.history, key=lambda entry: entry["structure"])

    def update(self, trainer, root):
        """Score the current EMA weights, keeping them only while they are best."""
        previous = self.best()
        entry = {"step": trainer.completed_steps, **self.evaluate(trainer)}
        self.history.append(entry)
        if self.best() is entry:
            self.best_state = {
                name: value.detach().cpu().clone()
                for name, value in trainer.ema.state_dict().items()
            }
            self.save_best(trainer, root)
            if previous is not None:
                weights_path(root, previous["step"]).unlink(missing_ok=True)
        self.save_report(root)
        return entry

    def best_payload(self, trainer):
        return trainer.export_payload(self.best()["step"], self.best_state)

    def save_best(self, trainer, root):
        path = weights_path(root, self.best()["step"])
        write_export(path, self.best_payload(trainer), overwrite=False)

    def save_report(self, root):
        save_yaml(
            root / "selection" / "scores.yaml",
            {"best_step": self.best()["step"], "steps": self.history},
        )

    def restore(self, trainer, root):
        """Write a resumed run's best weights and scores into its new directory."""
        if self.best() is None:
            return
        self.save_best(trainer, root)
        self.save_report(root)

    def state_dict(self):
        return {
            "history": [dict(entry) for entry in self.history],
            "best_state": self.best_state,
        }

    def load_state_dict(self, state):
        self.history = [dict(entry) for entry in state["history"]]
        self.best_state = state["best_state"]


class SRCheckpointSelection(CheckpointSelection):
    """Score refinements of fixed LR inputs; the bank itself is refreshed."""

    def __init__(self, reference, samples, inputs):
        super().__init__(reference, samples, inputs["height_origins"] is not None)
        self.inputs = inputs

    @classmethod
    def from_bank(cls, reference, samples, bank):
        """Keep the first bank volumes; scoring cycles over them when fewer."""
        heights = bank.height_origins is not None
        inputs = {
            "volumes": bank.volumes[:samples].clone(),
            "height_origins": bank.height_origins[:samples].clone()
            if heights
            else None,
            "height_extents": bank.height_extents[:samples].clone()
            if heights
            else None,
        }
        return cls(reference, samples, inputs)

    def heights(self):
        if not self.height_enabled:
            return [{}] * self.samples
        count = len(self.inputs["volumes"])
        origins, extents = self.inputs["height_origins"], self.inputs["height_extents"]
        return [
            {
                "height_origin": float(origins[index % count]),
                "height_extent": float(extents[index % count]),
            }
            for index in range(self.samples)
        ]

    @torch.no_grad()
    def evaluate(self, trainer):
        config = trainer.config
        patch = config.patch_size
        margin = choose_sr_margin(config.scale_factor)
        sampler = TileSampler(ema_predictor(trainer))
        low = self.inputs["volumes"]
        volumes, errors = [], []
        for index, height in enumerate(self.heights()):
            plan = height_plan(config, patch, margin, height, config.scale_factor)
            source = low[index % len(low)]
            conditions = SRConditions(config, plan, source, trainer.device, **height)
            high = sampler.sample(plan, seed=index, conditions=conditions)
            errors.append(lr_mse(high, source))
            volumes.append(high)
        # Recorded, not scored: SR is meant to repair corrupted LR input.
        return {**self.score(volumes), "lr_mse": float(np.mean(errors))}

    def state_dict(self):
        return {**super().state_dict(), "inputs": self.inputs}

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.inputs = state["inputs"]


def height_plan(config, patch, margin, height, step=1):
    """The plan of one scored patch-size volume at its height."""
    return make_height_plan(
        config,
        (patch,) * 3,
        patch,
        margin,
        0,
        height.get("height_origin", 0.0),
        height.get("height_extent"),
        step,
    )


def ema_predictor(trainer):
    return Predictor(trainer.ema, trainer.diffusion, trainer.config, trainer.device)


def pick_sections(volumes, axis, phases):
    """One-hot sections along axis, evenly spaced over all volumes' slices."""
    depth = volumes[0].shape[axis]
    total = len(volumes) * depth
    # A fixed count bounds the cost and still outnumbers the real crops.
    count = min(2 * REFERENCE_CROPS, total)
    indices = torch.linspace(0, total - 1, count).round().long().tolist()
    labels = torch.stack(
        [volumes[index // depth].select(axis, index % depth) for index in indices]
    )
    return torch.nn.functional.one_hot(labels.long(), phases).movedim(-1, 1).float()


def lr_mse(high, low):
    """MSE between downsampled HR labels and the LR fractions they refine."""
    shape = tuple(low.shape[-3:])
    coarse = torch.cat(
        [downsample((high == phase)[None, None], shape) for phase in range(len(low))],
        1,
    )
    return float((coarse[0] - low).square().mean())


def weights_path(root, step):
    return root / "selection" / f"generator_{step:08d}.pt"
