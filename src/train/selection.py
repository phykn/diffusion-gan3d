"""Export the step whose sections best match the real data, not the last one.

Only 2D sections of the real material exist, so a step is judged by how well
the sections of its volumes match fixed real crops: two-point correlation,
chord lengths, and how each phase splits into clusters. The last carries what
sections reveal of 3D connectivity. LR scores unanchored volumes; SR scores its
refinements of fixed LR bank volumes at the HR resolution.
"""

import shutil
from pathlib import Path

import numpy as np
import torch

from src.build.predict import make_generator
from src.config.files import save_yaml
from src.evaluate.structure import (
    SectionErrors,
    compare_sections,
    section_errors,
    section_statistics,
)
from src.predict.sr.inference import SuperResolutionAPI
from src.prepare.resize import downsample
from src.random import seeded_rng
from src.storage import torch_save
from src.train.state import export_sr

# Fixed reference sections per domain and plane, and the seed behind them and
# behind every evaluation volume, so each step meets the same data and noise.
REFERENCE_CROPS = 64
SEED = 0
# Generated sections per domain and plane, spread over the scored volumes. A
# fixed count keeps statistics cost growing with section area, not volume size,
# while still outnumbering the real crops.
SECTIONS = 2 * REFERENCE_CROPS


def build_reference(datasets: dict) -> dict:
    """Real crops per domain and plane, identical for every run on this data."""
    rng = np.random.RandomState(SEED)
    reference = {}
    for domain, planes in sorted(datasets.items()):
        reference[domain] = {}
        for axis, dataset in sorted(planes.items()):
            paths = sorted({path for group in dataset.path_groups for path in group})
            crops = [
                dataset.sample(paths[rng.randint(len(paths))], rng)
                for _ in range(REFERENCE_CROPS)
            ]
            reference[domain][axis] = {
                "images": torch.stack([crop["image"] for crop in crops]),
                "height_origins": [crop["height_origin"] for crop in crops],
                "height_extents": [crop["height_extent"] for crop in crops],
            }
    return reference


def weights_path(root: Path, step: int) -> Path:
    return root / "selection" / f"generator_{step:08d}.pt"


def inputs_path(root: Path) -> Path:
    return root / "selection" / "inputs.pt"


def pick_sections(volumes: list[torch.Tensor], axis: int, phases: int):
    """One-hot sections along axis, evenly spaced over all volumes' slices."""
    depth = volumes[0].shape[axis]
    total = len(volumes) * depth
    count = min(SECTIONS, total)
    indices = torch.linspace(0, total - 1, count).round().long().tolist()
    # Only the picked slices leave the uint8 volumes.
    labels = torch.stack(
        [volumes[index // depth].select(axis, index % depth) for index in indices]
    )
    return torch.nn.functional.one_hot(labels.long(), phases).movedim(-1, 1).float()


class CheckpointSelection:
    def __init__(self, reference: dict, samples: int, height_enabled: bool) -> None:
        self.reference = reference
        self.samples = samples
        self.height_enabled = height_enabled
        self.history: list[dict] = []
        self.real = {}
        self.noise = {}
        for domain, planes in reference.items():
            for axis, plane in planes.items():
                images = plane["images"].float()
                self.real[(domain, axis)] = section_statistics(images)
                # Errors between halves of the real crops: the sampling noise
                # that puts every error on one scale.
                self.noise[(domain, axis)] = [
                    max(error, 1e-6)
                    for error in section_errors(images[0::2], images[1::2])
                ]

    def heights(self, domain: int) -> list[dict]:
        """Height condition of each scored volume, cycling over side crops."""
        if not self.height_enabled:
            return [{}] * self.samples
        planes = self.reference[domain]
        # xy crops carry no depth.
        axes = [axis for axis in planes if axis != 0]
        heights = []
        for index in range(self.samples):
            plane = planes[axes[index % len(axes)]]
            crop = index // len(axes) % REFERENCE_CROPS
            heights.append(
                {
                    "height_origin": plane["height_origins"][crop],
                    "height_extent": plane["height_extents"][crop],
                }
            )
        return heights

    @torch.no_grad()
    def evaluate(self, trainer) -> dict[str, float]:
        generator = make_generator(
            trainer.ema_denoiser,
            trainer.cfg,
            trainer.device,
            trainer.settings.patch_size,
        )

        def volumes(domain):
            result = []
            for index, height in enumerate(self.heights(domain)):
                with seeded_rng(index, trainer.device):
                    result.append(generator.generate(domain=domain, **height))
            return result

        return self.score(volumes)

    def score(self, volumes) -> dict[str, float]:
        """Section errors of volumes(domain), uint8 labels, against the reference."""
        errors = {name: [] for name in SectionErrors._fields}
        structure = []
        for domain, planes in self.reference.items():
            labels = volumes(domain)
            for axis, plane in planes.items():
                fake = pick_sections(labels, axis, plane["images"].shape[1])
                values = compare_sections(
                    self.real[(domain, axis)], section_statistics(fake)
                )
                for name, value in values._asdict().items():
                    errors[name].append(value)
                noise = self.noise[(domain, axis)]
                structure.append(
                    np.mean([value / scale for value, scale in zip(values, noise)])
                )
        return {
            **{name: float(np.mean(values)) for name, values in errors.items()},
            "structure": float(np.mean(structure)),
        }

    def best(self) -> dict | None:
        if not self.history:
            return None
        return min(self.history, key=lambda entry: entry["structure"])

    def save_weights(self, trainer, path: Path) -> None:
        torch_save(trainer.ema_denoiser.state_dict(), path, overwrite=False)

    def update(self, trainer, root: Path) -> dict:
        """Score the current EMA weights, keeping them only while they are best."""
        previous = self.best()
        step = trainer.completed_steps
        entry = {"step": step, **self.evaluate(trainer)}
        self.history.append(entry)
        if self.best() is entry:
            self.save_weights(trainer, weights_path(root, step))
            if previous is not None:
                weights_path(root, previous["step"]).unlink(missing_ok=True)
        self.save_report(root)
        return entry

    def export(self, root: Path) -> bool:
        """Copy the best step's weights to generator.pt; False before any score."""
        best = self.best()
        if best is None:
            return False
        shutil.copyfile(weights_path(root, best["step"]), root / "generator.pt")
        return True

    def save_report(self, root: Path) -> None:
        save_yaml(
            root / "selection" / "scores.yaml",
            {"best_step": self.best()["step"], "steps": self.history},
        )

    def import_best(self, source: Path, root: Path) -> None:
        """Copy a resumed run's best weights into the new run directory."""
        best = self.best()
        if best is None:
            return
        path = weights_path(source, best["step"])
        if not path.is_file():
            raise FileNotFoundError(f"selection weights are missing: {path}")
        target = weights_path(root, best["step"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        self.save_report(root)

    def state_dict(self) -> dict:
        return {"history": [dict(entry) for entry in self.history]}

    def load_state_dict(self, state: dict) -> None:
        self.history = [dict(entry) for entry in state["history"]]


def lr_mse(high: torch.Tensor, low: torch.Tensor) -> float:
    """MSE between downsampled HR labels and the LR fractions they refine."""
    shape = tuple(low.shape[-3:])
    coarse = torch.cat(
        [downsample((high == phase)[None, None], shape) for phase in range(len(low))],
        1,
    )
    return float((coarse[0] - low).square().mean())


class SRCheckpointSelection(CheckpointSelection):
    """Score refinements of fixed LR inputs; the bank itself is refreshed."""

    def __init__(self, reference: dict, samples: int, inputs: dict) -> None:
        super().__init__(reference, samples, inputs["height_origins"] is not None)
        self.inputs = inputs

    @classmethod
    def from_bank(cls, reference, samples, bank, origins, extents):
        """Keep the first bank volumes; scoring cycles over them when fewer."""
        inputs = {
            "volumes": {d: v[:samples].clone() for d, v in bank.items()},
            "height_origins": None
            if origins is None
            else {d: o[:samples].clone() for d, o in origins.items()},
            "height_extents": None
            if origins is None
            else {d: torch.as_tensor(e)[:samples].clone() for d, e in extents.items()},
        }
        return cls(reference, samples, inputs)

    def heights(self, domain: int) -> list[dict]:
        if not self.height_enabled:
            return [{}] * self.samples
        count = len(self.inputs["volumes"][domain])
        origins = self.inputs["height_origins"][domain]
        extents = self.inputs["height_extents"][domain]
        return [
            {
                "height_origin": float(origins[index % count]),
                "height_extent": float(extents[index % count]),
            }
            for index in range(self.samples)
        ]

    @torch.no_grad()
    def evaluate(self, trainer) -> dict[str, float]:
        api = SuperResolutionAPI.from_model(
            trainer.ema_denoiser, trainer.cfg, trainer.device
        )
        errors = []

        def volumes(domain):
            low = self.inputs["volumes"][domain]
            result = []
            for index, height in enumerate(self.heights(domain)):
                source = low[index % len(low)]
                high = api.refine(source, domain=domain, seed=index, **height)
                errors.append(lr_mse(high, source))
                result.append(high)
            return result

        # LR agreement is recorded, not scored: SR learns to repair corrupted
        # LR input, so following it exactly is not the goal.
        return {**self.score(volumes), "lr_mse": float(np.mean(errors))}

    def save_weights(self, trainer, path: Path) -> None:
        export_sr(trainer, path, overwrite=False)

    def save_inputs(self, root: Path) -> None:
        torch_save(self.inputs, inputs_path(root), overwrite=False)

    def import_best(self, source: Path, root: Path) -> None:
        """Score the resumed run's LR inputs again and keep its best weights."""
        path = inputs_path(source)
        if not path.is_file():
            raise FileNotFoundError(f"selection inputs are missing: {path}")
        self.inputs = torch.load(path, map_location="cpu", weights_only=True)
        target = inputs_path(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        super().import_best(source, root)
