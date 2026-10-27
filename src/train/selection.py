"""Export the LR step whose sections best match the real data, not the last one.

Only 2D sections of the real material exist, so a step is judged by how well
the sections of its unanchored volumes match fixed real crops: two-point
correlation, chord lengths, and how each phase splits into clusters. The last
carries what sections reveal of 3D connectivity.
"""

import shutil
from pathlib import Path

import numpy as np
import torch

from src.build.predict import make_generator
from src.config.files import save_yaml
from src.evaluate.structure import SectionErrors, section_errors
from src.random import seeded_rng
from src.storage import torch_save

# Fixed reference sections per domain and plane, and the seed behind them and
# behind every evaluation volume, so each step meets the same data and noise.
REFERENCE_CROPS = 64
SEED = 0


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


class CheckpointSelection:
    def __init__(self, reference: dict, samples: int, height_enabled: bool) -> None:
        self.reference = reference
        self.samples = samples
        self.height_enabled = height_enabled
        self.history: list[dict] = []
        # Errors between halves of the real crops: the sampling noise that puts
        # every error on one scale.
        self.noise = {
            (domain, axis): [
                max(error, 1e-6)
                for error in section_errors(
                    plane["images"][0::2].float(), plane["images"][1::2].float()
                )
            ]
            for domain, planes in reference.items()
            for axis, plane in planes.items()
        }

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
        errors = {name: [] for name in SectionErrors._fields}
        structure = []
        for domain, planes in self.reference.items():
            volumes = []
            for index, height in enumerate(self.heights(domain)):
                with seeded_rng(index, trainer.device):
                    volumes.append(generator.generate(domain=domain, **height))
            probs = (
                torch.nn.functional.one_hot(
                    torch.stack(volumes).long(), generator.num_phases
                )
                .movedim(-1, 1)
                .float()
            )
            for axis, plane in planes.items():
                fake = probs.movedim(axis + 2, 1).flatten(0, 1)
                values = section_errors(plane["images"].float(), fake)
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

    def update(self, trainer, root: Path) -> dict:
        """Score the current EMA weights, keeping them only while they are best."""
        previous = self.best()
        step = trainer.completed_steps
        entry = {"step": step, **self.evaluate(trainer)}
        self.history.append(entry)
        if self.best() is entry:
            torch_save(
                trainer.ema_denoiser.state_dict(),
                weights_path(root, step),
                overwrite=False,
            )
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
