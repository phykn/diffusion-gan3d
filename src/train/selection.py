"""Export the LR step that best covers the real data, not the last one.

Coverage asks whether the generator can still reach real sections. Each fixed
real crop anchors one volume through posterior sampling, and the score is the
fraction of the anchor plane that volume reproduces. A generator that drops
modes of the data fails on the crops it no longer covers.

Weak early weights bend to any anchor, so coverage alone would favor them.
Section statistics of unanchored volumes guard it: a step is a candidate only
while its structure error stays within tolerance of the best step so far.
"""

import shutil
from pathlib import Path

import numpy as np
import torch

from src.build.predict import make_generator
from src.config.files import save_yaml
from src.evaluate.structure import section_errors
from src.predict.anchor import PlaneAnchor
from src.random import seeded_rng
from src.storage import torch_save

# Fixed reference sections per domain and plane, and the seed behind them and
# behind every evaluation volume, so each step meets the same data and noise.
REFERENCE_CROPS = 64
SEED = 0


def build_reference(datasets: dict, samples: int) -> dict:
    """Real crops per domain and plane, identical for every run on this data."""
    rng = np.random.RandomState(SEED)
    count = max(REFERENCE_CROPS, samples)
    reference = {}
    for domain, planes in sorted(datasets.items()):
        reference[domain] = {}
        for axis, dataset in sorted(planes.items()):
            paths = sorted({path for group in dataset.path_groups for path in group})
            crops = [
                dataset.sample(paths[rng.randint(len(paths))], rng)
                for _ in range(count)
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
    def __init__(
        self, reference: dict, samples: int, tolerance: float, height_enabled: bool
    ) -> None:
        self.reference = reference
        self.samples = samples
        self.tolerance = tolerance
        self.height_enabled = height_enabled
        self.history: list[dict] = []
        # Two-point and chord errors between halves of the real crops: the
        # sampling noise that puts both errors on one scale.
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

    def anchors(self, domain: int) -> list[tuple[int, int]]:
        """(axis, crop index) of each anchored volume, cycling over the planes."""
        planes = self.reference[domain]
        # A height-conditioned volume takes its depth from a side crop; xy
        # crops carry none.
        axes = [axis for axis in planes if axis != 0 or not self.height_enabled]
        return [
            (axes[index % len(axes)], index // len(axes))
            for index in range(self.samples)
        ]

    @torch.no_grad()
    def evaluate(self, trainer) -> dict[str, float]:
        generator = make_generator(
            trainer.ema_denoiser,
            trainer.cfg,
            trainer.device,
            trainer.settings.patch_size,
        )
        middle = generator.patch_size // 2
        matches, two_point, chord, structure = [], [], [], []
        for domain, planes in self.reference.items():
            volumes = []
            for index, (axis, crop) in enumerate(self.anchors(domain)):
                plane = planes[axis]
                image = plane["images"][crop].float().clamp(0.0, 1.0)
                height = {}
                if self.height_enabled:
                    height = {
                        "height_origin": plane["height_origins"][crop],
                        "height_extent": plane["height_extents"][crop],
                    }
                with seeded_rng(index, trainer.device):
                    anchored = generator.generate(
                        anchors=(PlaneAnchor(image, axis, middle),),
                        anchor_strength=1.0,
                        domain=domain,
                        **height,
                    )
                section = anchored.select(axis, middle).long()
                matches.append(float((section == image.argmax(0)).float().mean()))
                with seeded_rng(self.samples + index, trainer.device):
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
                errors = section_errors(plane["images"].float(), fake)
                two_point.append(errors[0])
                chord.append(errors[1])
                noise = self.noise[(domain, axis)]
                structure.append(
                    np.mean([error / scale for error, scale in zip(errors, noise)])
                )
        return {
            "coverage": float(np.mean(matches)),
            "two_point_mae": float(np.mean(two_point)),
            "chord_tv": float(np.mean(chord)),
            "structure": float(np.mean(structure)),
        }

    def candidates(self) -> list[dict]:
        """Steps whose structure error is within tolerance of the lowest so far.

        The lowest error only falls, so a step that drops out never returns.
        """
        if not self.history:
            return []
        limit = min(entry["structure"] for entry in self.history)
        limit *= 1.0 + self.tolerance
        return [entry for entry in self.history if entry["structure"] <= limit]

    def best(self) -> dict | None:
        candidates = self.candidates()
        if not candidates:
            return None
        return max(candidates, key=lambda entry: entry["coverage"])

    def update(self, trainer, root: Path) -> dict:
        """Score the current EMA weights, keeping them while they are a candidate."""
        step = trainer.completed_steps
        entry = {"step": step, **self.evaluate(trainer)}
        self.history.append(entry)
        if step in {candidate["step"] for candidate in self.candidates()}:
            torch_save(
                trainer.ema_denoiser.state_dict(),
                weights_path(root, step),
                overwrite=False,
            )
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
        best = self.best()
        candidates = {entry["step"] for entry in self.candidates()}
        save_yaml(
            root / "selection" / "scores.yaml",
            {
                "best_step": best["step"],
                "structure_tolerance": self.tolerance,
                "steps": [
                    {**entry, "candidate": entry["step"] in candidates}
                    for entry in self.history
                ],
            },
        )

    def import_candidates(self, source: Path, root: Path) -> None:
        """Copy a resumed run's candidate weights into the new run directory."""
        for entry in self.candidates():
            path = weights_path(source, entry["step"])
            if not path.is_file():
                raise FileNotFoundError(f"selection weights are missing: {path}")
            target = weights_path(root, entry["step"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        if self.history:
            self.save_report(root)

    def state_dict(self) -> dict:
        return {"history": [dict(entry) for entry in self.history]}

    def load_state_dict(self, state: dict) -> None:
        self.history = [dict(entry) for entry in state["history"]]
