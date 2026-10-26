from dataclasses import dataclass, field, fields
from typing import Literal

import torch

from src.anchor import AnchorCondition
from src.data.slice import BandBatch, TripletBatch


@dataclass(frozen=True)
class RealBatch:
    images: dict[int, torch.Tensor]
    domains: dict[int, int] = field(default_factory=dict)
    origins: dict[int, torch.Tensor] = field(default_factory=dict)
    extents: dict[int, torch.Tensor] = field(default_factory=dict)
    heights: dict[int, torch.Tensor] = field(default_factory=dict)
    profiles: dict[int, torch.Tensor] = field(default_factory=dict)
    geometry: dict[int, list[dict]] = field(default_factory=dict)


@dataclass(frozen=True)
class SampledPairs:
    pairs: dict[int, tuple[torch.Tensor, torch.Tensor]]
    heights: dict[int, torch.Tensor] = field(default_factory=dict)
    profiles: dict[int, torch.Tensor] = field(default_factory=dict)

    def add(self, axis: int, pairs: tuple[torch.Tensor, ...], has_profile: bool):
        """Store one axis's slice pairs; condition channel 0 is height, then profile."""
        self.pairs[axis] = pairs[:2]
        if len(pairs) == 3:
            self.heights[axis] = pairs[2][:, :1]
            if has_profile:
                self.profiles[axis] = pairs[2][:, 1:]

    def conditions(self, axis: int) -> dict[str, torch.Tensor]:
        """Critic height and profile inputs for one axis."""
        values = {}
        if axis in self.heights:
            values["height"] = self.heights[axis]
        if axis in self.profiles:
            values["profile"] = self.profiles[axis]
        return values


@dataclass(frozen=True)
class DenoiserUpdate:
    adversarial: torch.Tensor
    total: torch.Tensor
    connectivity: torch.Tensor
    connectivity_global: torch.Tensor
    connectivity_local: torch.Tensor
    anchor: torch.Tensor
    anchor_coarse: torch.Tensor
    anchor_pixel: torch.Tensor
    anchor_accuracy: torch.Tensor
    vf: torch.Tensor

    def detach(self) -> "DenoiserUpdate":
        return DenoiserUpdate(
            **{item.name: getattr(self, item.name).detach() for item in fields(self)}
        )


@dataclass(frozen=True)
class DenoiserBatch:
    transition: int
    connectivity_domains: torch.Tensor
    critic_domains: dict[int, int]
    fake: SampledPairs
    real: SampledPairs
    connectivity_fake: TripletBatch
    logits: torch.Tensor
    clean_probs: torch.Tensor
    anchor: AnchorCondition | None
    anchor_measured: AnchorCondition | None
    anchor_present: torch.Tensor
    anchor_ramp: float
    target_vf: torch.Tensor
    vf_present: torch.Tensor
    connectivity_ramp: float = 1.0
    profile: torch.Tensor | None = None
    coarse_target: torch.Tensor | None = None
    coarse_mask: torch.Tensor | None = None
    real_transition_loss: torch.Tensor | None = None
    anchor_continuity_loss: torch.Tensor | None = None
    anchor_neighborhood_loss: torch.Tensor | None = None
    seam_fake: BandBatch | None = None
    seam_domains: torch.Tensor | None = None
    observed_axes: tuple[int, ...] | None = None
    group_statistics_ramp: float = 1.0


@dataclass(frozen=True)
class StepPreparation:
    transition: int
    domain: int
    critic_domains: dict[int, int]
    real: RealBatch
    selection: "AnchorSelection | None"
    target_vf: torch.Tensor
    presence: "ConditionPresence"
    model_conditions: dict[str, torch.Tensor]
    anchor_ramp: float
    connectivity_ramp: float = 1.0
    coarse_target: torch.Tensor | None = None
    coarse_mask: torch.Tensor | None = None
    group_statistics_ramp: float = 1.0
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class AnchorSelection:
    condition: AnchorCondition
    source: Literal["real", "shared", "multi"]
    measured: AnchorCondition
    reference: torch.Tensor | None = None
    height: torch.Tensor | None = None
    profile: torch.Tensor | None = None
    geometry: list[dict] | None = None


@dataclass(frozen=True)
class ConditionPresence:
    anchor: torch.Tensor
    vf: torch.Tensor
