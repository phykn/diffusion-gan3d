from dataclasses import dataclass, field
from typing import Literal

import torch

from src.anchor import AnchorCondition
from src.data.slice import TripletBatch


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


@dataclass(frozen=True)
class DenoiserUpdate:
    adversarial: torch.Tensor
    total: torch.Tensor
    global_loss: torch.Tensor
    local_loss: torch.Tensor
    connectivity: torch.Tensor
    normal_transition: torch.Tensor
    anchor: torch.Tensor
    anchor_coarse: torch.Tensor
    anchor_pixel: torch.Tensor
    anchor_accuracy: torch.Tensor
    vf: torch.Tensor


@dataclass(frozen=True)
class DenoiserBatch:
    transition: int
    connectivity_domains: torch.Tensor
    critic_domains: dict[int, int]
    fake: dict[int, tuple[torch.Tensor, torch.Tensor]]
    connectivity_real: TripletBatch
    connectivity_fake: TripletBatch
    logits: torch.Tensor
    clean_probs: torch.Tensor
    anchor: AnchorCondition | None
    anchor_observed_mask: torch.Tensor | None
    anchor_observed_axis_masks: torch.Tensor | None
    anchor_present: torch.Tensor
    anchor_ramp: float
    target_vf: torch.Tensor
    vf_present: torch.Tensor
    connectivity_ramp: float = 1.0
    fake_heights: dict[int, torch.Tensor] = field(default_factory=dict)
    fake_profiles: dict[int, torch.Tensor] = field(default_factory=dict)
    profile: torch.Tensor | None = None
    coarse_target: torch.Tensor | None = None
    real_transition_loss: torch.Tensor | None = None


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


@dataclass(frozen=True)
class AnchorSelection:
    condition: AnchorCondition | None
    observed_mask: torch.Tensor | None
    observed_axis_masks: torch.Tensor | None
    source: Literal["real", "shared", "multi"]
    measured: AnchorCondition | None = None
    reference: torch.Tensor | None = None
    height: torch.Tensor | None = None
    profile: torch.Tensor | None = None
    geometry: list[dict] | None = None


@dataclass(frozen=True)
class ConditionPresence:
    anchor: torch.Tensor
    vf: torch.Tensor
