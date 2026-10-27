from dataclasses import dataclass, field, fields

import torch


@dataclass(frozen=True)
class RealBatch:
    images: dict[int, torch.Tensor]
    domains: dict[int, int] = field(default_factory=dict)
    origins: dict[int, torch.Tensor] = field(default_factory=dict)
    extents: dict[int, torch.Tensor] = field(default_factory=dict)
    heights: dict[int, torch.Tensor] = field(default_factory=dict)
    profiles: dict[int, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class SampledPairs:
    pairs: dict[int, tuple[torch.Tensor, torch.Tensor]]
    heights: dict[int, torch.Tensor] = field(default_factory=dict)

    def add(self, axis: int, pairs: tuple[torch.Tensor, ...]):
        """Store one axis's slice pairs and, when present, their height."""
        self.pairs[axis] = pairs[:2]
        if len(pairs) == 3:
            self.heights[axis] = pairs[2]

    def conditions(self, axis: int) -> dict[str, torch.Tensor]:
        """Critic height input for one axis."""
        return {"height": self.heights[axis]} if axis in self.heights else {}


@dataclass(frozen=True)
class DenoiserUpdate:
    adversarial: torch.Tensor
    total: torch.Tensor
    vf: torch.Tensor

    def detach(self) -> "DenoiserUpdate":
        return DenoiserUpdate(
            **{item.name: getattr(self, item.name).detach() for item in fields(self)}
        )


@dataclass(frozen=True)
class DenoiserBatch:
    transition: int
    critic_domains: dict[int, int]
    fake: SampledPairs
    real: SampledPairs
    clean_probs: torch.Tensor
    target_vf: torch.Tensor
    vf_present: torch.Tensor
    real_transition_ramp: float = 1.0
    profile: torch.Tensor | None = None
    coarse_target: torch.Tensor | None = None
    coarse_mask: torch.Tensor | None = None
    real_transition_loss: torch.Tensor | None = None
    observed_axes: tuple[int, ...] | None = None
    group_statistics_ramp: float = 1.0


@dataclass(frozen=True)
class StepPreparation:
    transition: int
    domain: int
    critic_domains: dict[int, int]
    real: RealBatch
    target_vf: torch.Tensor
    vf_present: torch.Tensor
    model_conditions: dict[str, torch.Tensor]
    real_transition_ramp: float = 1.0
    coarse_target: torch.Tensor | None = None
    coarse_mask: torch.Tensor | None = None
    group_statistics_ramp: float = 1.0
    diagnostics: dict[str, torch.Tensor] = field(default_factory=dict)
