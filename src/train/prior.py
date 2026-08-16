from collections import deque
from dataclasses import dataclass

import torch

from .. import AXES
from ..anchor import AnchorCondition, PlaneAnchor, build_anchors


@dataclass(frozen=True)
class PriorCondition:
    condition: AnchorCondition
    observed_mask: torch.Tensor
    observed_axis_masks: torch.Tensor


@dataclass(frozen=True)
class _Entry:
    labels: torch.Tensor
    observed: PlaneAnchor


class _Bank:
    def __init__(self, volume_count: int) -> None:
        if (
            not isinstance(volume_count, int)
            or isinstance(volume_count, bool)
            or volume_count < 1
        ):
            raise ValueError("prior volume count must be a positive integer.")
        self.volume_count = volume_count
        self.storage_bytes = 0
        self._volume_bytes: int | None = None
        self._items: deque[_Entry] = deque()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def ready(self) -> bool:
        return len(self._items) >= self.volume_count

    @property
    def items(self) -> tuple[_Entry, ...]:
        return tuple(self._items)

    def add(self, entries: tuple[_Entry, ...]) -> None:
        for entry in entries:
            if self.ready:
                return
            self._validate(entry)
            stored = _clone_entry(entry)
            self._items.append(stored)
            self.storage_bytes += _entry_bytes(stored)

    def replace_oldest(self, entry: _Entry) -> None:
        if not self.ready:
            raise RuntimeError("the initial prior must be complete before refresh.")
        self._validate(entry)
        removed = self._items.popleft()
        self.storage_bytes -= _entry_bytes(removed)
        stored = _clone_entry(entry)
        self._items.append(stored)
        self.storage_bytes += _entry_bytes(stored)

    def _validate(self, entry: _Entry) -> None:
        volume_bytes = entry.labels.numel() * entry.labels.element_size()
        if self._volume_bytes is None:
            self._volume_bytes = volume_bytes
        elif self._volume_bytes != volume_bytes:
            raise ValueError("prior volume shapes must remain fixed.")


class ConditionalPrior:
    """Rolling EMA completions rooted in observed real anchor planes."""

    def __init__(
        self,
        *,
        num_phases: int,
        num_domains: int,
        patch_size: int,
        volume_count: int,
        plane_stride: int,
        owned_axes: dict[int, tuple[int, ...]],
    ) -> None:
        for name, value in (
            ("num_phases", num_phases),
            ("num_domains", num_domains),
            ("patch_size", patch_size),
            ("plane_stride", plane_stride),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if num_phases > 256:
            raise ValueError("uint8 prior storage supports at most 256 phases.")
        if patch_size % plane_stride:
            raise ValueError("patch size must be divisible by the plane stride.")
        if set(owned_axes) != set(range(num_domains)):
            raise ValueError("owned axes must cover every domain.")
        for axes in owned_axes.values():
            if not axes or any(axis not in AXES for axis in axes):
                raise ValueError("every domain must own valid axes.")

        self.num_phases = num_phases
        self.patch_size = patch_size
        self.plane_stride = plane_stride
        self.owned_axes = {domain: tuple(axes) for domain, axes in owned_axes.items()}
        self._banks = tuple(_Bank(volume_count) for _ in range(num_domains))
        self.updates = 0

    @property
    def count(self) -> int:
        return sum(map(len, self._banks))

    @property
    def storage_bytes(self) -> int:
        return sum(bank.storage_bytes for bank in self._banks)

    @property
    def ready(self) -> bool:
        return all(bank.ready for bank in self._banks)

    def needs(self, domain: int) -> bool:
        self._check_domain(domain)
        return not self._banks[domain].ready

    def record(
        self,
        prediction: torch.Tensor,
        observed: AnchorCondition,
        domain: int,
    ) -> None:
        self._check_domain(domain)
        entries = self._entries(prediction, observed)
        self._banks[domain].add(entries)

    def refresh(
        self,
        prediction: torch.Tensor,
        observed: AnchorCondition,
        domain: int,
    ) -> None:
        self._check_domain(domain)
        entries = self._entries(prediction, observed)
        if len(entries) != 1:
            raise ValueError("a rolling refresh must contain exactly one volume.")
        self._banks[domain].replace_oldest(entries[0])
        self.updates += 1

    def volumes(self, domain: int, axis: int) -> tuple[torch.Tensor, ...]:
        self._check_domain(domain)
        self._check_axis(axis)
        if axis in self.owned_axes[domain]:
            if not self._banks[domain].ready:
                return ()
            return tuple(entry.labels for entry in self._banks[domain].items)
        return tuple(
            entry.labels
            for source, bank in enumerate(self._banks)
            if axis in self.owned_axes[source] and bank.ready
            for entry in bank.items
        )

    def sample_condition(
        self,
        domain: int,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> PriorCondition:
        self._check_domain(domain)
        bank = self._banks[domain]
        if not bank.ready:
            raise RuntimeError("the domain prior must be ready before sampling.")
        entries = _sample_entries(bank.items, batch_size, generator)
        max_planes = max(1, self.patch_size // self.plane_stride)
        count = _log_uniform_count(max_planes, generator)
        conditions = []
        observed_conditions = []
        for entry in entries:
            condition, observed = self._sample_entry_condition(
                entry,
                count=count,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            conditions.append(condition)
            observed_conditions.append(observed)
        return PriorCondition(
            _concat_conditions(tuple(conditions)),
            torch.cat(tuple(value.mask for value in observed_conditions)),
            torch.cat(tuple(value.axis_masks for value in observed_conditions)),
        )

    def _sample_entry_condition(
        self,
        entry: _Entry,
        *,
        count: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None,
    ) -> tuple[AnchorCondition, AnchorCondition]:
        labels = entry.labels.clone()
        _overlay_observed(labels, entry.observed)
        planes = [entry.observed]
        candidates = self._plane_candidates(entry.observed, generator)
        if count > 1 and candidates:
            order = torch.randperm(len(candidates), generator=generator)
            for choice in order[: count - 1].tolist():
                axis, index = candidates[choice]
                planes.append(
                    PlaneAnchor(
                        image=labels.select(axis, index),
                        axis=axis,
                        index=index,
                    )
                )

        condition = self._build_condition(
            tuple(planes),
            device=device,
            dtype=dtype,
            reconcile=False,
        )
        observed = self._build_condition(
            (entry.observed,),
            device=device,
            dtype=dtype,
        )
        return condition, observed

    def _build_condition(
        self,
        planes: tuple[PlaneAnchor, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
        reconcile: bool = True,
    ) -> AnchorCondition:
        condition = build_anchors(
            planes,
            batch_size=1,
            num_phases=self.num_phases,
            volume_size=self.patch_size,
            device=device,
            dtype=dtype,
            reconcile=reconcile,
        )
        if condition is None:
            raise RuntimeError("prior anchor construction returned no condition.")
        return condition

    def _entries(
        self,
        prediction: torch.Tensor,
        observed: AnchorCondition,
    ) -> tuple[_Entry, ...]:
        if prediction.ndim != 5 or prediction.shape[1] != self.num_phases:
            raise ValueError("prior prediction must have shape [B, C, D, H, W].")
        if tuple(prediction.shape[2:]) != (
            self.patch_size,
            self.patch_size,
            self.patch_size,
        ):
            raise ValueError("prior prediction must use the configured patch size.")
        if observed.planes != 1:
            raise ValueError("a conditional prior requires exactly one observed plane.")
        if observed.target.shape[0] < prediction.shape[0]:
            raise ValueError("the observed anchor batch must cover the prediction.")
        labels = prediction.detach().argmax(dim=1).to(device="cpu", dtype=torch.uint8)
        entries = []
        for index in range(prediction.shape[0]):
            plane = _extract_observed(observed, index)
            values = labels[index].contiguous()
            entries.append(_Entry(values, plane))
        return tuple(entries)

    def _plane_candidates(
        self,
        observed: PlaneAnchor,
        generator: torch.Generator | None,
    ) -> list[tuple[int, int]]:
        candidates = []
        for axis in AXES:
            offset = _random_index(self.plane_stride, generator)
            for index in range(offset, self.patch_size, self.plane_stride):
                if (
                    axis == observed.axis
                    and abs(index - observed.index) < self.plane_stride
                ):
                    continue
                candidates.append((axis, index))
        return candidates

    def _check_domain(self, domain: int) -> None:
        if not isinstance(domain, int) or isinstance(domain, bool):
            raise TypeError("domain must be an integer.")
        if not 0 <= domain < len(self._banks):
            raise ValueError("domain is outside the prior bank.")

    @staticmethod
    def _check_axis(axis: int) -> None:
        if not isinstance(axis, int) or isinstance(axis, bool) or axis not in AXES:
            raise ValueError("axis must be 0, 1, or 2.")


def _extract_observed(condition: AnchorCondition, batch: int) -> PlaneAnchor:
    found = []
    for axis in AXES:
        mask = condition.axis_masks[batch, axis].movedim(axis, 0)
        indices = mask.flatten(1).any(dim=1).nonzero().flatten()
        found.extend((axis, int(index)) for index in indices)
    if len(found) != 1:
        raise ValueError("a conditional prior seed must contain one plane.")
    axis, index = found[0]
    mask = condition.mask[batch, 0].select(axis, index)
    points = mask.nonzero()
    if not points.numel():
        raise ValueError("the observed prior plane must not be empty.")
    top = int(points[:, 0].min())
    bottom = int(points[:, 0].max()) + 1
    left = int(points[:, 1].min())
    right = int(points[:, 1].max()) + 1
    image = condition.target[batch].select(axis, index)[top:bottom, left:right]
    return PlaneAnchor(
        image=image.detach().to(device="cpu", dtype=torch.uint8).contiguous(),
        axis=axis,
        index=index,
        position=(top, left),
    )


def _clone_entry(entry: _Entry) -> _Entry:
    observed = entry.observed
    return _Entry(
        entry.labels.detach().to(device="cpu", dtype=torch.uint8).contiguous().clone(),
        PlaneAnchor(
            image=observed.image.detach()
            .to(device="cpu", dtype=torch.uint8)
            .contiguous()
            .clone(),
            axis=observed.axis,
            index=observed.index,
            position=observed.position,
        ),
    )


def _overlay_observed(labels: torch.Tensor, observed: PlaneAnchor) -> None:
    if observed.position is None:
        raise ValueError("stored observed anchors must include a position.")
    row, col = observed.position
    height, width = observed.image.shape
    labels.select(observed.axis, observed.index)[
        row : row + height,
        col : col + width,
    ].copy_(observed.image)


def _concat_conditions(conditions: tuple[AnchorCondition, ...]) -> AnchorCondition:
    if not conditions:
        raise ValueError("prior conditions must not be empty.")
    plane_counts = {condition.planes for condition in conditions}
    if len(plane_counts) != 1:
        raise ValueError("prior batch items must use the same plane count.")
    return AnchorCondition(
        image=torch.cat(tuple(condition.image for condition in conditions)),
        mask=torch.cat(tuple(condition.mask for condition in conditions)),
        axis_masks=torch.cat(tuple(condition.axis_masks for condition in conditions)),
        target=torch.cat(tuple(condition.target for condition in conditions)),
        planes=conditions[0].planes,
        conflicts=sum(condition.conflicts for condition in conditions),
        source_voxels=sum(condition.source_voxels for condition in conditions),
    )


def _entry_bytes(entry: _Entry) -> int:
    return (
        entry.labels.numel() * entry.labels.element_size()
        + entry.observed.image.numel() * entry.observed.image.element_size()
    )


def _log_uniform_count(
    maximum: int,
    generator: torch.Generator | None,
) -> int:
    if maximum == 1:
        return 1
    weights = torch.arange(1, maximum + 1, dtype=torch.float32).reciprocal()
    return int(torch.multinomial(weights, 1, generator=generator).item()) + 1


def _sample_entries(
    entries: tuple[_Entry, ...],
    batch_size: int,
    generator: torch.Generator | None,
) -> tuple[_Entry, ...]:
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer.")
    if batch_size <= len(entries):
        indices = torch.randperm(len(entries), generator=generator)[:batch_size]
    else:
        indices = torch.randint(
            len(entries),
            (batch_size,),
            generator=generator,
        )
    return tuple(entries[index] for index in indices.tolist())


def _random_index(size: int, generator: torch.Generator | None) -> int:
    return int(torch.randint(size, (), generator=generator).item())
