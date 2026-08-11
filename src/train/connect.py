import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .. import AXES
from ..anchor import AnchorCondition, PlaneAnchor, build_anchors

TEACHER_MIN_ENTRIES = 4


@dataclass(frozen=True)
class TripletBatch:
    values: torch.Tensor
    axes: torch.Tensor
    center_slots: torch.Tensor

    def __post_init__(self) -> None:
        if self.values.ndim != 5 or self.values.shape[1] != 3:
            raise ValueError("triplets must have shape [B, 3, C, H, W].")
        if self.axes.shape != (self.values.shape[0],):
            raise ValueError("triplet axes must have shape [B].")
        if self.axes.dtype != torch.long or self.axes.device != self.values.device:
            raise ValueError("triplet axes must use torch.long on the values device.")
        if self.center_slots.shape != (self.values.shape[0],):
            raise ValueError("triplet center slots must have shape [B].")
        if self.center_slots.dtype != torch.long:
            raise ValueError("triplet center slots must use torch.long dtype.")
        if self.center_slots.device != self.values.device:
            raise ValueError("triplet center slots must be on the values device.")
        if self.center_slots.numel() and (
            int(self.center_slots.min()) < 0 or int(self.center_slots.max()) > 2
        ):
            raise ValueError("triplet center slots must be zero, one, or two.")

    def __len__(self) -> int:
        return self.values.shape[0]

    def index_select(self, indices: torch.Tensor) -> "TripletBatch":
        return TripletBatch(
            values=self.values.index_select(0, indices),
            axes=self.axes.index_select(0, indices),
            center_slots=self.center_slots.index_select(0, indices),
        )


def normal_transition_loss(real: TripletBatch, fake: TripletBatch) -> torch.Tensor:
    """Compare center-to-neighbor phase transitions along the triplet normal."""
    if real.values.shape != fake.values.shape:
        raise ValueError("real and fake triplets must have the same shape.")
    if not torch.equal(real.axes, fake.axes):
        raise ValueError("real and fake triplets must use the same axes.")
    _, _, phase_count, height, width = real.values.shape
    if phase_count == 0 or height == 0 or width == 0:
        raise ValueError("triplets must contain phases and spatial values.")
    if len(fake) == 0:
        return fake.values.sum() * 0.0

    batch_indices = torch.arange(len(real), device=real.values.device)

    def transition_matrices(
        values: torch.Tensor,
        center_slots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        neighbor_slots = torch.stack(
            (
                (center_slots - 1).clamp_min(0),
                (center_slots + 1).clamp_max(2),
            ),
            dim=1,
        )
        valid_neighbors = torch.stack(
            (center_slots > 0, center_slots < 2),
            dim=1,
        )
        probabilities = (values.to(torch.float32) + 1.0) * 0.5
        center = probabilities[batch_indices, center_slots]
        neighbors = probabilities[batch_indices[:, None], neighbor_slots]
        transitions = torch.einsum(
            "bchw,bnkhw->bnck",
            center,
            neighbors,
        ) / (height * width)
        return transitions, valid_neighbors

    real_transitions, real_valid = transition_matrices(
        real.values,
        real.center_slots,
    )
    fake_transitions, fake_valid = transition_matrices(
        fake.values,
        fake.center_slots,
    )
    valid_neighbors = real_valid & fake_valid
    total_variation = 0.5 * (real_transitions - fake_transitions).abs().sum(
        dim=(-2, -1)
    )
    return (total_variation * valid_neighbors).sum() / valid_neighbors.sum().clamp_min(
        1
    )


@dataclass(frozen=True)
class TeacherAnchor:
    condition: AnchorCondition
    target_vf: torch.Tensor


@dataclass(frozen=True)
class _ReplayEntry:
    values: torch.Tensor
    phase_fraction: torch.Tensor
    center_slot: int


class _TripletReplay:
    def __init__(self, capacity_per_axis: int) -> None:
        if (
            not isinstance(capacity_per_axis, int)
            or isinstance(capacity_per_axis, bool)
            or capacity_per_axis < 1
        ):
            raise ValueError("replay capacity must be a positive integer.")
        self._items = {axis: deque(maxlen=capacity_per_axis) for axis in AXES}

    def __len__(self) -> int:
        return sum(len(values) for values in self._items.values())

    def add(self, batch: TripletBatch) -> None:
        for values, axis_value, center_slot in zip(
            batch.values,
            batch.axes.tolist(),
            batch.center_slots.tolist(),
            strict=True,
        ):
            self._check_axis(axis_value)
            stored = (
                values.detach()
                .to(
                    device="cpu",
                    dtype=torch.float16,
                )
                .contiguous()
                .clone()
            )
            fraction = ((stored.to(torch.float32) + 1.0) * 0.5).mean(dim=(0, 2, 3))
            self._items[axis_value].append(_ReplayEntry(stored, fraction, center_slot))

    def sample_matched(
        self,
        target: TripletBatch,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TripletBatch, torch.Tensor]:
        selected = []
        selected_centers = []
        indices = []
        target_fractions = (
            ((target.values.detach().to(torch.float32) + 1.0) * 0.5)
            .mean(dim=(1, 3, 4))
            .to(device="cpu")
        )
        for index, (axis_value, fraction) in enumerate(
            zip(target.axes.tolist(), target_fractions, strict=True)
        ):
            self._check_axis(axis_value)
            candidates = self._items[axis_value]
            if not candidates:
                continue
            distances = torch.stack(
                [(entry.phase_fraction - fraction).abs().mean() for entry in candidates]
            )
            nearest = distances.argsort()[: min(4, len(candidates))]
            choice = int(
                torch.randint(
                    len(nearest),
                    (),
                    generator=generator,
                ).item()
            )
            entry = candidates[int(nearest[choice])]
            selected.append(entry.values.clone())
            selected_centers.append(entry.center_slot)
            indices.append(index)

        if not selected:
            empty_indices = torch.empty(
                0,
                device=target.values.device,
                dtype=torch.long,
            )
            return (
                target.index_select(empty_indices),
                empty_indices,
            )
        matched = torch.tensor(
            indices,
            device=target.values.device,
            dtype=torch.long,
        )
        metadata = target.index_select(matched)
        return (
            TripletBatch(
                values=torch.stack(selected).to(
                    device=target.values.device,
                    dtype=target.values.dtype,
                ),
                axes=metadata.axes,
                center_slots=torch.tensor(
                    selected_centers,
                    device=target.values.device,
                    dtype=torch.long,
                ),
            ),
            matched,
        )

    @staticmethod
    def _check_axis(axis: int) -> None:
        if not isinstance(axis, int) or isinstance(axis, bool) or axis not in AXES:
            raise ValueError("axis must be 0, 1, or 2.")


@dataclass(frozen=True)
class _TeacherEntry:
    labels: torch.Tensor
    seed: PlaneAnchor
    target_vf: torch.Tensor

    @property
    def volume_size(self) -> int:
        return self.labels.shape[0]

    @property
    def storage_bytes(self) -> int:
        seed = self.seed.image
        return (
            self.labels.numel() * self.labels.element_size()
            + seed.numel() * seed.element_size()
            + self.target_vf.numel() * self.target_vf.element_size()
        )


class _TeacherBank:
    def __init__(self, max_bytes: int) -> None:
        if (
            not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool)
            or max_bytes < 1
        ):
            raise ValueError("teacher bank byte budget must be a positive integer.")
        self.max_bytes = max_bytes
        self.storage_bytes = 0
        self._items: deque[_TeacherEntry] = deque()

    def __len__(self) -> int:
        return len(self._items)

    def count_for(self, volume_size: int) -> int:
        return sum(entry.volume_size == volume_size for entry in self._items)

    def add(
        self,
        labels: torch.Tensor,
        seeds: Sequence[PlaneAnchor],
        num_phases: int,
    ) -> None:
        if labels.ndim != 4 or len(set(labels.shape[1:])) != 1:
            raise ValueError("teacher labels must have shape [B, S, S, S].")
        if len(seeds) != labels.shape[0]:
            raise ValueError("one real seed plane is required per teacher volume.")
        for values, seed in zip(labels, seeds, strict=True):
            stored = (
                values.detach()
                .to(
                    device="cpu",
                    dtype=torch.uint8,
                )
                .contiguous()
                .clone()
            )
            if int(stored.max()) >= num_phases:
                raise ValueError("teacher labels contain a phase outside num_phases.")
            image = seed.image
            if image.ndim != 2:
                raise ValueError("stored seed images must have shape [H, W].")
            height, width = image.shape
            if seed.position is None:
                row = (stored.shape[1] - height) // 2
                col = (stored.shape[2] - width) // 2
            else:
                row, col = seed.position
            predicted_seed = stored.select(seed.axis, seed.index)[
                row : row + height,
                col : col + width,
            ]
            stored_seed = PlaneAnchor(
                image=predicted_seed.contiguous().clone(),
                axis=seed.axis,
                index=seed.index,
                position=seed.position,
            )
            counts = torch.bincount(
                stored.to(torch.long).flatten(),
                minlength=num_phases,
            ).to(torch.float32)
            target_vf = counts.div(counts.sum())
            entry = _TeacherEntry(stored, stored_seed, target_vf)
            if entry.storage_bytes > self.max_bytes:
                raise ValueError("one teacher volume exceeds the bank byte budget.")
            while self._items and (
                self.storage_bytes + entry.storage_bytes > self.max_bytes
            ):
                self.storage_bytes -= self._items.popleft().storage_bytes
            self._items.append(entry)
            self.storage_bytes += entry.storage_bytes

    def sample(
        self,
        volume_size: int,
        count: int,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[_TeacherEntry, ...]:
        matches = tuple(
            entry for entry in self._items if entry.volume_size == volume_size
        )
        if not matches:
            return ()
        indices = torch.randint(
            len(matches),
            (count,),
            generator=generator,
        ).tolist()
        return tuple(matches[index] for index in indices)


class Connectivity:
    def __init__(
        self,
        *,
        num_phases: int,
        num_domains: int,
        patch_size: int,
        replay_triplets_per_axis: int,
        replay_capacity_per_axis: int,
        max_triplets_per_step: int,
        teacher_bank_bytes: int,
        teacher_min_entries: int,
        max_density: float,
        min_spacing: int,
        mixed_axis_probability: float,
    ) -> None:
        for name, value in (
            ("num_phases", num_phases),
            ("num_domains", num_domains),
            ("patch_size", patch_size),
            ("replay_triplets_per_axis", replay_triplets_per_axis),
            ("max_triplets_per_step", max_triplets_per_step),
            ("teacher_min_entries", teacher_min_entries),
            ("min_spacing", min_spacing),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if num_phases > 256:
            raise ValueError("uint8 teacher storage supports at most 256 phases.")
        if (
            not isinstance(max_density, (int, float))
            or isinstance(max_density, bool)
            or not math.isfinite(max_density)
            or not 0.0 < max_density <= 1.0
        ):
            raise ValueError("max_density must be between zero and one.")
        if (
            not isinstance(mixed_axis_probability, (int, float))
            or isinstance(mixed_axis_probability, bool)
            or not math.isfinite(mixed_axis_probability)
            or not 0.0 <= mixed_axis_probability <= 1.0
        ):
            raise ValueError("mixed_axis_probability must be between zero and one.")

        self.num_phases = num_phases
        self.patch_size = patch_size
        self.replay_triplets_per_axis = replay_triplets_per_axis
        self.max_triplets_per_step = max_triplets_per_step
        self.teacher_min_entries = teacher_min_entries
        self.max_density = float(max_density)
        self.min_spacing = min_spacing
        self.mixed_axis_probability = float(mixed_axis_probability)
        self._replays = tuple(
            _TripletReplay(replay_capacity_per_axis) for _ in range(num_domains)
        )
        self._teachers = tuple(
            _TeacherBank(teacher_bank_bytes) for _ in range(num_domains)
        )

    @property
    def replay_size(self) -> int:
        return sum(map(len, self._replays))

    @property
    def teacher_count(self) -> int:
        return sum(map(len, self._teachers))

    @property
    def teacher_storage_bytes(self) -> int:
        return sum(bank.storage_bytes for bank in self._teachers)

    def teacher_count_for(self, volume_size: int, domain: int) -> int:
        return self._teachers[domain].count_for(volume_size)

    def record_unconditional(self, prediction: torch.Tensor, domain: int) -> None:
        for axis in AXES:
            self._replays[domain].add(
                self._sample_unconditional_triplets(
                    prediction,
                    axis,
                    self.replay_triplets_per_axis,
                )
            )

    def record_seeded(
        self,
        prediction: torch.Tensor,
        seeds: Sequence[PlaneAnchor],
        domain: int,
    ) -> None:
        labels = self._hard_labels(prediction)
        self._teachers[domain].add(labels, seeds, self.num_phases)

    def match_anchor(
        self,
        prediction: torch.Tensor,
        condition: AnchorCondition,
        domain: int,
    ) -> tuple[TripletBatch, TripletBatch]:
        categorical = self._straight_through(prediction)
        candidates = self._sample_anchor_triplets(categorical, condition)
        candidates = self._limit_triplets(candidates)
        real, matched = self._replays[domain].sample_matched(candidates)
        return real, candidates.index_select(matched)

    def sample_teacher(
        self,
        *,
        domain: int,
        volume_size: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> TeacherAnchor | None:
        if self.teacher_count_for(volume_size, domain) < self.teacher_min_entries:
            return None
        entries = self._teachers[domain].sample(
            volume_size,
            batch_size,
            generator=generator,
        )
        if len(entries) != batch_size:
            return None

        target_count = self._sample_plane_count(volume_size, generator)
        plane_sets = [
            self._sample_planes(entry, target_count, generator) for entry in entries
        ]
        actual_count = min(len(planes) for planes in plane_sets)
        if actual_count < 2:
            return None

        conditions = []
        for planes in plane_sets:
            condition = build_anchors(
                planes[:actual_count],
                batch_size=1,
                num_phases=self.num_phases,
                volume_size=volume_size,
                device=device,
                dtype=dtype,
                reconcile=False,
            )
            if condition is None:
                raise RuntimeError("teacher anchor construction returned no condition.")
            conditions.append(condition)
        condition = self._combine_conditions(conditions, actual_count)
        target_vf = torch.stack([entry.target_vf for entry in entries]).to(
            device=device,
            dtype=torch.float32,
        )
        return TeacherAnchor(condition, target_vf)

    def _sample_plane_count(
        self,
        volume_size: int,
        generator: torch.Generator | None,
    ) -> int:
        maximum = max(
            2,
            round(self.max_density * volume_size**3 / self.patch_size**2),
        )
        return int(
            torch.randint(
                2,
                maximum + 1,
                (),
                generator=generator,
            ).item()
        )

    def _sample_planes(
        self,
        entry: _TeacherEntry,
        count: int,
        generator: torch.Generator | None,
    ) -> tuple[PlaneAnchor, ...]:
        seed = entry.seed
        selected = [(seed.axis, seed.index)]
        mixed = bool(
            torch.rand((), generator=generator).item() < self.mixed_axis_probability
        )
        while len(selected) < count:
            candidates = [
                (axis, index)
                for axis in (AXES if mixed else (seed.axis,))
                for index in range(entry.volume_size)
                if (axis, index) not in selected
                and all(
                    other_axis != axis or abs(other_index - index) >= self.min_spacing
                    for other_axis, other_index in selected
                )
            ]
            if not candidates:
                break
            missing_axes = tuple(
                axis for axis in AXES if axis not in {a for a, _ in selected}
            )
            if mixed and missing_axes:
                required = missing_axes[
                    int(
                        torch.randint(
                            len(missing_axes),
                            (),
                            generator=generator,
                        ).item()
                    )
                ]
                required_candidates = [
                    candidate for candidate in candidates if candidate[0] == required
                ]
                if required_candidates:
                    candidates = required_candidates
            choice = int(
                torch.randint(
                    len(candidates),
                    (),
                    generator=generator,
                ).item()
            )
            selected.append(candidates[choice])

        planes = [
            PlaneAnchor(
                image=seed.image.clone(),
                axis=seed.axis,
                index=seed.index,
                position=seed.position,
            )
        ]
        for axis, index in selected[1:]:
            top = self._random_start(entry.volume_size, self.patch_size, generator)
            left = self._random_start(entry.volume_size, self.patch_size, generator)
            image = entry.labels.select(axis, index)[
                top : top + self.patch_size,
                left : left + self.patch_size,
            ].clone()
            planes.append(
                PlaneAnchor(
                    image=image,
                    axis=axis,
                    index=index,
                    position=(top, left),
                )
            )
        return tuple(planes)

    def _sample_unconditional_triplets(
        self,
        volume: torch.Tensor,
        axis: int,
        count: int,
    ) -> TripletBatch:
        self._check_volume(volume)
        self._check_axis(axis)
        moved = volume.movedim(axis + 2, 2)
        depth, height, width = moved.shape[2:]
        if depth < 3:
            raise ValueError("triplet axis size must be at least three.")
        if self.patch_size > min(height, width):
            raise ValueError("patch size must fit inside every triplet plane.")

        batch_indices = torch.randint(
            moved.shape[0],
            (count,),
            device=volume.device,
        )
        starts = torch.randint(depth - 2, (count,), device=volume.device)
        triplets = []
        for batch, start in zip(batch_indices.tolist(), starts.tolist(), strict=True):
            top = self._device_random_start(height, volume.device)
            left = self._device_random_start(width, volume.device)
            logits = moved[
                batch,
                :,
                start : start + 3,
                top : top + self.patch_size,
                left : left + self.patch_size,
            ]
            labels = logits.argmax(dim=0)
            categorical = (
                F.one_hot(
                    labels,
                    num_classes=self.num_phases,
                )
                .movedim(-1, 1)
                .to(dtype=volume.dtype)
                .mul_(2.0)
                .sub_(1.0)
            )
            triplets.append(categorical)
        return TripletBatch(
            values=torch.stack(triplets),
            axes=torch.full(
                (count,),
                axis,
                device=volume.device,
                dtype=torch.long,
            ),
            center_slots=torch.ones(
                count,
                device=volume.device,
                dtype=torch.long,
            ),
        )

    def _sample_anchor_triplets(
        self,
        volume: torch.Tensor,
        condition: AnchorCondition,
    ) -> TripletBatch:
        self._check_volume(volume)
        if condition.axis_masks.shape != (volume.shape[0], 3, *volume.shape[2:]):
            raise ValueError("anchor axis masks must match the generated volume.")
        if condition.mask.shape != (volume.shape[0], 1, *volume.shape[2:]):
            raise ValueError("anchor mask must match the generated volume.")

        triplets = []
        axes = []
        center_slots = []
        occupied: set[tuple[int, int, int, int, int]] = set()
        for batch in range(volume.shape[0]):
            for axis in AXES:
                moved = volume[batch].movedim(axis + 1, 1)
                moved_axis_mask = condition.axis_masks[batch, axis].movedim(axis, 0)
                moved_full_mask = condition.mask[batch].movedim(axis + 1, 1)
                depth, height, width = moved.shape[1:]
                indices = moved_axis_mask.flatten(1).any(dim=1).nonzero().flatten()
                for index_value in indices.tolist():
                    plane_mask = moved_axis_mask[index_value]
                    points = plane_mask.nonzero()
                    if not points.numel():
                        continue
                    row = (int(points[:, 0].min()) + int(points[:, 0].max())) // 2
                    col = (int(points[:, 1].min()) + int(points[:, 1].max())) // 2
                    top = self._centered_start(row, height)
                    left = self._centered_start(col, width)
                    start = self._window_start(index_value, depth)
                    key = (batch, axis, start, top, left)
                    if key in occupied:
                        continue
                    occupied.add(key)
                    values = moved[
                        :,
                        start : start + 3,
                        top : top + self.patch_size,
                        left : left + self.patch_size,
                    ].movedim(0, 1)
                    mask = moved_full_mask[
                        :,
                        start : start + 3,
                        top : top + self.patch_size,
                        left : left + self.patch_size,
                    ].movedim(0, 1)
                    if bool(mask.all().item()):
                        continue
                    triplets.append(values)
                    axes.append(axis)
                    center_slots.append(index_value - start)

        if not triplets:
            return self._empty_triplets(volume)
        return TripletBatch(
            values=torch.stack(triplets),
            axes=torch.tensor(axes, device=volume.device, dtype=torch.long),
            center_slots=torch.tensor(
                center_slots,
                device=volume.device,
                dtype=torch.long,
            ),
        )

    def _limit_triplets(self, batch: TripletBatch) -> TripletBatch:
        if len(batch) <= self.max_triplets_per_step:
            return batch
        selected = []
        for axis in torch.randperm(len(AXES)).tolist():
            candidates = (batch.axes == axis).nonzero().flatten()
            if candidates.numel() and len(selected) < self.max_triplets_per_step:
                choice = int(torch.randint(candidates.numel(), ()).item())
                selected.append(int(candidates[choice]))
        if len(selected) < self.max_triplets_per_step:
            remaining = torch.tensor(
                [index for index in range(len(batch)) if index not in selected],
                device=batch.axes.device,
                dtype=torch.long,
            )
            order = torch.randperm(remaining.numel(), device=remaining.device)
            selected.extend(
                remaining[order[: self.max_triplets_per_step - len(selected)]].tolist()
            )
        indices = torch.tensor(selected, device=batch.axes.device, dtype=torch.long)
        return batch.index_select(indices)

    def _straight_through(
        self,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        self._check_volume(prediction)
        hard = (
            F.one_hot(
                prediction.argmax(dim=1),
                num_classes=self.num_phases,
            )
            .movedim(-1, 1)
            .to(dtype=prediction.dtype)
        )
        values = hard.mul(2.0).sub(1.0)
        values = values + (prediction - prediction.detach())
        return values

    def _hard_labels(
        self,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        self._check_volume(prediction)
        return prediction.detach().argmax(dim=1)

    def _check_volume(self, volume: torch.Tensor) -> None:
        if volume.ndim != 5 or volume.shape[1] != self.num_phases:
            raise ValueError("volume must have shape [B, C, D, H, W].")

    @staticmethod
    def _check_axis(axis: int) -> None:
        if not isinstance(axis, int) or isinstance(axis, bool) or axis not in AXES:
            raise ValueError("axis must be 0, 1, or 2.")

    @staticmethod
    def _window_start(index: int, size: int) -> int:
        if size < 3:
            raise ValueError("triplet axis size must be at least three.")
        if not 0 <= index < size:
            raise ValueError("triplet index is outside the volume.")
        return min(max(index - 1, 0), size - 3)

    @staticmethod
    def _combine_conditions(
        conditions: Sequence[AnchorCondition],
        planes: int,
    ) -> AnchorCondition:
        return AnchorCondition(
            image=torch.cat([condition.image for condition in conditions]),
            mask=torch.cat([condition.mask for condition in conditions]),
            axis_masks=torch.cat([condition.axis_masks for condition in conditions]),
            target=torch.cat([condition.target for condition in conditions]),
            planes=planes,
            conflicts=sum(condition.conflicts for condition in conditions),
            source_voxels=sum(condition.source_voxels for condition in conditions),
        )

    def _empty_triplets(self, volume: torch.Tensor) -> TripletBatch:
        return TripletBatch(
            values=volume.new_empty(
                (0, 3, self.num_phases, self.patch_size, self.patch_size)
            ),
            axes=torch.empty(0, device=volume.device, dtype=torch.long),
            center_slots=torch.empty(0, device=volume.device, dtype=torch.long),
        )

    def _device_random_start(self, size: int, device: torch.device) -> int:
        if size == self.patch_size:
            return 0
        return int(
            torch.randint(
                size - self.patch_size + 1,
                (),
                device=device,
            ).item()
        )

    def _centered_start(self, center: int, size: int) -> int:
        return min(
            max(center - self.patch_size // 2, 0),
            size - self.patch_size,
        )

    @staticmethod
    def _random_start(
        size: int,
        patch_size: int,
        generator: torch.Generator | None,
    ) -> int:
        if size == patch_size:
            return 0
        return int(
            torch.randint(
                size - patch_size + 1,
                (),
                generator=generator,
            ).item()
        )
