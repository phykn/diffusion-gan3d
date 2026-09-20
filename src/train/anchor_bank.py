import math
from dataclasses import dataclass

import torch

from src.anchor import AnchorCondition, PlaneAnchor, encode_anchors


@dataclass(frozen=True)
class AnchorReplay:
    condition: AnchorCondition
    measured: AnchorCondition
    reference: torch.Tensor
    height: torch.Tensor | None
    profile: torch.Tensor | None
    geometry: list[dict] | None


class AnchorBank:
    def __init__(self, capacity: int = 4, plane_spacing: int = 16):
        if (
            type(capacity) is not int
            or capacity < 1
            or type(plane_spacing) is not int
            or plane_spacing < 1
        ):
            raise ValueError(
                "anchor bank capacity and plane spacing must be positive integers."
            )
        self.capacity = capacity
        self.plane_spacing = plane_spacing
        self.entries: dict[int, list[dict]] = {}

    def add(
        self,
        domain,
        prediction,
        measured,
        visible,
        height=None,
        profile=None,
        geometry=None,
    ):
        entries = self.entries.setdefault(domain, [])
        for batch in visible.nonzero().flatten().tolist():
            # Transfer only when storing a completed, measured-conditioned sample.
            volume = prediction[batch : batch + 1].detach().float().cpu()
            probs = ((volume + 1) * 0.5).clamp(0, 1)
            volume = 2 * probs / probs.sum(1, keepdim=True).clamp_min(1e-8) - 1
            planes = []
            for region in measured.regions:
                image = measured.image[batch : batch + 1].select(
                    region.axis + 2, region.index
                )
                image = image[
                    ...,
                    region.row : region.row + region.height,
                    region.col : region.col + region.width,
                ]
                planes.append(
                    dict(
                        image=(image.detach().float().cpu() + 1) * 0.5,
                        axis=region.axis,
                        index=region.index,
                        position=(region.row, region.col),
                    )
                )
            entries.append(
                dict(
                    geometry=None if geometry is None else geometry[batch],
                    profile=None
                    if profile is None
                    else profile[batch : batch + 1].detach().cpu(),
                    volume=volume,
                    planes=planes,
                    height=None
                    if height is None
                    else height[batch : batch + 1].detach().cpu(),
                )
            )
        del entries[: -self.capacity]

    def sample(self, domain, batch_size, device) -> AnchorReplay | None:
        entries = self.entries.get(domain, [])
        if not entries:
            return None
        entry = entries[int(torch.randint(len(entries), ()))]
        volume = entry["volume"].to(device).expand(batch_size, -1, -1, -1, -1)
        phases, size = volume.shape[1:3]
        measured_planes = [
            PlaneAnchor(
                p["image"].to(device).expand(batch_size, -1, -1, -1),
                p["axis"],
                p["index"],
                tuple(p["position"]),
            )
            for p in entry["planes"]
        ]
        measured = encode_anchors(
            measured_planes,
            batch_size,
            phases,
            size,
            device,
            torch.float32,
            validate=False,
        )
        # Every replay includes the measurement; reconcile pseudo intersections to it.
        reconciled = torch.where(measured.mask, measured.image, volume)
        count = min(3 * size, max(2, math.ceil(size / self.plane_spacing)))
        occupied = {(p.axis, p.index) for p in measured_planes}
        candidates = [
            (axis, index)
            for axis in range(3)
            for index in range(size)
            if (axis, index) not in occupied
        ]
        order = torch.randperm(len(candidates))[
            : max(0, count - len(measured_planes))
        ].tolist()
        planes = list(measured_planes)
        for slot in order:
            axis, index = candidates[slot]
            planes.append(
                PlaneAnchor((reconciled.select(axis + 2, index) + 1) * 0.5, axis, index)
            )
        condition = encode_anchors(
            planes,
            batch_size,
            phases,
            size,
            device,
            torch.float32,
            reconcile=True,
            validate=False,
        )
        height = entry["height"]
        if height is not None:
            height = height.to(device).expand(batch_size, -1, -1, -1, -1)
        # Continuity targets describe the replay's changes, never artificial
        # jumps introduced by pasting a measured plane. Anchor loss owns the
        # measured values themselves; reconciliation is for conditions only.
        profile = entry["profile"]
        return AnchorReplay(
            condition=condition,
            measured=measured,
            reference=volume,
            height=height,
            profile=None
            if profile is None
            else profile.to(device).expand(batch_size, -1, -1),
            geometry=[entry["geometry"]] * batch_size
            if entry["geometry"] is not None
            else None,
        )
