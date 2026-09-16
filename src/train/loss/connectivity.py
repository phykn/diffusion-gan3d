import torch

from src.data.slice import TripletBatch


def compute_transition_loss(real: TripletBatch, fake: TripletBatch) -> torch.Tensor:
    if real.values.shape != fake.values.shape:
        raise ValueError("real and fake triplets must have the same shape.")
    if not (
        (real.axes is fake.axes or torch.equal(real.axes, fake.axes))
        and (real.gaps is fake.gaps or torch.equal(real.gaps, fake.gaps))
        and (
            real.center_slots is fake.center_slots
            or torch.equal(real.center_slots, fake.center_slots)
        )
    ):
        raise ValueError("real and fake triplets must use matching metadata.")
    if len(fake) == 0:
        return fake.values.sum().mul(0.0)

    triplet_indices = torch.arange(len(real), device=real.values.device)
    center_slots = real.center_slots
    valid_neighbors = torch.stack(
        (center_slots > 0, center_slots < 2),
        dim=1,
    )
    probs = torch.stack(
        (
            real.values.to(torch.float32),
            fake.values.to(torch.float32),
        )
    )
    probs = (probs + 1.0) * 0.5
    centers = probs[:, triplet_indices, center_slots]
    left = probs[:, triplet_indices, (center_slots - 1).clamp_min(0)]
    right = probs[:, triplet_indices, (center_slots + 1).clamp_max(2)]
    changes = torch.stack((left - centers, right - centers), dim=2)
    transition_error = (changes[0] - changes[1]).abs().mean(dim=(2, 3, 4))
    valid_count = valid_neighbors.sum(dim=1)
    per_triplet = (transition_error * valid_neighbors).sum(dim=1) / valid_count
    middle = center_slots == 1
    bend = changes[:, :, 1] - changes[:, :, 0]
    bend_error = 0.5 * (bend[0] - bend[1]).abs().mean(dim=(1, 2, 3))
    per_triplet = torch.where(middle, 0.5 * (per_triplet + bend_error), per_triplet)
    return per_triplet.mean()
