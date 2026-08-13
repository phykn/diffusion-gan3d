import torch
from torch import nn

NULL_DOMAIN = -1


def masked_domain_embedding(
    embedding: nn.Embedding,
    domain: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Embed domain IDs while treating -1 as an absent domain condition."""
    if not isinstance(domain, torch.Tensor) or domain.ndim != 1:
        raise ValueError("domain must have shape [B].")
    domain = domain.to(device=device, dtype=torch.long)
    if domain.numel() and (
        int(domain.min()) < NULL_DOMAIN or int(domain.max()) >= embedding.num_embeddings
    ):
        raise ValueError("domain contains an invalid ID.")
    present = domain != NULL_DOMAIN
    values = embedding(domain.clamp_min(0)).to(dtype=dtype)
    return values * present[:, None].to(dtype=dtype)
