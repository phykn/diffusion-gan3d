from collections.abc import Iterator

import torch
from torch.utils.data import DataLoader, RandomSampler

from .dataset import LabelPatchDataset


class BatchStream:
    def __init__(self, loader: DataLoader[torch.Tensor]) -> None:
        self.loader = loader
        self._iterator: Iterator[torch.Tensor] = iter(loader)

    def next(self) -> torch.Tensor:
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader)
            return next(self._iterator)


def build_batch_stream(
    dataset: LabelPatchDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> BatchStream:
    samples_per_cycle = max(1024, batch_size)
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=samples_per_cycle,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    return BatchStream(loader)
