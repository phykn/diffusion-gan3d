from collections.abc import Iterator
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler

from src.data.dataset import RealDataset


class BatchStream:
    def __init__(self, loader: DataLoader[dict]) -> None:
        self.loader = loader
        self.iterator: Iterator[dict] = iter(loader)

    def next(self) -> dict:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        return batch


class FolderBatchSampler(Sampler[list[Path]]):
    def __init__(self, dataset: RealDataset, batch_size: int) -> None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            raise ValueError("batch size must be a positive integer.")

        self.batch_size = batch_size
        self.num_batches = max(len(dataset), batch_size) // batch_size
        self._path_groups = dataset.path_groups

    def __iter__(self) -> Iterator[list[Path]]:
        for _ in range(self.num_batches):
            bucket_index = int(torch.randint(len(self._path_groups), ()).item())
            bucket = self._path_groups[bucket_index]
            choices = torch.randint(len(bucket), (self.batch_size,))
            yield [bucket[int(choice)] for choice in choices]

    def __len__(self) -> int:
        return self.num_batches
