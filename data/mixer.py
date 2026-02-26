from __future__ import annotations

import random
from typing import Iterator, Optional

from torch.utils.data import Sampler


class MixedBatchSampler(Sampler):
    def __init__(
        self,
        off_size: int,
        on_size: int,
        ratio: float,
        total_steps: int,
        seed: int = 0,
        conflict_scores: Optional[list] = None,
    ):
        self.off_size = off_size
        self.on_size = on_size
        self.ratio = max(0.0, min(1.0, ratio))
        self.total_steps = total_steps
        self.seed = seed
        self.conflict_scores = conflict_scores or []

    def __iter__(self) -> Iterator[tuple]:
        rng = random.Random(self.seed)
        off_stream = _shuffled_stream(self.off_size, rng)
        if self.on_size <= 0:
            on_stream = None
        elif self.conflict_scores and len(self.conflict_scores) == self.on_size:
            on_stream = _weighted_stream(self.on_size, self.conflict_scores, rng)
        else:
            on_stream = _shuffled_stream(self.on_size, rng)
        for _ in range(self.total_steps):
            use_on = self.on_size > 0 and rng.random() < self.ratio
            if use_on:
                yield ("on", next(on_stream))  # type: ignore[arg-type]
            else:
                yield ("off", next(off_stream))

    def __len__(self) -> int:
        return self.total_steps


def _shuffled_stream(size: int, rng: random.Random) -> Iterator[int]:
    while True:
        indices = list(range(size))
        rng.shuffle(indices)
        for idx in indices:
            yield idx


def _weighted_stream(size: int, weights: list, rng: random.Random) -> Iterator[int]:
    population = list(range(size))
    while True:
        yield rng.choices(population, weights=weights, k=1)[0]
