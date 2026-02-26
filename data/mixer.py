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
        for _ in range(self.total_steps):
            use_on = self.on_size > 0 and rng.random() < self.ratio
            if use_on:
                if self.conflict_scores and len(self.conflict_scores) == self.on_size:
                    idx = rng.choices(range(self.on_size), weights=self.conflict_scores, k=1)[0]
                else:
                    idx = rng.randrange(self.on_size)
                yield ("on", idx)
            else:
                idx = rng.randrange(self.off_size)
                yield ("off", idx)

    def __len__(self) -> int:
        return self.total_steps
