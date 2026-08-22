"""Shared RNG helpers used by dataset workers to keep augmentations deterministic."""
from __future__ import annotations


def worker_init_fn(worker_id: int) -> None:  # pragma: no cover - exercised via DataLoader
    import numpy as np
    import torch

    seed = torch.initial_seed() % 2**32
    np.random.seed((seed + worker_id) % 2**32)
