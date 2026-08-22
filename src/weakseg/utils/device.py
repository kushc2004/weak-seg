"""Device selection (CUDA -> MPS -> CPU) and seeding utilities."""
from __future__ import annotations

import random

import numpy as np
import torch


def get_device(preference: str = "auto") -> torch.device:
    """Return a torch.device honouring an explicit preference or auto-selecting."""
    pref = preference.lower().strip()
    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA requested but not available.")
    if pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("Apple MPS requested but not available.")
    if pref == "cpu":
        return torch.device("cpu")

    # auto: CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_name(device: torch.device | str) -> str:
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    if dev.type == "mps":
        return "Apple Silicon GPU (MPS)"
    return "CPU"
