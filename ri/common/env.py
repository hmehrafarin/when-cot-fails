from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch
from transformers import set_seed
import os


def set_random_seed(seed: int = 42, deterministic: Optional[bool] = True) -> None:
    """
    Set global RNG state for reproducible runs.

    Parameters
    ----------
    seed:
        Base seed used across Python, NumPy, and Torch.
    deterministic:
        Optional flag forwarded to Torch to enforce deterministic CUDA kernels.
        When ``None`` Torch keeps its default.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic is not None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(deterministic)

    set_seed(seed)
