from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluationConfig:
    """Configuration for batched evaluation."""

    batch_size: int = 16
    max_gen_len: int = 400
    steps: Optional[int] = None
    seed: int = 42
