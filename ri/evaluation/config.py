from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    """Configuration for batched evaluation."""

    batch_size: int = 16
    max_gen_len: int = 400
    steps: int | None = None
    seed: int = 42
