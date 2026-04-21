from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluationConfig(BaseModel):
    """Configuration for batched evaluation."""

    batch_size: int = Field(default=16, gt=0)
    max_gen_len: int = Field(default=400, gt=0)
    seed: int = 42
