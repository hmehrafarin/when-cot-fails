from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ExtractionMode = Literal["flexible", "strict"]


class EvaluationConfig(BaseModel):
    """Configuration for batched evaluation."""

    batch_size: int = Field(default=16, gt=0)
    max_gen_len: int = Field(default=400, gt=0)
    seed: int = 42
    extraction_mode: ExtractionMode = "flexible"

    @field_validator("extraction_mode", mode="before")
    @classmethod
    def _validate_extraction_mode(cls, v: Any) -> str:
        if v not in ("flexible", "strict"):
            raise ValueError(f"extraction_mode must be 'flexible' or 'strict', got {v!r}")
        return str(v)
