from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

StepsLiteral = Literal["all", "no_steps"]
StepsType = int | StepsLiteral | None


class EvaluationConfig(BaseModel):
    """Configuration for batched evaluation."""

    batch_size: int = Field(default=16, gt=0)
    max_gen_len: int = Field(default=400, gt=0)
    steps: StepsType = None
    seed: int = 42

    @field_validator("steps", mode="before")
    @classmethod
    def _coerce_steps(cls, v: object) -> StepsType:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("steps must be an integer, 'all', or 'no_steps'")
        if isinstance(v, int):
            return v if v > 0 else 1
        if isinstance(v, str):
            normalized = v.strip().lower()
            if normalized in ("all", "no_steps"):
                return normalized  # type: ignore[return-value]
            if normalized == "none":
                return None
            try:
                parsed = int(normalized)
            except ValueError as e:
                raise ValueError("steps must be an integer, 'all', 'no_steps', or None") from e
            return parsed if parsed > 0 else 1
        raise ValueError("steps must be an integer, 'all', 'no_steps', or None")
