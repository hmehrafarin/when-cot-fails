from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from ri.settings.settings import Constants

MODEL_DIR = Constants.MODEL_OUTPUT_DIR


class PatchConfig(BaseModel):
    """Configuration for hidden state patching experiments."""

    max_gen_len: int = Field(gt=0)
    source_layer: int
    target_layer: int
    patch_position: int | None = None
    hs_selection: int = -1
    include_all_tokens: bool = False
    gen_cache_dir: str | None = None

    @field_validator("hs_selection", mode="before")
    @classmethod
    def _coerce_hs_selection(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("hs_selection cannot be a boolean value")
        try:
            return int(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"hs_selection must be an integer, got {v!r}") from e
