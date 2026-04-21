from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ri.settings.settings import Constants

ExtractionMode = Literal["flexible", "strict"]
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
    extraction_mode: ExtractionMode = "flexible"

    @field_validator("hs_selection", mode="before")
    @classmethod
    def _coerce_hs_selection(cls, v: Any) -> int:
        if isinstance(v, bool):
            raise ValueError("hs_selection cannot be a boolean value")
        try:
            return int(v)
        except (TypeError, ValueError) as e:
            raise ValueError(f"hs_selection must be an integer, got {v!r}") from e

    @field_validator("extraction_mode", mode="before")
    @classmethod
    def _validate_extraction_mode(cls, v: Any) -> str:
        if v not in ("flexible", "strict"):
            raise ValueError(f"extraction_mode must be 'flexible' or 'strict', got {v!r}")
        return str(v)
