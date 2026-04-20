from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ri.settings.settings import Constants

HSSelectionLiteral = Literal["random_k", "early", "mid", "late"]
HSSelectionMode = HSSelectionLiteral | int
StepsLiteral = Literal["all", "no_steps"]
StepsType = int | StepsLiteral | None
VALID_HS_SELECTIONS = {"random_k", "early", "mid", "late", "step_wise"}
MODEL_DIR = Constants.MODEL_OUTPUT_DIR


def _normalize_hs_selection(value: Any) -> HSSelectionMode:
    if isinstance(value, bool):
        raise ValueError("hs_selection cannot be a boolean value")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in VALID_HS_SELECTIONS:
            return normalized  # type: ignore[return-value]
        try:
            return int(normalized)
        except ValueError:
            pass
    raise ValueError(
        f"Unsupported hs_selection '{value}'. Expected one of {sorted(VALID_HS_SELECTIONS)} "
        "or an integer index."
    )


class PatchConfig(BaseModel):
    """Configuration for hidden state patching experiments."""

    max_gen_len: int = Field(gt=0)
    source_layer: int
    target_layer: int
    patch_position: int | None = None
    steps: StepsType = None
    hs_selection: HSSelectionMode = -1
    patching_k: int = Field(default=1, gt=0)
    include_all_tokens: bool = False
    gen_cache_dir: str | None = None

    @field_validator("steps", mode="before")
    @classmethod
    def _coerce_steps(cls, v: object) -> StepsType:
        if v is None:
            return None
        if isinstance(v, bool):
            raise ValueError("steps must be a positive integer, 'all' or 'no_steps'")
        if isinstance(v, int):
            return v if v > 0 else 1
        if isinstance(v, str):
            normalized = v.strip().lower()
            if normalized == "all":
                return "all"
            if normalized == "no_steps":
                return "no_steps"
            if normalized == "none":
                return None
            try:
                parsed = int(normalized)
            except ValueError as e:
                raise ValueError("steps must be a positive integer, 'all' or 'no_steps'") from e
            return parsed if parsed > 0 else 1
        raise ValueError("steps must be a positive integer, 'all' or 'no_steps'")

    @field_validator("hs_selection", mode="before")
    @classmethod
    def _coerce_hs_selection(cls, v: object) -> HSSelectionMode:
        return _normalize_hs_selection(v)

    @field_validator("patching_k", mode="before")
    @classmethod
    def _coerce_patching_k(cls, v: object) -> int:
        if v is None:
            return 1
        try:
            parsed = int(v)  # type: ignore[call-overload]
        except (TypeError, ValueError):
            return 1
        return parsed if parsed > 0 else 1
