from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ri.config.settings import Constants

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


@dataclass
class PatchConfig:
    """Configuration for hidden state patching experiments."""

    max_gen_len: int
    source_layer: int
    target_layer: int
    patch_position: int | None = None
    steps: StepsType = None
    hs_selection: HSSelectionMode = -1
    patching_k: int | None = None
    include_all_tokens: bool = False
    gen_cache_dir: str | None = None

    def __post_init__(self) -> None:
        if self.max_gen_len <= 0:
            raise ValueError("max_gen_len must be a positive integer")

        if isinstance(self.steps, int):
            if self.steps <= 0:
                self.steps = 1
        elif isinstance(self.steps, str):
            normalized = self.steps.strip().lower()
            if normalized == "all":
                self.steps = "all"
            elif normalized == "no_steps":
                self.steps = "no_steps"
            else:
                raise ValueError("steps must be a positive integer, 'all' or 'no_steps'")
        elif self.steps is None:
            pass
        else:
            raise ValueError("steps must be a positive integer, 'all' or 'no_steps'")

        self.hs_selection = _normalize_hs_selection(self.hs_selection)

        if not self.patching_k or self.patching_k <= 0:
            self.patching_k = 1

        self.include_all_tokens = bool(self.include_all_tokens)
