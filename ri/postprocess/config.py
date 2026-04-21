from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class PostprocessConfig(BaseModel):
    sweep_root: str = Field(min_length=1)
    output_file: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    alignment_model: str = Field(min_length=1)
    output_schema: str = Field(default="full_results", min_length=1)
    pe_root: str | None = None
    eval_json: str | None = None
    sample_idx: int = 0
    spacy_model: str = Field(default="en_core_web_sm", min_length=1)
    generation_other_label: str | None = None
    progress_every: int = Field(default=25, ge=0)
    source_tokens_file: str | None = None
    entity_codes_file: str | None = None
    behavior_codes_file: str | None = None

    @field_validator(
        "sweep_root",
        "output_file",
        "model_name",
        "alignment_model",
        "output_schema",
        "spacy_model",
        mode="before",
    )
    @classmethod
    def _strip_required_text(cls, value: object) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError("Expected a non-empty string value")
        return text

    @field_validator("alignment_model", mode="before")
    @classmethod
    def _validate_alignment_model(cls, value: object) -> str:
        text = str(value).strip().lower()
        if text not in {"llama", "qwen"}:
            raise ValueError("alignment_model must be either 'llama' or 'qwen'")
        return text

    @field_validator("output_schema", mode="before")
    @classmethod
    def _validate_output_schema(cls, value: object) -> str:
        text = str(value).strip().lower()
        if text not in {"full_results", "published_export"}:
            raise ValueError("output_schema must be either 'full_results' or 'published_export'")
        return text

    @field_validator("pe_root", "eval_json", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("generation_other_label", mode="before")
    @classmethod
    def _validate_generation_other_label(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip().lower()
        if text not in {"other", "noise"}:
            raise ValueError("generation_other_label must be either 'other' or 'noise'")
        return text

    @model_validator(mode="after")
    def _validate_schema_dependencies(self) -> PostprocessConfig:
        if self.output_schema == "published_export" and not self.pe_root:
            raise ValueError("pe_root is required when output_schema='published_export'")
        return self
