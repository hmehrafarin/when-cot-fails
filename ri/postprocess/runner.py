from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from transformers import AutoTokenizer

from .codebooks import ENTITY_ROLE_CODES
from .config import PostprocessConfig
from .generation_labels import classify_generation_type, generation_type_codes
from .spacy_rules import (
    PRIORITY,
    VALID_LABELS,
    LabeledSpan,
    get_nlp,
    label_reasoning_with_question,
)

SAMPLE_DIR_RE = re.compile(r"^sample_(\d+)$")
SWEEP_FILE_RE = re.compile(r"^layer_(\d+)_pos_(-?\d+)\.json$")
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
ANSWER_PREFIX_RE = re.compile(r"(?is)^\s*(?:final\s+answer|answer)\s*[:\-]?\s*")
FINAL_ANSWER_RE = re.compile(r"(?im)^\s*final answer\s*:")
FINAL_NUMERIC_RE = re.compile(r"^\s*[-+]?\$?\d[\d,]*(?:\.\d+)?\s*%?\s*$")
SETUP_HINT_RE = re.compile(
    r"(?i)\b(let'?s|we(?:\s+need|\s+have|\s+can|\s+will|\s+must)|given|to find|to solve|first|start|consider|we are asked)\b"
)
QUESTION_RE = re.compile(r"Question:\s*(.*?)\s*Answer:", flags=re.IGNORECASE | re.DOTALL)


def run_postprocess(
    *,
    sweep_root: str,
    output_file: str,
    tokenizer_name: str,
    alignment_model: str,
    output_schema: str = "full_results",
    pe_root: str | None = None,
    eval_json: str | None = None,
    sample_idx: int = 0,
    spacy_model: str = "en_core_web_sm",
    generation_other_label: str | None = None,
    progress_every: int = 25,
    source_tokens_file: str | None = None,
    entity_codes_file: str | None = None,
    behavior_codes_file: str | None = None,
) -> None:
    config = PostprocessConfig(
        sweep_root=sweep_root,
        output_file=output_file,
        tokenizer_name=tokenizer_name,
        alignment_model=alignment_model,
        output_schema=output_schema,
        pe_root=pe_root,
        eval_json=eval_json,
        sample_idx=sample_idx,
        spacy_model=spacy_model,
        generation_other_label=generation_other_label,
        progress_every=progress_every,
        source_tokens_file=source_tokens_file,
        entity_codes_file=entity_codes_file,
        behavior_codes_file=behavior_codes_file,
    )
    PostprocessRunner(config).run()


def run_full_results(
    *,
    sweep_root: str,
    output_file: str,
    tokenizer_name: str,
    alignment_model: str,
    sample_idx: int = 0,
    spacy_model: str = "en_core_web_sm",
    generation_other_label: str | None = None,
    progress_every: int = 25,
    source_tokens_file: str | None = None,
    entity_codes_file: str | None = None,
    behavior_codes_file: str | None = None,
) -> None:
    run_postprocess(
        sweep_root=sweep_root,
        output_file=output_file,
        tokenizer_name=tokenizer_name,
        alignment_model=alignment_model,
        output_schema="full_results",
        sample_idx=sample_idx,
        spacy_model=spacy_model,
        generation_other_label=generation_other_label,
        progress_every=progress_every,
        source_tokens_file=source_tokens_file,
        entity_codes_file=entity_codes_file,
        behavior_codes_file=behavior_codes_file,
    )


class PostprocessRunner:
    """Build postprocessed patch-sweep tables in a selected output schema."""

    def __init__(self, config: PostprocessConfig):
        self.config = config
        self.sweep_root = Path(config.sweep_root)
        self.pe_root = Path(config.pe_root) if config.pe_root else None
        self.eval_json = Path(config.eval_json) if config.eval_json else None
        self.output_file = Path(config.output_file)
        self.source_tokens_file = _derive_sidecar_path(
            self.output_file,
            config.source_tokens_file,
            "source_tokens",
        )
        self.entity_codes_file = _derive_sidecar_path(
            self.output_file,
            config.entity_codes_file,
            "entity_codes",
        )
        self.behavior_codes_file = _derive_sidecar_path(
            self.output_file,
            config.behavior_codes_file,
            "behavior_codes",
        )

    def run(self) -> None:
        if not self.sweep_root.exists():
            raise FileNotFoundError(f"Sweep root does not exist: {self.sweep_root}")
        if self.pe_root is not None and not self.pe_root.exists():
            raise FileNotFoundError(f"PE root does not exist: {self.pe_root}")
        if self.eval_json is not None and not self.eval_json.exists():
            raise FileNotFoundError(f"Eval JSON does not exist: {self.eval_json}")

        nlp = _load_nlp(self.config.spacy_model)
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name, use_fast=True)
        token_len_cache: dict[str, int] = {}
        generation_other_label = _effective_generation_other_label(
            self.config.output_schema,
            self.config.generation_other_label,
        )
        eval_items = (
            _load_eval_items(self.eval_json)
            if self.config.output_schema == "published_export" and self.eval_json is not None
            else None
        )

        sample_sources = _discover_sample_sources(self.sweep_root, self.config.sample_idx)
        output_rows = 0
        source_token_rows = 0
        output_fieldnames = _output_fieldnames(self.config.output_schema)

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.source_tokens_file.parent.mkdir(parents=True, exist_ok=True)
        self.entity_codes_file.parent.mkdir(parents=True, exist_ok=True)
        self.behavior_codes_file.parent.mkdir(parents=True, exist_ok=True)

        with (
            self.output_file.open("w", newline="", encoding="utf-8") as output_handle,
            self.source_tokens_file.open("w", newline="", encoding="utf-8") as source_tokens_handle,
        ):
            output_writer = csv.DictWriter(output_handle, fieldnames=output_fieldnames)
            source_tokens_writer = csv.DictWriter(
                source_tokens_handle,
                fieldnames=[
                    "sample_idx",
                    "source_pos",
                    "patched_token_str",
                    "step_id",
                    "label",
                    "entity_role",
                ],
            )
            output_writer.writeheader()
            source_tokens_writer.writeheader()

            for processed, (sample_idx, sample_dir) in enumerate(sample_sources, start=1):
                sample_data = self._load_sample_data(
                    sample_idx=sample_idx,
                    sample_dir=sample_dir,
                    eval_items=eval_items,
                    nlp=nlp,
                    tokenizer=tokenizer,
                    generation_other_label=generation_other_label,
                    token_len_cache=token_len_cache,
                )

                for row in sample_data["source_tokens"]:
                    source_tokens_writer.writerow(row)
                    source_token_rows += 1

                for row in sample_data["output_rows"]:
                    output_writer.writerow(row)
                    output_rows += 1

                if self.config.progress_every > 0 and processed % self.config.progress_every == 0:
                    print(
                        "[progress] "
                        f"processed_samples={processed} "
                        f"source_token_rows={source_token_rows} "
                        f"output_rows={output_rows}"
                    )

        _write_codebook(
            self.entity_codes_file,
            codebook_name="entity_role",
            codes=ENTITY_ROLE_CODES,
        )
        _write_codebook(
            self.behavior_codes_file,
            codebook_name="generation_type",
            codes=generation_type_codes(generation_other_label),
        )

        print(f"Processed samples: {len(sample_sources)}")
        print(f"Output schema: {self.config.output_schema}")
        print(f"Source token rows written: {source_token_rows}")
        print(f"Output rows written: {output_rows}")
        print(f"Saved output CSV: {self.output_file}")
        print(f"Saved source token CSV: {self.source_tokens_file}")
        print(f"Saved entity codebook: {self.entity_codes_file}")
        print(f"Saved behavior codebook: {self.behavior_codes_file}")

    def _load_sample_data(
        self,
        *,
        sample_idx: int,
        sample_dir: Path,
        eval_items: list[dict[str, Any]] | None,
        nlp,
        tokenizer,
        generation_other_label: str,
        token_len_cache: dict[str, int],
    ) -> dict[str, list[dict[str, Any]]]:
        if self.config.output_schema == "full_results":
            return _load_full_results_sample(
                sample_idx=sample_idx,
                sample_dir=sample_dir,
                nlp=nlp,
                tokenizer=tokenizer,
                alignment_model=self.config.alignment_model,
                generation_other_label=generation_other_label,
                token_len_cache=token_len_cache,
            )

        if self.config.output_schema == "published_export":
            assert self.pe_root is not None
            eval_item = None
            if eval_items is not None:
                if sample_idx >= len(eval_items):
                    raise RuntimeError(
                        f"Eval JSON has {len(eval_items)} samples but sample {sample_idx} was requested."
                    )
                eval_item = eval_items[sample_idx]
            return _load_published_export_sample(
                sample_idx=sample_idx,
                sample_dir=sample_dir,
                pe_dir=self.pe_root / f"sample_{sample_idx}",
                eval_item=eval_item,
                nlp=nlp,
                tokenizer=tokenizer,
                alignment_model=self.config.alignment_model,
                generation_other_label=generation_other_label,
                token_len_cache=token_len_cache,
            )

        raise ValueError(f"Unsupported output schema: {self.config.output_schema!r}")


def _effective_generation_other_label(output_schema: str, configured_label: str | None) -> str:
    if configured_label is not None:
        return configured_label
    if output_schema == "published_export":
        return "other"
    return "noise"


def _output_fieldnames(output_schema: str) -> list[str]:
    if output_schema == "full_results":
        return [
            "sample_idx",
            "layer",
            "target_pos",
            "target_pos_resolved",
            "source_pos",
            "patched_token_str",
            "step_id",
            "label",
            "generation_type",
            "generated_text",
            "pred_num",
            "is_numeric",
            "is_correct",
            "abs_error",
            "signed_error",
            "generated_token_length",
            "average_generated_token_length",
            "source_generated_token_length",
            "generated_token_length_delta_vs_source",
            "average_generated_token_length_delta_vs_source",
            "entity_role",
        ]
    if output_schema == "published_export":
        return [
            "sample_idx",
            "layer",
            "target_pos",
            "target_pos_resolved",
            "source_pos",
            "patched_token_str",
            "generation_type",
            "pe",
            "generated_text",
            "pred_num",
            "is_numeric",
            "is_correct",
            "abs_error",
            "signed_error",
            "generated_token_length",
            "average_generated_token_length",
            "source_generated_token_length",
            "generated_token_length_delta_vs_source",
            "average_generated_token_length_delta_vs_source",
            "entity_role",
        ]
    raise ValueError(f"Unsupported output schema: {output_schema!r}")


def _derive_sidecar_path(output_file: Path, configured_path: str | None, suffix: str) -> Path:
    if configured_path:
        return Path(configured_path)
    extension = ".json" if suffix.endswith("codes") else ".csv"
    return output_file.with_name(f"{output_file.stem}_{suffix}{extension}")


def _load_nlp(model_name: str):
    try:
        return get_nlp(model_name)
    except OSError as e:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Run `uv run python -m spacy download {model_name}` and retry."
        ) from e


def _write_codebook(file_path: Path, *, codebook_name: str, codes: dict[str, str]) -> None:
    payload = {
        "name": codebook_name,
        "codes": [{"code": code, "description": description} for code, description in codes.items()],
    }
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _discover_sample_sources(sweep_root: Path, sample_idx: int) -> list[tuple[int, Path]]:
    sample_dirs = [
        path for path in sweep_root.iterdir() if path.is_dir() and _parse_sample_idx(path.name) is not None
    ]
    if sample_dirs:
        return [
            (_parse_sample_idx(path.name), path)
            for path in sorted(sample_dirs, key=lambda path: (_parse_sample_idx(path.name), path.name))
        ]

    flat_sweep_files = _sort_sweep_files(sweep_root)
    if flat_sweep_files:
        return [(sample_idx, sweep_root)]

    raise FileNotFoundError(
        f"No patch sweep files found under {sweep_root}. "
        "Expected either sample_<idx>/layer_<L>_pos_<T>.json directories or flat layer_<L>_pos_<T>.json files."
    )


@dataclass(slots=True)
class PESampleMetadata:
    patched_map: dict[int, str]
    source_target_meta: dict[int, tuple[dict[int, int], list[int]]]
    pe_map: dict[tuple[int, int, int], float]
    cot_text_hint: str


def _load_full_results_sample(
    *,
    sample_idx: int,
    sample_dir: Path,
    nlp,
    tokenizer,
    alignment_model: str,
    generation_other_label: str,
    token_len_cache: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    sweep_files = _sort_sweep_files(sample_dir)
    if not sweep_files:
        raise FileNotFoundError(f"No sweep files found for sample {sample_idx} in {sample_dir}")

    canonical_file = _choose_canonical_sweep_file(sweep_files)
    if canonical_file is None:
        raise FileNotFoundError(f"Could not choose a canonical sweep file in {sample_dir}")

    canonical_payload = json.loads(canonical_file.read_text(encoding="utf-8"))
    token_map = _build_source_token_map_from_payload(canonical_payload)
    source_generated_answer = str(canonical_payload.get("source_generated_answer", ""))
    question = _extract_question_from_payload(canonical_payload)
    step_meta = _build_step_mapping(token_map, cot_text_hint=source_generated_answer)
    sorted_step_positions = sorted(step_meta)
    entity_role_map = _build_entity_role_map_for_model(
        token_map,
        alignment_model=alignment_model,
        question=question,
        reasoning=source_generated_answer,
        tokenizer=tokenizer,
        nlp=nlp,
    )

    source_tokens = [
        {
            "sample_idx": sample_idx,
            "source_pos": source_pos,
            "patched_token_str": token_map[source_pos],
            "step_id": step_meta.get(source_pos, (0, "Setup"))[0],
            "label": step_meta.get(source_pos, (0, "Setup"))[1],
            "entity_role": entity_role_map.get(source_pos, "OTHER"),
        }
        for source_pos in sorted(token_map)
    ]

    source_generated_token_length = _count_token_lengths_batch(
        [source_generated_answer],
        tokenizer,
        token_len_cache,
    )[0]

    output_rows: list[dict[str, Any]] = []
    for sweep_path in sweep_files:
        payload = json.loads(sweep_path.read_text(encoding="utf-8"))
        layer = int(payload.get("layer", 0))
        target_pos_requested = int(payload.get("target_pos", 0))
        target_pos_resolved = target_pos_requested
        patch_results = payload.get("patch_result") or []
        gold_num = _parse_number(payload.get("target_gold_answer"))

        generated_texts = [str(item.get("generated_text", "")) for item in patch_results]
        generated_lengths = _count_token_lengths_batch(generated_texts, tokenizer, token_len_cache)
        avg_generated_length = float(mean(generated_lengths)) if generated_lengths else float("nan")

        for item, generated_text, generated_len in zip(
            patch_results,
            generated_texts,
            generated_lengths,
            strict=True,
        ):
            source_pos = int(item["pos"])
            pred_num = _parse_number(generated_text)
            is_numeric = pred_num is not None
            is_correct = bool(
                is_numeric
                and gold_num is not None
                and math.isclose(pred_num, gold_num, rel_tol=0.0, abs_tol=1e-9)
            )
            abs_error = abs(pred_num - gold_num) if (is_numeric and gold_num is not None) else float("nan")
            signed_error = pred_num - gold_num if (is_numeric and gold_num is not None) else float("nan")
            step_id, step_label = _step_for_source_pos(step_meta, sorted_step_positions, source_pos)

            output_rows.append(
                {
                    "sample_idx": sample_idx,
                    "layer": layer,
                    "target_pos": target_pos_requested,
                    "target_pos_resolved": target_pos_resolved,
                    "source_pos": source_pos,
                    "patched_token_str": token_map.get(source_pos, item.get("patching_token", "")),
                    "step_id": step_id,
                    "label": step_label,
                    "generation_type": classify_generation_type(
                        generated_text,
                        other_label=generation_other_label,
                    ),
                    "generated_text": generated_text,
                    "pred_num": pred_num,
                    "is_numeric": is_numeric,
                    "is_correct": is_correct,
                    "abs_error": abs_error,
                    "signed_error": signed_error,
                    "generated_token_length": generated_len,
                    "average_generated_token_length": avg_generated_length,
                    "source_generated_token_length": source_generated_token_length,
                    "generated_token_length_delta_vs_source": generated_len - source_generated_token_length,
                    "average_generated_token_length_delta_vs_source": avg_generated_length
                    - source_generated_token_length,
                    "entity_role": entity_role_map.get(source_pos, "OTHER"),
                }
            )

    return {"source_tokens": source_tokens, "output_rows": output_rows}


def _load_published_export_sample(
    *,
    sample_idx: int,
    sample_dir: Path,
    pe_dir: Path,
    eval_item: dict[str, Any] | None,
    nlp,
    tokenizer,
    alignment_model: str,
    generation_other_label: str,
    token_len_cache: dict[str, int],
) -> dict[str, list[dict[str, Any]]]:
    sweep_files = _sort_sweep_files(sample_dir)
    if not sweep_files:
        raise FileNotFoundError(f"No sweep files found for sample {sample_idx} in {sample_dir}")

    if not pe_dir.exists():
        raise FileNotFoundError(f"Missing PE directory for sample {sample_idx}: {pe_dir}")

    canonical_file = _choose_canonical_sweep_file(sweep_files)
    if canonical_file is None:
        raise FileNotFoundError(f"Could not choose a canonical sweep file in {sample_dir}")

    canonical_payload = json.loads(canonical_file.read_text(encoding="utf-8"))
    token_map = _build_source_token_map_from_payload(canonical_payload)
    source_generated_answer = str(canonical_payload.get("source_generated_answer", ""))
    question = _extract_question_from_eval_item(eval_item) or _extract_question_from_payload(canonical_payload)
    canonical_reasoning = _extract_reasoning_from_eval_item(eval_item) or source_generated_answer
    requested_targets = {
        int(SWEEP_FILE_RE.match(path.name).group(2))
        for path in sweep_files
        if SWEEP_FILE_RE.match(path.name)
    }
    pe_metadata = _load_pe_sample_metadata(pe_dir, requested_targets)
    cot_text_hint = pe_metadata.cot_text_hint or source_generated_answer
    step_meta = _build_step_mapping(token_map, cot_text_hint=cot_text_hint)
    entity_role_map = _build_entity_role_map_for_model(
        token_map,
        alignment_model=alignment_model,
        question=question,
        reasoning=canonical_reasoning,
        tokenizer=tokenizer,
        nlp=nlp,
    )

    source_tokens = [
        {
            "sample_idx": sample_idx,
            "source_pos": source_pos,
            "patched_token_str": token_map[source_pos],
            "step_id": step_meta.get(source_pos, (0, "Setup"))[0],
            "label": step_meta.get(source_pos, (0, "Setup"))[1],
            "entity_role": entity_role_map.get(source_pos, "OTHER"),
        }
        for source_pos in sorted(token_map)
    ]

    source_generated_token_length = _count_token_lengths_batch(
        [source_generated_answer],
        tokenizer,
        token_len_cache,
    )[0]

    output_rows: list[dict[str, Any]] = []
    for sweep_path in sweep_files:
        payload = json.loads(sweep_path.read_text(encoding="utf-8"))
        layer = int(payload.get("layer", 0))
        target_pos_requested = int(payload.get("target_pos", 0))
        patch_results = payload.get("patch_result") or []
        gold_num = _parse_number(payload.get("target_gold_answer"))

        generated_texts = [str(item.get("generated_text", "")) for item in patch_results]
        generated_lengths = _count_token_lengths_batch(generated_texts, tokenizer, token_len_cache)
        pending_rows: list[dict[str, Any]] = []
        group_lengths: dict[int, list[int]] = {}

        for item, generated_text, generated_len in zip(
            patch_results,
            generated_texts,
            generated_lengths,
            strict=True,
        ):
            source_pos = int(item["pos"])
            pred_num = _parse_number(generated_text)
            is_numeric = pred_num is not None
            is_correct = bool(
                is_numeric
                and gold_num is not None
                and math.isclose(pred_num, gold_num, rel_tol=0.0, abs_tol=1e-9)
            )
            abs_error = abs(pred_num - gold_num) if (is_numeric and gold_num is not None) else float("nan")
            signed_error = pred_num - gold_num if (is_numeric and gold_num is not None) else float("nan")
            req_to_resolved, resolved_positions = pe_metadata.source_target_meta.get(source_pos, ({}, []))
            target_pos_resolved = _resolve_target_position(
                target_pos_requested,
                req_to_resolved,
                resolved_positions,
            )
            pe = pe_metadata.pe_map.get((layer, target_pos_resolved, source_pos), float("nan"))
            group_lengths.setdefault(target_pos_resolved, []).append(generated_len)
            pending_rows.append(
                {
                    "sample_idx": sample_idx,
                    "layer": layer,
                    "target_pos": target_pos_requested,
                    "target_pos_resolved": target_pos_resolved,
                    "source_pos": source_pos,
                    "patched_token_str": pe_metadata.patched_map.get(
                        source_pos,
                        token_map.get(source_pos, item.get("patching_token", "")),
                    ),
                    "generation_type": classify_generation_type(
                        generated_text,
                        other_label=generation_other_label,
                    ),
                    "pe": pe,
                    "generated_text": generated_text,
                    "pred_num": pred_num,
                    "is_numeric": is_numeric,
                    "is_correct": is_correct,
                    "abs_error": abs_error,
                    "signed_error": signed_error,
                    "generated_token_length": generated_len,
                    "source_generated_token_length": source_generated_token_length,
                    "entity_role": entity_role_map.get(source_pos, "OTHER"),
                }
            )

        group_averages = {
            target_pos_resolved: float(mean(lengths)) if lengths else float("nan")
            for target_pos_resolved, lengths in group_lengths.items()
        }
        for row in pending_rows:
            generated_len = int(row["generated_token_length"])
            avg_generated_length = group_averages[row["target_pos_resolved"]]
            row["average_generated_token_length"] = avg_generated_length
            row["generated_token_length_delta_vs_source"] = generated_len - source_generated_token_length
            row["average_generated_token_length_delta_vs_source"] = (
                avg_generated_length - source_generated_token_length
            )
            output_rows.append(row)

    return {"source_tokens": source_tokens, "output_rows": output_rows}


def _parse_sample_idx(name: str) -> int | None:
    match = SAMPLE_DIR_RE.match(name)
    if not match:
        return None
    return int(match.group(1))


def _sort_sweep_files(sample_dir: Path) -> list[Path]:
    files = [path for path in sample_dir.glob("layer_*_pos_*.json") if SWEEP_FILE_RE.match(path.name)]
    return sorted(
        files,
        key=lambda path: (
            int(SWEEP_FILE_RE.match(path.name).group(1)),
            int(SWEEP_FILE_RE.match(path.name).group(2)),
        ),
    )


def _sort_source_pe_files(pe_dir: Path) -> list[Path]:
    files = [path for path in pe_dir.glob("source_*.json") if re.search(r"source_(\d+)\.json$", path.name)]
    return sorted(files, key=lambda path: int(re.search(r"source_(\d+)\.json$", path.name).group(1)))


def _choose_canonical_sweep_file(files: list[Path]) -> Path | None:
    if not files:
        return None
    for path in files:
        match = SWEEP_FILE_RE.match(path.name)
        if match and int(match.group(1)) == 1 and int(match.group(2)) == -1:
            return path
    return files[0]


def _build_source_token_map_from_payload(payload: dict[str, Any]) -> dict[int, str]:
    token_map: dict[int, str] = {}
    for item in payload.get("patch_result") or []:
        if not isinstance(item, dict) or "pos" not in item:
            continue
        token_map[int(item["pos"])] = str(item.get("patching_token", ""))
    return token_map


def _is_control_token(token: str) -> bool:
    return token.startswith("<|") and token.endswith("|>")


def _resolve_target_position(
    requested_target_pos: int,
    req_to_resolved: dict[int, int],
    resolved_positions: Sequence[int],
) -> int:
    if requested_target_pos in req_to_resolved:
        return req_to_resolved[requested_target_pos]
    if requested_target_pos == -1 and resolved_positions:
        return int(resolved_positions[-1])
    return requested_target_pos


def _maybe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _load_pe_sample_metadata(pe_dir: Path, requested_targets: set[int]) -> PESampleMetadata:
    source_files = _sort_source_pe_files(pe_dir)
    if not source_files:
        raise FileNotFoundError(f"No source PE files found in {pe_dir}")

    patched_map: dict[int, str] = {}
    source_target_meta: dict[int, tuple[dict[int, int], list[int]]] = {}
    pe_map: dict[tuple[int, int, int], float] = {}
    cot_text_hint = ""

    for source_path in source_files:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        source_pos = int(payload["source_position"])
        patched_map[source_pos] = str(payload.get("patched_token_str", ""))
        if not cot_text_hint:
            cot_text_hint = str(payload.get("source_generated_answer_cot") or "")

        req_raw = payload.get("target_positions_requested")
        resolved_positions = [int(value) for value in (payload.get("target_positions_resolved") or [])]
        req_to_resolved: dict[int, int] = {}
        if isinstance(req_raw, list) and len(req_raw) == len(resolved_positions):
            for requested, resolved in zip(req_raw, resolved_positions, strict=True):
                if requested is None:
                    continue
                req_to_resolved[int(requested)] = int(resolved)

        source_target_meta[source_pos] = (req_to_resolved, resolved_positions)
        needed_resolved_targets = {
            _resolve_target_position(target_pos, req_to_resolved, resolved_positions)
            for target_pos in requested_targets
        }

        for layer_pe_str, layer_payload in (payload.get("layer_results") or {}).items():
            layer_patch = int(layer_pe_str) + 1
            for item in (layer_payload or {}).get("positions", []):
                target_pos = int(item["target_position"])
                if target_pos not in needed_resolved_targets:
                    continue
                pe_map[(layer_patch, target_pos, source_pos)] = _maybe_float(
                    item.get("pe", item.get("indirect_effect"))
                )

    return PESampleMetadata(
        patched_map=patched_map,
        source_target_meta=source_target_meta,
        pe_map=pe_map,
        cot_text_hint=cot_text_hint,
    )


def _load_eval_items(eval_json: Path) -> list[dict[str, Any]]:
    payload = json.loads(eval_json.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected eval JSON to contain a list, found {type(payload).__name__}")
    return [item if isinstance(item, dict) else {} for item in payload]


def _extract_question_from_eval_item(item: dict[str, Any] | None) -> str:
    if not item:
        return ""
    for key in ("question", "Question", "Input"):
        text = str(item.get(key, "") or "").strip()
        if text:
            return text
    return ""


def _extract_reasoning_from_eval_item(item: dict[str, Any] | None) -> str:
    if not item:
        return ""
    for key in ("Generated Answer_cot", "generated_answer_cot"):
        text = str(item.get(key, "") or "")
        if text:
            return text
    return ""


def _extract_question_from_payload(payload: dict[str, Any]) -> str:
    for key in ("source_prompt", "target_prompt"):
        text = str(payload.get(key, "") or "")
        match = QUESTION_RE.search(text)
        if match:
            return match.group(1).strip()
    return ""


def _decode_patch_token(token: str | None) -> str:
    if token is None:
        return ""
    if token.startswith("<|") and token.endswith("|>"):
        return token
    return token.replace("Ċ", "\n").replace("Ġ", " ").replace("▁", " ")


def _line_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        start = cursor
        end = start + len(line)
        spans.append((start, end, line))
        cursor = end
    if not spans:
        spans.append((0, len(text), text))
    return spans


def _looks_final_answer_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if FINAL_ANSWER_RE.search(stripped):
        return True
    core = ANSWER_PREFIX_RE.sub("", stripped, count=1).strip()
    if FINAL_NUMERIC_RE.match(core):
        return True
    return bool(re.match(r"^[A-Za-z]\w*\s*=\s*[-+]?\d[\d,]*(?:\.\d+)?\s*$", core))


def _classify_lines_setup_step_final(text: str) -> list[str]:
    raw_lines = text.splitlines()
    if not raw_lines:
        return []

    labels = [""] * len(raw_lines)
    nonempty_idx = [idx for idx, line in enumerate(raw_lines) if line.strip()]
    if not nonempty_idx:
        return ["Setup"] * len(raw_lines)

    first_idx = nonempty_idx[0]
    labels[first_idx] = "Setup"
    _ = bool(SETUP_HINT_RE.search(raw_lines[first_idx].strip()))

    if len(nonempty_idx) == 1:
        if _looks_final_answer_line(raw_lines[first_idx]):
            labels[first_idx] = "Final Answer"
    else:
        last_idx = nonempty_idx[-1]
        labels[last_idx] = "Final Answer"
        step_counter = 1
        for idx in nonempty_idx[1:-1]:
            labels[idx] = f"Step {step_counter}"
            step_counter += 1

    current = labels[first_idx] if labels[first_idx] else "Setup"
    for idx, label in enumerate(labels):
        if label:
            current = label
        else:
            labels[idx] = current
    return labels


def _map_source_labels_to_target_line_count(source_labels: list[str], target_count: int) -> list[str]:
    if target_count <= 0:
        return []
    if not source_labels:
        return ["Setup"] * target_count
    if len(source_labels) == target_count:
        return list(source_labels)
    if target_count == 1:
        return [source_labels[0]]
    if len(source_labels) == 1:
        return [source_labels[0]] * target_count

    mapped: list[str] = []
    source_count = len(source_labels)
    for idx in range(target_count):
        source_idx = round(idx * (source_count - 1) / (target_count - 1))
        mapped.append(source_labels[source_idx])
    return mapped


def _enforce_final_answer_last(
    pos_to_step: dict[int, tuple[int, str]],
) -> dict[int, tuple[int, str]]:
    positions = sorted(pos_to_step)
    if not positions:
        return pos_to_step

    original_labels = {pos: label for pos, (_step_id, label) in pos_to_step.items()}
    final_positions = [pos for pos in positions if original_labels[pos] == "Final Answer"]
    if not final_positions:
        return pos_to_step

    tail_positions = set(positions[-len(final_positions) :])
    assigned = dict(original_labels)
    for pos in tail_positions:
        assigned[pos] = "Final Answer"

    for idx, pos in enumerate(positions):
        if pos in tail_positions or original_labels[pos] != "Final Answer":
            continue
        replacement = None
        for prev in reversed(positions[:idx]):
            if assigned[prev] != "Final Answer":
                replacement = assigned[prev]
                break
        assigned[pos] = replacement or "Step 1"

    id_map: dict[str, int] = {}
    next_id = 0
    if any(label == "Setup" for label in assigned.values()):
        id_map["Setup"] = next_id
        next_id += 1

    step_numbers = sorted(
        int(match.group(1))
        for label in set(assigned.values())
        if (match := re.match(r"^Step (\d+)$", label))
    )
    for step_number in step_numbers:
        id_map[f"Step {step_number}"] = next_id
        next_id += 1

    for label in sorted({label for label in assigned.values() if label not in id_map and label != "Final Answer"}):
        id_map[label] = next_id
        next_id += 1

    if any(label == "Final Answer" for label in assigned.values()):
        id_map["Final Answer"] = next_id

    return {pos: (id_map[assigned[pos]], assigned[pos]) for pos in positions}


def _build_step_mapping(
    token_map: dict[int, str],
    cot_text_hint: str | None,
) -> dict[int, tuple[int, str]]:
    if not token_map:
        return {}

    max_pos = max(token_map)
    decoded_tokens = [_decode_patch_token(token_map.get(pos, "")) for pos in range(max_pos + 1)]
    reconstructed = "".join(decoded_tokens)

    spans: list[tuple[int, int, int]] = []
    cursor = 0
    for pos, text in enumerate(decoded_tokens):
        start = cursor
        end = start + len(text)
        spans.append((pos, start, end))
        cursor = end

    target_line_spans = _line_spans(reconstructed)
    source_text = str(cot_text_hint or "").strip()
    if source_text:
        source_labels = _classify_lines_setup_step_final(source_text)
        line_labels = _map_source_labels_to_target_line_count(source_labels, len(target_line_spans))
    else:
        line_labels = _classify_lines_setup_step_final(reconstructed)
        line_labels = _map_source_labels_to_target_line_count(line_labels, len(target_line_spans))

    segments: list[tuple[int, str, int, int]] = []
    segment_id = 0
    for idx, (seg_start, seg_end, _line_text) in enumerate(target_line_spans):
        label = line_labels[idx] if idx < len(line_labels) else "Setup"
        if not segments:
            segments.append((segment_id, label, seg_start, seg_end))
            continue
        prev_id, prev_label, prev_start, prev_end = segments[-1]
        if prev_label == label and prev_end == seg_start:
            segments[-1] = (prev_id, prev_label, prev_start, seg_end)
        else:
            segment_id += 1
            segments.append((segment_id, label, seg_start, seg_end))

    if not segments:
        segments = [(0, "Setup", 0, len(reconstructed))]

    pos_to_step: dict[int, tuple[int, str]] = {}
    for pos, start, end in spans:
        midpoint = start + max(end - start, 1) / 2.0
        chosen = segments[-1]
        for segment in segments:
            _seg_id, _seg_label, seg_start, seg_end = segment
            if seg_start <= midpoint < seg_end:
                chosen = segment
                break
        pos_to_step[pos] = (chosen[0], chosen[1])

    return _enforce_final_answer_last(pos_to_step)


def _step_for_source_pos(
    step_meta: dict[int, tuple[int, str]],
    sorted_step_positions: list[int],
    source_pos: int,
) -> tuple[int, str]:
    if source_pos in step_meta:
        return step_meta[source_pos]
    if not sorted_step_positions:
        return (1, "Step 1")

    fallback = sorted_step_positions[0]
    for pos in sorted_step_positions:
        if pos > source_pos:
            break
        fallback = pos
    return step_meta[fallback]


def _assign_labels_from_spans(
    token_char_spans: list[tuple[int, int, int]],
    spans: Sequence[LabeledSpan],
) -> list[str]:
    labels: list[str] = []
    for _pos, start, end in token_char_spans:
        if start == end:
            labels.append("OTHER")
            continue

        best_label = "OTHER"
        best_priority = PRIORITY[best_label]
        for span in spans:
            if max(start, span.start) < min(end, span.end):
                priority = PRIORITY[span.label]
                if priority > best_priority:
                    best_label = span.label
                    best_priority = priority
        labels.append(best_label)
    return labels


def _assign_labels_from_offsets(
    offsets: Sequence[tuple[int, int]],
    spans: Sequence[LabeledSpan],
) -> list[str]:
    labels: list[str] = []
    for start, end in offsets:
        if start == end:
            labels.append("OTHER")
            continue

        best_label = "OTHER"
        best_priority = PRIORITY[best_label]
        for span in spans:
            if max(start, span.start) < min(end, span.end):
                priority = PRIORITY[span.label]
                if priority > best_priority:
                    best_label = span.label
                    best_priority = priority
        labels.append(best_label)
    return labels


def _tokenize_reasoning_with_offsets(
    reasoning: str,
    tokenizer,
) -> tuple[list[str], list[tuple[int, int]]]:
    encoded = tokenizer(reasoning, add_special_tokens=False, return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    return tokens, offsets


def _reconstruct_text_and_offsets(
    tokens: Sequence[str],
    tokenizer,
) -> tuple[str, list[tuple[int, int]]]:
    offsets: list[tuple[int, int]] = []
    parts: list[str] = []
    cursor = 0
    for token in tokens:
        if _is_control_token(token):
            offsets.append((cursor, cursor))
            continue
        piece = tokenizer.convert_tokens_to_string([token])
        start = cursor
        cursor += len(piece)
        offsets.append((start, cursor))
        parts.append(piece)
    return "".join(parts), offsets


def _build_entity_role_map_for_model(
    token_map: dict[int, str],
    *,
    alignment_model: str,
    question: str,
    reasoning: str,
    tokenizer,
    nlp,
) -> dict[int, str]:
    if alignment_model == "qwen":
        return _build_qwen_entity_role_map(token_map, question=question, nlp=nlp)
    if alignment_model == "llama":
        return _build_llama_entity_role_map(
            token_map,
            question=question,
            reasoning=reasoning,
            tokenizer=tokenizer,
            nlp=nlp,
        )
    raise ValueError(f"Unsupported alignment model: {alignment_model!r}")


def _build_qwen_entity_role_map(token_map: dict[int, str], *, question: str, nlp) -> dict[int, str]:
    if not token_map:
        return {}

    positions = sorted(token_map)
    decoded = [_decode_patch_token(token_map[pos]) for pos in positions]
    reasoning = "".join(decoded)

    token_spans: list[tuple[int, int, int]] = []
    cursor = 0
    for pos, token_text in zip(positions, decoded, strict=True):
        start = cursor
        end = start + len(token_text)
        token_spans.append((pos, start, end))
        cursor = end

    _, _, spans = label_reasoning_with_question(
        question=question,
        reasoning=reasoning,
        nlp=nlp,
    )
    labels = _assign_labels_from_spans(token_spans, spans)

    out: dict[int, str] = {}
    for (pos, _start, _end), label in zip(token_spans, labels, strict=True):
        out[pos] = label if label in VALID_LABELS else "OTHER"
    return out


def _build_llama_entity_role_map(
    token_map: dict[int, str],
    *,
    question: str,
    reasoning: str,
    tokenizer,
    nlp,
) -> dict[int, str]:
    if not token_map:
        return {}

    positions = sorted(token_map)
    canonical_tokens = [token_map[pos] for pos in positions]
    json_tokens, json_offsets = _tokenize_reasoning_with_offsets(reasoning, tokenizer)
    _doc, _unused, spans = label_reasoning_with_question(
        question=question,
        reasoning=reasoning,
        nlp=nlp,
    )

    if canonical_tokens == json_tokens:
        labels = _assign_labels_from_offsets(json_offsets, spans)
    elif canonical_tokens == [*json_tokens, "<|eot_id|>"]:
        labels = [*_assign_labels_from_offsets(json_offsets, spans), "OTHER"]
    else:
        reconstructed_text, canonical_offsets = _reconstruct_text_and_offsets(canonical_tokens, tokenizer)
        _doc, _unused, reconstructed_spans = label_reasoning_with_question(
            question=question,
            reasoning=reconstructed_text,
            nlp=nlp,
        )
        labels = _assign_labels_from_offsets(canonical_offsets, reconstructed_spans)

    out: dict[int, str] = {}
    for pos, label in zip(positions, labels, strict=True):
        out[pos] = label if label in VALID_LABELS else "OTHER"
    return out


def _parse_number(text: object) -> float | None:
    if text is None:
        return None
    matches = NUM_RE.findall(str(text))
    if not matches:
        return None
    raw = matches[-1].replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _count_token_lengths_batch(
    texts: list[object],
    tokenizer,
    cache: dict[str, int],
) -> list[int]:
    normalized = ["" if text is None else str(text) for text in texts]
    out = [0] * len(normalized)
    pending: dict[str, list[int]] = {}

    for idx, text in enumerate(normalized):
        cached = cache.get(text)
        if cached is not None:
            out[idx] = cached
            continue
        pending.setdefault(text, []).append(idx)

    if pending:
        missing_texts = list(pending)
        nonempty_texts = [text for text in missing_texts if text]
        if nonempty_texts:
            encoded = tokenizer(nonempty_texts, add_special_tokens=False)
            for text, input_ids in zip(nonempty_texts, encoded["input_ids"], strict=True):
                cache[text] = len(input_ids)
        for text in missing_texts:
            cache.setdefault(text, 0)
            for idx in pending[text]:
                out[idx] = cache[text]

    return out
