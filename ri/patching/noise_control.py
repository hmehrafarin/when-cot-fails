"""Direction-noise control (Appendix C).

Replays patches that recovered the correct answer with the patched source hidden state
rotated to a fixed cosine similarity ``alpha`` to the original (norm preserved), and records
whether the target run still produces the correct final answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd
from tqdm import tqdm

from ri.patching.config import ExtractionMode, PatchConfig
from ri.patching.runner import PatchRunner
from ri.utils.extraction import parse_number, score_prediction

REQUIRED_COLUMNS = ("sample_idx", "layer", "target_pos", "source_pos", "is_correct")
OPTIONAL_COLUMNS = ("patched_token_str",)
KEY_FIELDS = ("sample_idx", "layer", "target_pos", "source_pos", "alpha")
DEFAULT_ALPHAS = (1.0, 0.75, 0.5, 0.25, 0.0)


def stable_patch_seed(
    seed: int, sample_idx: int, layer: int, target_pos: int, source_pos: int
) -> int:
    """Deterministic per-patch RNG seed, identical across runs and machines."""
    key = f"{seed}:{sample_idx}:{layer}:{target_pos}:{source_pos}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") >> 1


def load_successful_patches(results_csv: str, target_pos: int | None = None) -> pd.DataFrame:
    """Rows of a ``full_results`` CSV whose patch recovered the correct answer."""
    wanted = {*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS}
    df = pd.read_csv(results_csv, usecols=lambda column: column in wanted)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{results_csv} lacks required columns {missing}")

    is_correct = df["is_correct"]
    if is_correct.dtype != bool:
        is_correct = is_correct.astype(str).str.strip().str.lower().eq("true")
    df = df[is_correct]
    if target_pos is not None:
        df = df[df["target_pos"] == target_pos]

    if "patched_token_str" not in df.columns:
        df = df.assign(patched_token_str="")
    df = df.assign(patched_token_str=df["patched_token_str"].fillna("").astype(str))
    df = df.astype({"sample_idx": int, "layer": int, "target_pos": int, "source_pos": int})
    return df.sort_values(["sample_idx", "layer", "target_pos", "source_pos"]).reset_index(
        drop=True
    )


def select_examples(
    available: Iterable[int],
    n_examples: int,
    seed: int,
    explicit: Sequence[int] | None = None,
) -> list[int]:
    """Choose which examples to replay: an explicit list, or a seeded random sample."""
    pool = sorted({int(value) for value in available})
    if explicit is not None:
        chosen = sorted({int(value) for value in explicit})
        missing = sorted(set(chosen) - set(pool))
        if missing:
            raise ValueError(f"Examples {missing} have no successful patches in the results CSV.")
        return chosen
    if n_examples <= 0:
        raise ValueError("n_examples must be positive")
    if n_examples >= len(pool):
        return pool
    return sorted(random.Random(seed).sample(pool, n_examples))


def _sample_output_path(output_dir: str, sample_idx: int) -> str:
    return os.path.join(output_dir, f"sample_{sample_idx}.jsonl")


def _read_records(path: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                records.append(json.loads(text))
            except json.JSONDecodeError:
                continue  # partially written last line of an interrupted run
    return records


def _record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record[field] for field in KEY_FIELDS)


def summarize(output_dir: str) -> pd.DataFrame:
    """Aggregate every ``sample_*.jsonl`` in ``output_dir`` into the Table 8 summary.

    Writes ``summary.csv`` with, per ``alpha``, the number of examples and patches and the
    percentage of patches that still recover the correct final answer.
    """
    records: list[dict[str, Any]] = []
    for name in sorted(os.listdir(output_dir)):
        if name.startswith("sample_") and name.endswith(".jsonl"):
            records.extend(_read_records(os.path.join(output_dir, name)))
    if not records:
        raise FileNotFoundError(f"No sample_*.jsonl results found in {output_dir}")

    df = pd.DataFrame(records)
    summary = (
        df.groupby("alpha")
        .agg(
            n_examples=("sample_idx", "nunique"),
            n_patches=("is_correct", "size"),
            n_correct=("is_correct", "sum"),
        )
        .sort_index(ascending=False)
        .reset_index()
    )
    summary["success_rate"] = 100.0 * summary["n_correct"] / summary["n_patches"]
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    return summary


def _print_summary(summary: pd.DataFrame) -> None:
    print("Direction-noise control: patches still recovering the correct answer")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def _make_runner(
    *,
    patch_config: PatchConfig,
    seed: int,
    source_model_name: str,
    target_model_name: str | None,
    source_dataset: str,
    target_dataset: str,
    src_prompt_template: str,
    tgt_prompt_template: str,
) -> PatchRunner:
    return PatchRunner(
        source_model_name=source_model_name,
        target_model_name=target_model_name,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        src_prompt_template=src_prompt_template,
        tgt_prompt_template=tgt_prompt_template,
        patch_from_generation=True,
        patch_config=patch_config,
        seed=seed,
        batch_size=1,
    )


def run_noise_control(
    *,
    results_csv: str,
    output_dir: str,
    source_model_name: str,
    source_dataset: str,
    target_dataset: str,
    target_model_name: str | None = None,
    src_prompt_template: str = "gsm8k_cot",
    tgt_prompt_template: str = "gsm8k_non_cot",
    n_examples: int = 100,
    sample_indices: Sequence[int] | None = None,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    target_pos: int | None = None,
    max_gen_len: int = 400,
    source_max_gen_len: int | None = None,
    gen_cache_dir: str | None = None,
    extraction_mode: ExtractionMode = "flexible",
    resume: bool = False,
    summarize_only: bool = False,
    seed: int = 42,
) -> None:
    """Replay successful patches with direction noise and summarise recovery per ``alpha``.

    ``results_csv`` is a ``full_results`` table from ``task=full_results``; ``max_gen_len``,
    ``source_max_gen_len`` and ``extraction_mode`` must match the sweep that produced it so the
    replayed source CoT is the one whose patches succeeded. The random direction of each patch
    is derived from ``seed`` and the patch coordinates and shared across ``alphas``, so the
    levels form a paired comparison. Results are written per example to
    ``<output_dir>/sample_<idx>.jsonl`` (append-only, so ``resume`` skips finished pairs and
    ``sample_indices`` shards the run across jobs) and aggregated into ``summary.csv``.
    """
    os.makedirs(output_dir, exist_ok=True)
    if summarize_only:
        _print_summary(summarize(output_dir))
        return

    alpha_levels = [float(alpha) for alpha in alphas]
    if not alpha_levels or any(not 0.0 <= alpha <= 1.0 for alpha in alpha_levels):
        raise ValueError(f"alphas must be a non-empty list of values in [0, 1], got {alphas!r}")
    if len(set(alpha_levels)) != len(alpha_levels):
        raise ValueError(f"alphas must be unique, got {alphas!r}")

    patches = load_successful_patches(results_csv, target_pos=target_pos)
    if patches.empty:
        raise ValueError(f"No successful patches found in {results_csv}")
    examples = select_examples(patches["sample_idx"], n_examples, seed, sample_indices)
    patches = patches[patches["sample_idx"].isin(examples)]
    print(
        f"Replaying {len(patches)} successful patches from {len(examples)} examples at "
        f"{len(alpha_levels)} alpha levels ({len(patches) * len(alpha_levels)} generations)."
    )

    cache_dir = gen_cache_dir or os.path.join(output_dir, "gen_cache")
    patch_config = PatchConfig(
        max_gen_len=max_gen_len,
        source_max_gen_len=source_max_gen_len,
        source_layer=0,
        target_layer=0,
        hs_selection=0,
        include_all_tokens=True,
        gen_cache_dir=cache_dir,
        extraction_mode=extraction_mode,
    )
    runner = _make_runner(
        patch_config=patch_config,
        seed=seed,
        source_model_name=source_model_name,
        target_model_name=target_model_name,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        src_prompt_template=src_prompt_template,
        tgt_prompt_template=tgt_prompt_template,
    )

    metadata = {
        "results_csv": results_csv,
        "examples": examples,
        "alphas": alpha_levels,
        "target_pos": target_pos,
        "seed": seed,
        "source_model_name": source_model_name,
        "target_model_name": target_model_name or source_model_name,
        "source_dataset": source_dataset,
        "target_dataset": target_dataset,
        "max_gen_len": max_gen_len,
        "source_max_gen_len": source_max_gen_len,
        "extraction_mode": extraction_mode,
    }
    metadata_path = os.path.join(output_dir, f"metadata_{examples[0]}-{examples[-1]}.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    for sample_idx in tqdm(examples, desc="Examples"):
        sample_patches = patches[patches["sample_idx"] == sample_idx]
        out_path = _sample_output_path(output_dir, sample_idx)
        done = {_record_key(record) for record in _read_records(out_path)} if resume else set()
        if not resume and os.path.exists(out_path):
            os.remove(out_path)

        with open(out_path, "a", encoding="utf-8") as handle:
            for row in tqdm(
                list(sample_patches.itertuples(index=False)),
                desc=f"Sample {sample_idx}",
                leave=False,
            ):
                layer, tgt_pos, src_pos = int(row.layer), int(row.target_pos), int(row.source_pos)
                cfg = runner.patch_config
                cfg.source_layer = layer - 1  # sweep files and the CSV number layers from 1
                cfg.target_layer = layer - 1
                cfg.patch_position = tgt_pos
                cfg.hs_selection = src_pos
                cfg.perturb_seed = stable_patch_seed(seed, sample_idx, layer, tgt_pos, src_pos)

                for alpha in alpha_levels:
                    if (sample_idx, layer, tgt_pos, src_pos, alpha) in done:
                        continue
                    cfg.perturb_cosine = None if alpha >= 1.0 else alpha
                    batch_res = runner._run_single_batch(sample_idx)
                    generated_text = str(batch_res["Generated Answer_cot"][0])
                    score = score_prediction(generated_text, parse_number(batch_res["answer"][0]))
                    record = {
                        "sample_idx": sample_idx,
                        "layer": layer,
                        "target_pos": tgt_pos,
                        "source_pos": src_pos,
                        "alpha": alpha,
                        "expected_token": str(row.patched_token_str),
                        "patched_token": str(batch_res["patch_from"][0]),
                        "generated_text": generated_text,
                        "pred_num": score.pred_num,
                        "is_correct": score.is_correct,
                    }
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()

    _print_summary(summarize(output_dir))
