from __future__ import annotations

import math
from typing import Any

import pandas as pd
from tqdm import tqdm

from ri.common import choose_torch_dtype, get_dataset, prepare_batch_data, set_random_seed
from ri.core.model import ModelAndTokenizer
from ri.prompts.prompter import Prompter
from ri.settings.settings import Constants
from ri.tracking import ExperimentTracker
from ri.utils.extraction import extract_answer

from .config import ExtractionMode, PatchConfig
from .pipeline import patch_and_generate


class PatchRunner:
    """Orchestrates dataset loading, model execution with patching, and output serialization."""

    def __init__(
        self,
        source_model_name: str,
        source_dataset: str,
        target_dataset: str,
        src_prompt_template: str,
        tgt_prompt_template: str,
        batch_size: int,
        patch_from_generation: bool,
        patch_config: PatchConfig,
        seed: int,
        target_model_name: str | None = None,
        cache_dir: str = Constants.HUGGINGFACE_CACHE_DIR,
    ):
        set_random_seed(seed, deterministic=True)

        source_ds = get_dataset(source_dataset)
        self.source_data = (
            source_ds["train"] if source_dataset.endswith(".json") else source_ds["test"]
        )

        target_ds = get_dataset(target_dataset)
        self.target_data = (
            target_ds["train"] if target_dataset.endswith(".json") else target_ds["test"]
        )

        self.batch_size = batch_size
        self.patch_config = patch_config
        self.patch_from_generation = patch_from_generation

        target_model_name = target_model_name or source_model_name

        if target_model_name == source_model_name:
            torch_dtype = choose_torch_dtype(source_model_name)
            shared_mt = ModelAndTokenizer(
                model_name=source_model_name,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
            )
            self.source_mt = shared_mt
            self.target_mt = shared_mt
        else:
            source_dtype = choose_torch_dtype(source_model_name)
            target_dtype = choose_torch_dtype(target_model_name)
            self.source_mt = ModelAndTokenizer(
                model_name=source_model_name,
                torch_dtype=source_dtype,
                cache_dir=cache_dir,
            )
            self.target_mt = ModelAndTokenizer(
                model_name=target_model_name,
                torch_dtype=target_dtype,
                cache_dir=cache_dir,
            )

        self.src_prompter = Prompter(template_name=src_prompt_template)
        self.tgt_prompter = Prompter(template_name=tgt_prompt_template)

    def _run_single_batch(self, batch_idx: int) -> dict[str, list]:
        #!TODO: refactor to drop the token importance, we will remove it in the next update
        _q, a, batched_input_source = prepare_batch_data(
            self.source_data,
            batch_idx,
            self.batch_size,
            include_importance=False,
            tokenizer=self.source_mt.tokenizer,
        )

        q_tgt, a_tgt, batched_input_tgt = prepare_batch_data(
            self.target_data,
            batch_idx,
            self.batch_size,
            include_importance=False,
            tokenizer=self.target_mt.tokenizer,
        )

        extract_answer(a)
        numeric_answers = extract_answer(a_tgt)

        gen = patch_and_generate(
            self.source_mt,
            self.target_mt,
            self.src_prompter,
            self.tgt_prompter,
            patch_from_generation=self.patch_from_generation,
            batched_input_source=batched_input_source,
            batched_input_tgt=batched_input_tgt,
            config=self.patch_config,
            batch_size=self.batch_size,
        )

        # Convert token lists to text for presentation
        source_tokens = gen.get("source_selected_tokens", [])
        target_tokens = gen.get("target_patch_token", [])
        patch_from_text = [" ".join(t) if isinstance(t, list) else str(t) for t in source_tokens]
        patch_to_text = [" ".join(t) if isinstance(t, list) else str(t) for t in target_tokens]

        result = {
            "question": q_tgt,
            "answer": a_tgt,
            "Answer_num": numeric_answers,
            "Generated Answer_num": gen["answer_num"],
            "Generated Answer_cot": gen["answer_cot"],
            "patch_from": patch_from_text,
            "patch_to": patch_to_text,
            "source_prompt": gen.get("source_prompt", []),
            "target_prompt": gen.get("target_prompt", []),
        }
        if self.patch_from_generation:
            result["source_generated_answer"] = gen.get("source_generated_answer", [])
        return result

    def run(self, output_file: str, tracker: ExperimentTracker | None = None) -> None:
        results: dict[str, list[Any]] = {
            "question": [],
            "answer": [],
            "Answer_num": [],
            "Generated Answer_num": [],
            "Generated Answer_cot": [],
            "patch_from": [],
            "patch_to": [],
            "source_prompt": [],
            "target_prompt": [],
        }
        if self.patch_from_generation:
            results["source_generated_answer"] = []

        total_samples = min(len(self.source_data), len(self.target_data))
        num_batches = math.ceil(total_samples / self.batch_size)

        with tqdm(total=total_samples, desc="Processing samples", unit="sample") as pbar:
            for idx in range(num_batches):
                batch_res = self._run_single_batch(idx)
                for k, v in results.items():
                    v.extend(batch_res.get(k, []))

                processed = len(batch_res.get("question", []))
                pbar.update(processed if processed > 0 else self.batch_size)
                pbar.set_postfix(batch=f"{idx + 1}/{num_batches}")

        df = pd.DataFrame(results)
        df.to_json(output_file, orient="records")

        if tracker is not None and tracker.is_enabled:
            tracker.log_config(self.patch_config)
            tracker.log_dataframe("patch_samples", df)
            n_correct = int(
                (df["Answer_num"].astype(str) == df["Generated Answer_num"].astype(str)).sum()
            )
            tracker.log_metrics(
                {
                    "n_correct": n_correct,
                    "n_samples": len(df),
                    "accuracy": n_correct / max(len(df), 1),
                    "source_layer": self.patch_config.source_layer,
                    "target_layer": self.patch_config.target_layer,
                }
            )


def run_patch(
    *,
    source_model_name: str,
    target_model_name: str | None = None,
    source_dataset: str,
    target_dataset: str,
    src_prompt_template: str,
    tgt_prompt_template: str,
    batch_size: int,
    max_gen_len: int,
    source_layer: int,
    target_layer: int,
    patch_position: int | None,
    seed: int,
    patch_from_generation: bool,
    hs_selection: int,
    include_all_tokens: bool = False,
    gen_cache_dir: str | None = None,
    extraction_mode: ExtractionMode = "flexible",
    output_file: str,
    tracker: ExperimentTracker | None = None,
) -> None:
    patch_config = PatchConfig(
        max_gen_len=max_gen_len,
        source_layer=source_layer,
        target_layer=target_layer,
        patch_position=patch_position,
        hs_selection=hs_selection,
        include_all_tokens=include_all_tokens,
        gen_cache_dir=gen_cache_dir,
        extraction_mode=extraction_mode,
    )

    runner = PatchRunner(
        source_model_name=source_model_name,
        target_model_name=target_model_name,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        src_prompt_template=src_prompt_template,
        patch_from_generation=patch_from_generation,
        tgt_prompt_template=tgt_prompt_template,
        batch_size=batch_size,
        patch_config=patch_config,
        seed=seed,
    )

    runner.run(output_file, tracker=tracker)
