from __future__ import annotations

import json
from typing import Any

import pandas as pd
from tqdm import tqdm

from ri.patching.config import ExtractionMode, PatchConfig
from ri.patching.runner import PatchRunner
from ri.settings import DEFAULT_MODEL_NAME
from ri.tracking import ExperimentTracker

from . import pipeline as cma_pipeline


class CausalMediationRunner(PatchRunner):
    """Runs causal mediation analysis by patching hidden states sample-by-sample.

    For each sample pair (source, target), patches source hidden states into
    target model at specified layer/position and measures probability changes.
    """

    def __init__(
        self,
        source_model_name: str,
        source_dataset: str,
        target_dataset: str,
        src_prompt_template: str,
        tgt_prompt_template: str,
        patch_from_generation: bool,
        patch_config: PatchConfig,
        seed: int,
        target_model_name: str | None = None,
    ):
        super().__init__(
            source_model_name=source_model_name,
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            src_prompt_template=src_prompt_template,
            tgt_prompt_template=tgt_prompt_template,
            batch_size=1,
            patch_from_generation=patch_from_generation,
            patch_config=patch_config,
            seed=seed,
            target_model_name=target_model_name,
        )
        self.config = patch_config

    def _run_single_sample(self, sample_idx: int) -> dict[str, Any]:
        source_sample = self.source_data[sample_idx]
        target_sample = self.target_data[sample_idx]

        batched_input_source = cma_pipeline.make_batched_input(
            source_sample, include_generated=True
        )
        batched_input_tgt = cma_pipeline.make_batched_input(target_sample, include_generated=False)

        return cma_pipeline.analyze_patch_positions(
            source_mt=self.source_mt,
            target_mt=self.target_mt,
            src_prompter=self.src_prompter,
            tgt_prompter=self.tgt_prompter,
            batched_input_source=batched_input_source,
            batched_input_tgt=batched_input_tgt,
            config=self.config,
            patch_from_generation=self.patch_from_generation,
            sample_idx=sample_idx,
            batch_size=self.batch_size,
        )

    def run(
        self,
        output_file: str = "patch_position_analysis.json",
        tracker: ExperimentTracker | None = None,
    ) -> None:
        total_samples = min(len(self.source_data), len(self.target_data))
        all_results = []

        for idx in tqdm(range(total_samples), desc="Analyzing samples"):
            try:
                result = self._run_single_sample(idx)
                all_results.append(result)
            except Exception as e:
                print(f"Error analyzing sample {idx}: {e}")
                all_results.append({"sample_idx": idx, "error": str(e)})

        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Results saved to {output_file}")

        if tracker is not None and tracker.is_enabled:
            tracker.log_config(self.config)
            tracker.log_dataframe("cma_results", pd.DataFrame(all_results))
            tracker.log_metrics(
                {
                    "n_samples": total_samples,
                    "source_layer": self.config.source_layer,
                    "target_layer": self.config.target_layer,
                }
            )


class PatchPositionAnalyzer(CausalMediationRunner):
    """Backward compatible alias."""


def run_cma(
    *,
    source_model_name: str | None = None,
    target_model_name: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    source_dataset: str = "outputs/output_42_no_icl_deterministic.json",
    target_dataset: str = "outputs/output_42_no_icl_deterministic.json",
    src_prompt_template: str = "gsm8k_cot",
    tgt_prompt_template: str = "gsm8k_non_cot",
    source_layer: int = 25,
    target_layer: int = 25,
    seed: int = 42,
    hs_selection: int = -1,
    include_all_tokens: bool = False,
    patch_from_generation: bool = False,
    max_gen_len: int = 400,
    patch_position: int | None = None,
    gen_cache_dir: str | None = None,
    extraction_mode: ExtractionMode = "flexible",
    output_file: str = "patch_position_analysis.json",
    tracker: ExperimentTracker | None = None,
) -> None:
    resolved_source_model = source_model_name or model_name
    resolved_target_model = target_model_name or resolved_source_model

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

    runner = CausalMediationRunner(
        source_model_name=resolved_source_model,
        target_model_name=resolved_target_model,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        src_prompt_template=src_prompt_template,
        tgt_prompt_template=tgt_prompt_template,
        patch_from_generation=patch_from_generation,
        patch_config=patch_config,
        seed=seed,
    )

    runner.run(output_file=output_file, tracker=tracker)
