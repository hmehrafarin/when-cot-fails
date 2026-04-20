from __future__ import annotations

import json
from typing import Any

from tqdm import tqdm

from ri.patching.config import HSSelectionMode, PatchConfig, StepsType
from ri.patching.runner import PatchRunner
from ri.settings import DEFAULT_MODEL_NAME

from . import pipeline as cma_pipeline


def coerce_steps(value: int | str | None) -> StepsType:
    """Convert steps argument to StepsType."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    lowered = value.strip().lower()
    if lowered == "all":
        return "all"
    if lowered == "no_steps":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(
            f"--steps must be an integer, 'all', or 'no_steps', got {value!r}"
        ) from exc


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
        gold_step: bool,
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
            gold_step=gold_step,
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
            gold_step=self.gold_step,
            patch_from_generation=self.patch_from_generation,
            sample_idx=sample_idx,
            batch_size=self.batch_size,
        )

    def run(
        self,
        output_file: str = "patch_position_analysis.json",
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
    gold_step: bool = True,
    source_layer: int = 25,
    target_layer: int = 25,
    seed: int = 42,
    steps: int | str | None = 1,
    hs_selection: HSSelectionMode = -1,
    patching_k: int = 1,
    include_all_tokens: bool = False,
    patch_from_generation: bool = False,
    max_gen_len: int = 400,
    patch_position: int | None = None,
    output_file: str = "patch_position_analysis.json",
) -> None:
    resolved_source_model = source_model_name or model_name
    resolved_target_model = target_model_name or resolved_source_model
    parsed_steps = coerce_steps(steps)

    patch_config = PatchConfig(
        max_gen_len=max_gen_len,
        source_layer=source_layer,
        target_layer=target_layer,
        patch_position=patch_position,
        steps=parsed_steps,
        hs_selection=hs_selection,
        patching_k=patching_k,
        include_all_tokens=include_all_tokens,
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
        gold_step=gold_step,
    )

    runner.run(output_file=output_file)


def main(**kwargs):
    run_cma(**kwargs)
