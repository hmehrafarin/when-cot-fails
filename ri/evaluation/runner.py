from __future__ import annotations

import math
from typing import Any

import pandas as pd
from tqdm import tqdm

from ri.common import (
    choose_torch_dtype,
    get_dataset,
    prepare_batch_data,
    set_random_seed,
)
from ri.core.model import ModelAndTokenizer
from ri.prompts.prompter import Prompter
from ri.settings.settings import Constants
from ri.tracking import ExperimentTracker
from ri.utils.extraction import extract_answer

from .config import EvaluationConfig
from .pipeline import generate_batch_outputs


def run_evaluation(
    model_name: str,
    dataset: str,
    prompt_template: str,
    batch_size: int,
    max_gen_len: int,
    seed: int,
    output_file: str,
    extraction_mode: str = "flexible",
    tracker: ExperimentTracker | None = None,
) -> None:
    config = EvaluationConfig(
        batch_size=batch_size,
        max_gen_len=max_gen_len,
        seed=seed,
        extraction_mode=extraction_mode,
    )

    runner = EvaluationRunner(
        model_name=model_name,
        dataset=dataset,
        prompt_template=prompt_template,
        config=config,
    )

    runner.run(output_file, tracker=tracker)


class EvaluationRunner:
    """Run batched evaluation for a dataset."""

    def __init__(
        self,
        model_name: str,
        dataset: str,
        prompt_template: str,
        config: EvaluationConfig,
    ):
        self.config = config
        set_random_seed(config.seed)

        ds = get_dataset(dataset)
        self.data = ds["train"] if dataset.endswith(".json") else ds["test"]

        torch_dtype = choose_torch_dtype(model_name)
        self.mt = ModelAndTokenizer(
            model_name=model_name,
            torch_dtype=torch_dtype,
            cache_dir=Constants.HUGGINGFACE_CACHE_DIR,
        )

        self.prompter = Prompter(template_name=prompt_template)

    def _run_single_batch(self, batch_idx: int):
        batch_questions, batch_answers, batched_input = prepare_batch_data(
            self.data,
            batch_idx,
            self.config.batch_size,
            include_importance=False,
            tokenizer=self.mt.tokenizer,
        )

        numeric_answers = extract_answer(batch_answers)

        generation_results = generate_batch_outputs(
            self.mt,
            self.prompter,
            batched_input,
            self.config,
        )

        return {
            "questions": batch_questions,
            "answers": batch_answers,
            "Generated Answer_num": generation_results["answer_num"],
            "Generated Answer_cot": generation_results["answer_cot"],
            "Input": generation_results["input"],
            "Answer_num": numeric_answers,
        }

    def run(self, output_file: str, tracker: ExperimentTracker | None = None) -> None:
        results: dict[str, list[Any]] = {
            "Input": [],
            "question": [],
            "answer": [],
            "Answer_num": [],
            "Generated Answer_num": [],
            "Generated Answer_cot": [],
        }

        num_batches = math.ceil(len(self.data) / self.config.batch_size)
        for batch_idx in tqdm(range(num_batches), desc="Evaluating"):
            batch_res = self._run_single_batch(batch_idx)
            results["question"].extend(batch_res["questions"])
            results["answer"].extend(batch_res["answers"])
            results["Generated Answer_num"].extend(batch_res["Generated Answer_num"])
            results["Generated Answer_cot"].extend(batch_res["Generated Answer_cot"])
            results["Input"].extend(batch_res["Input"])
            results["Answer_num"].extend(batch_res["Answer_num"])

        df = pd.DataFrame(results)
        df.to_json(output_file, orient="records")

        if tracker is not None and tracker.is_enabled:
            tracker.log_config(self.config)
            tracker.log_dataframe("eval_samples", df)
            n_correct = int(
                (df["Answer_num"].astype(str) == df["Generated Answer_num"].astype(str)).sum()
            )
            tracker.log_metrics(
                {
                    "n_correct": n_correct,
                    "n_samples": len(df),
                    "accuracy": n_correct / max(len(df), 1),
                }
            )
