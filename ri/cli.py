#!/usr/bin/env python3
import argparse

from typing import Optional, Union

from ri.evaluation import run_evaluation
from ri.patching import run_patch
from ri.patching.cma import run_cma


def _parse_steps(value: str) -> Union[int, str, None]:
    """Parse --steps argument: accepts int, 'all', 'no_steps', or None."""
    if value is None or value.lower() == "none":
        return None
    if value.lower() in ("all", "no_steps"):
        return value.lower()
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid steps value: {value!r}. Expected int, 'all', 'no_steps', or 'none'."
        )


def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="ri",
        description="Reasoning interpretability toolkit CLI",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # evaluate subcommand
    ev = sub.add_parser("evaluate", help="Run model generation/evaluation")
    ev.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    ev.add_argument(
        "--dataset", default="gsm8k")
    ev.add_argument("--prompt_template", default="gsm8k_cot")
    ev.add_argument("--steps", type=int, default=None)
    ev.add_argument("--batch_size", type=int, default=16)
    ev.add_argument("--max_gen_len", type=int, default=400)
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--output_file", default="output.json")

    # patch subcommand
    pa = sub.add_parser("patch", help="Run patch-based generation experiments")
    pa.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    pa.add_argument("--source_model_name", default=None)
    pa.add_argument("--target_model_name", default=None)
    pa.add_argument("--source_dataset", default="outputs/single_batch_output_cot.json",
                    help="Source dataset providing hidden states/importance")
    pa.add_argument("--target_dataset", default="outputs/single_batch_output_non_cot.json",
                    help="Target dataset to generate answers for")
    pa.add_argument("--prompt_template", default=None,
                    help="Override both src/tgt prompt templates")
    pa.add_argument("--src_prompt_template", default="gsm8k_cot")
    pa.add_argument("--tgt_prompt_template", default="gsm8k_non_cot")
    pa.add_argument("--gold_step", action="store_true", default=False)
    pa.add_argument("--batch_size", type=int, default=1)
    pa.add_argument("--max_gen_len", type=int, default=400)
    pa.add_argument("--steps", type=_parse_steps, default="no_steps")
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--source_layer", type=int, default=-1,
                    help="Layer index to extract hidden states from (source)")
    pa.add_argument("--target_layer", type=int, default=-1,
                    help="Layer index to patch hidden states into (target)")
    pa.add_argument("--patching_k", type=int, default=1,
                    help="Number of token positions (per sample) to patch",
                    dest="patching_k")
    pa.add_argument("--hs_selection", default=-1,
                    help="Hidden state position: int index, or 'random_k', 'early', 'mid', 'late', 'step_wise', 'every'")
    pa.add_argument("--include_all_tokens", action="store_true", default=False)
    pa.add_argument("--patch_from_generation",
                    action="store_true", default=False)
    pa.add_argument("--patch_position", type=int, default=None)
    pa.add_argument("--output_file", default="output_patched.json")

    # causal mediation analysis subcommand
    cma = sub.add_parser(
        "causal-mediation-analysis",
        aliases=["cma"],
        help="Analyze patch positions and token probabilities",
    )
    cma.add_argument(
        "--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    cma.add_argument("--source_model_name", default=None)
    cma.add_argument("--target_model_name", default=None)
    cma.add_argument("--source_dataset",
                     default="outputs/output_42_no_icl_deterministic.json")
    cma.add_argument("--target_dataset",
                     default="outputs/output_42_no_icl_deterministic.json")
    cma.add_argument(
        "--prompt_template",
        default=None,
        help="Override both src/tgt prompt templates",
    )
    cma.add_argument("--src_prompt_template", default="gsm8k_cot")
    cma.add_argument("--tgt_prompt_template", default="gsm8k_non_cot")
    cma.add_argument("--gold_step", default=False, action="store_true")
    cma.add_argument("--source_layer", type=int, default=25)
    cma.add_argument("--target_layer", type=int, default=25)
    cma.add_argument("--seed", type=int, default=42)
    cma.add_argument("--steps", type=_parse_steps, default="1")
    cma.add_argument("--hs_selection", default=-1,
                     help="Hidden state position: int index, or 'random_k', 'early', 'mid', 'late', 'step_wise', 'every'")
    cma.add_argument("--patching_k", type=int, default=1)
    cma.add_argument("--include_all_tokens",
                     action="store_true", default=False)
    cma.add_argument("--patch_from_generation",
                     action="store_true", default=False)
    cma.add_argument("--max_gen_len", type=int, default=400)
    cma.add_argument("--patch_position", type=int, default=None)
    cma.add_argument("--output_file", default="patch_position_analysis.json")

    args = parser.parse_args()

    if args.cmd == "evaluate":
        run_evaluation(
            model_name=args.model_name,
            dataset=args.dataset,
            prompt_template=args.prompt_template,
            steps=args.steps,
            batch_size=args.batch_size,
            max_gen_len=args.max_gen_len,
            seed=args.seed,
            output_file=args.output_file,
        )

    elif args.cmd == "patch":
        src_template = args.prompt_template or args.src_prompt_template
        tgt_template = args.prompt_template or args.tgt_prompt_template
        resolved_source_model = args.source_model_name or args.model_name
        resolved_target_model = args.target_model_name or resolved_source_model

        run_patch(
            source_model_name=resolved_source_model,
            target_model_name=resolved_target_model,
            source_dataset=args.source_dataset,
            target_dataset=args.target_dataset,
            src_prompt_template=src_template,
            tgt_prompt_template=tgt_template,
            gold_step=args.gold_step,
            batch_size=args.batch_size,
            max_gen_len=args.max_gen_len,
            source_layer=args.source_layer,
            target_layer=args.target_layer,
            seed=args.seed,
            steps=args.steps,
            hs_selection=args.hs_selection,
            patching_k=args.patching_k,
            include_all_tokens=args.include_all_tokens,
            patch_from_generation=args.patch_from_generation,
            patch_position=args.patch_position,
            output_file=args.output_file,
        )
    elif args.cmd in {"causal-mediation-analysis", "cma"}:
        src_template = args.prompt_template or args.src_prompt_template
        tgt_template = args.prompt_template or args.tgt_prompt_template
        resolved_source_model = args.source_model_name or args.model_name
        resolved_target_model = args.target_model_name or resolved_source_model

        run_cma(
            source_model_name=resolved_source_model,
            target_model_name=resolved_target_model,
            source_dataset=args.source_dataset,
            target_dataset=args.target_dataset,
            src_prompt_template=src_template,
            tgt_prompt_template=tgt_template,
            gold_step=args.gold_step,
            source_layer=args.source_layer,
            target_layer=args.target_layer,
            seed=args.seed,
            steps=args.steps,
            hs_selection=args.hs_selection,
            patching_k=args.patching_k,
            include_all_tokens=args.include_all_tokens,
            patch_from_generation=args.patch_from_generation,
            max_gen_len=args.max_gen_len,
            patch_position=args.patch_position,
            output_file=args.output_file,
        )


if __name__ == "__main__":
    cli()
