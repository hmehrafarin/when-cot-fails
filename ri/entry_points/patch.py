import argparse
from typing import Optional

from ri.patching.runner import run_patch


def _str_to_bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {v!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single patch experiment")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Fallback model name when --source_model_name is not set")
    parser.add_argument("--source_model_name", type=str, default=None)
    parser.add_argument("--target_model_name", type=str, default=None)
    parser.add_argument("--source_dataset", type=str, required=True)
    parser.add_argument("--target_dataset", type=str, required=True)
    parser.add_argument("--src_prompt_template", type=str, default="gsm8k_cot")
    parser.add_argument("--tgt_prompt_template", type=str, default="gsm8k_cot")
    parser.add_argument("--gold_step", type=_str_to_bool, default=False)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hs_selection", type=str, default="-1")
    parser.add_argument("--max_gen_len", type=int, default=400)
    parser.add_argument("--steps", type=str, default="1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch_from_generation", type=_str_to_bool, default=False)
    parser.add_argument("--gen_cache_dir", type=str, default=None)
    parser.add_argument("--patch_position", type=int, default=None)
    parser.add_argument("--source_layer", type=int, required=True)
    parser.add_argument("--target_layer", type=int, required=True)
    parser.add_argument("--patching_k", type=int, default=1)
    parser.add_argument("--include_all_tokens", type=_str_to_bool, default=False)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    source_model_name: Optional[str] = args.source_model_name or args.model_name
    if not source_model_name:
        parser.error("Provide --source_model_name or --model_name")

    run_patch(
        source_model_name=source_model_name,
        target_model_name=args.target_model_name,
        source_dataset=args.source_dataset,
        target_dataset=args.target_dataset,
        src_prompt_template=args.src_prompt_template,
        tgt_prompt_template=args.tgt_prompt_template,
        gold_step=args.gold_step,
        batch_size=args.batch_size,
        max_gen_len=args.max_gen_len,
        source_layer=args.source_layer,
        target_layer=args.target_layer,
        patch_position=args.patch_position,
        seed=args.seed,
        patch_from_generation=args.patch_from_generation,
        steps=args.steps,
        hs_selection=args.hs_selection,
        patching_k=args.patching_k,
        include_all_tokens=args.include_all_tokens,
        gen_cache_dir=args.gen_cache_dir,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
