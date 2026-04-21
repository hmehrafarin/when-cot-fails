from __future__ import annotations

from ri.common.prompts import build_prompt_batch
from ri.utils.extraction import extract_answer_from_generation
from ri.utils.tokenizer import (
    decode_tokens,
    get_eos_token_ids,
    get_pad_id,
    make_inputs,
)

from .config import EvaluationConfig


def generate_batch_outputs(
    mt,
    prompter,
    batched_input,
    config: EvaluationConfig,
) -> dict[str, list[str | None]]:
    """
    Generate answers for a batch of prompts and extract numeric / CoT outputs.
    """
    convos = build_prompt_batch(prompter, batched_input)

    supports_system_prompt = bool(getattr(mt, "is_instruct_model", False))

    tokenized_inp = make_inputs(
        mt.tokenizer,
        convos,
        mt.device,
        system_prompt=supports_system_prompt,
        add_generation_prompt=supports_system_prompt,
    )

    pad_id = get_pad_id(mt.tokenizer)

    eos_ids = get_eos_token_ids(mt.tokenizer)

    output_toks = mt.model.generate(
        **tokenized_inp,
        max_new_tokens=config.max_gen_len,
        eos_token_id=eos_ids,
        pad_token_id=pad_id,
        do_sample=False,
    )

    output = decode_tokens(mt.tokenizer, output_toks)

    extracted_answers = extract_answer_from_generation(
        output,
        tokenizer=mt.tokenizer,
        template_name=getattr(prompter, "template_name", None),
    )

    rendered_input: list[str | None] = [convo[-1]["content"] if convo else "" for convo in convos]

    return {
        "input": rendered_input,
        "answer_num": extracted_answers["answer_num"],
        "answer_cot": extracted_answers["answer_text"],
    }
