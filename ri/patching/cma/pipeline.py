from __future__ import annotations

import hashlib
import os
from typing import Any

import torch
import torch.nn.functional as F

from ri.common.prompts import build_prompt_batch
from ri.core.hooks import remove_hooks, set_patch
from ri.patching.config import HSSelectionMode, PatchConfig, StepsType
from ri.patching.pipeline import _resolve_hidden_state_index, _safe_token_from_id
from ri.patching.selectors import select_positions_with_mode, select_step_positions
from ri.patching.tensor_ops import (
    compute_core_token_positions,
    left_pad_offsets,
    mask_to_positions,
)
from ri.utils.extraction import extract_answer, extract_answer_from_generation
from ri.utils.text import prompt_text_from_rendered
from ri.utils.tokenizer import (
    decode_tokens,
    get_eos_token_ids,
    get_pad_id,
    make_inputs,
    render_prompts,
)


def normalize_hs_selections(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [value]


def extract_question_answer(sample: dict[str, Any]) -> dict[str, str]:
    return {
        "question": sample.get("question", sample.get("Question", "")),
        "answer": sample.get("answer", sample.get("Answer", "")),
    }


def make_batched_input(sample: dict[str, Any], *, include_generated: bool) -> list[dict[str, Any]]:
    payload: dict[str, Any] = extract_question_answer(sample)
    if include_generated:
        payload["Generated Answer_num"] = sample.get("Generated Answer_num", "")
    if "hs_selection" in sample:
        payload["hs_selection"] = sample["hs_selection"]
    return [payload]


def build_prompt_inputs(
    tokenizer,
    prompter,
    batched_input: list[dict[str, Any]],
    *,
    steps: StepsType,
    device,
    system_prompt: bool,
    add_generation_prompt: bool,
) -> tuple[list[Any], list[str], list[str], dict[str, torch.Tensor]]:
    convos = build_prompt_batch(
        prompter,
        batched_input,
        steps=steps,
    )

    rendered_prompts = render_prompts(
        tokenizer,
        convos,
        system_prompt=system_prompt,
        add_generation_prompt=add_generation_prompt,
    )

    prompt_texts = [prompt_text_from_rendered(p) for p in convos]

    tokenized = make_inputs(
        tokenizer,
        convos,
        device,
        system_prompt=system_prompt,
        add_generation_prompt=add_generation_prompt,
        rendered_prompts=rendered_prompts,
    )
    return convos, rendered_prompts, prompt_texts, tokenized


def resolve_target_steps(
    steps: StepsType,
    template_name: str,
    gold_step: bool,
) -> StepsType:
    if "non_cot" in template_name.lower() or not gold_step:
        return None
    if isinstance(steps, int) and steps > 1:
        return steps - 1
    if steps == "all":
        return "all"
    return None


def select_candidates_from_prompt(
    tokenized_source: dict[str, torch.Tensor],
    prompt_texts: list[str],
    tokenizer,
    *,
    steps: StepsType,
    include_all_tokens: bool,
) -> list[int]:
    source_prompt_positions = [mask_to_positions(am) for am in tokenized_source["attention_mask"]]

    source_core_positions, _ = compute_core_token_positions(
        tokenized_source,
        prompt_texts,
        tokenizer,
    )

    step_positions: list[list[int]]
    if steps in (None, "no_steps"):
        step_positions = [[]]
    else:
        step_positions = [
            select_step_positions(
                source_core_positions[0] if source_core_positions else [],
                prompt_texts[0] if prompt_texts else "",
                tokenizer,
                steps,
            )
        ]

    if include_all_tokens:
        return list(source_prompt_positions[0])

    candidates = step_positions[0] if step_positions[0] else []
    if not candidates and source_core_positions:
        candidates = list(source_core_positions[0])
    if not candidates:
        candidates = list(source_prompt_positions[0])
    return candidates


def encode_answer_tokens(tokenizer, answer_value: Any | None) -> tuple[str | None, list[int]]:
    if answer_value is None:
        return None, []
    text = str(answer_value).strip()
    if not text:
        return text, []
    return text, tokenizer.encode(text, add_special_tokens=False)


def resolve_patch_position(
    patch_position: int | None,
    offset: int,
    valid_len: int,
) -> int:
    if patch_position is None:
        return offset + valid_len - 1
    if patch_position < 0:
        return max(offset, offset + valid_len + int(patch_position))
    pos = min(offset + int(patch_position), offset + valid_len - 1)
    return max(pos, offset)


def compute_gold_label_probability(
    logits: torch.Tensor,
    gold_token_ids: list[int],
    position: int = -1,
) -> dict[str, Any]:
    """Compute probability of gold tokens at given position."""
    # logits: (batch, seq_len, vocab_size)
    pos_logits = logits[:, position, :]  # (batch, vocab_size)
    probs = F.softmax(pos_logits, dim=-1)  # (batch, vocab_size)
    log_probs = F.log_softmax(pos_logits, dim=-1)

    per_token_probs = [probs[0, tid].item() for tid in gold_token_ids]
    per_token_log_probs = [log_probs[0, tid].item() for tid in gold_token_ids]

    return {
        "sum_prob": sum(per_token_probs),
        "sum_log_prob": sum(per_token_log_probs),
        "per_token_probs": per_token_probs,
        "per_token_log_probs": per_token_log_probs,
        "gold_token_ids": gold_token_ids,
    }


def positions_from_generation_mode(
    mode: HSSelectionMode,
    generated_token_ids: list[list[int]] | None,
    tokenizer,
) -> list[int]:
    if not generated_token_ids or not isinstance(mode, str):
        return []

    tokens = generated_token_ids[0]
    if mode == "step_wise":
        positions: list[int] = []
        for i, tid in enumerate(tokens):
            if "\n" in tokenizer.decode([tid]) and i > 0:
                positions.append(i)
        return list(dict.fromkeys(positions))
    if mode == "every":
        return list(range(len(tokens)))
    return []


def resolve_source_positions(
    *,
    candidates: list[int],
    generated_token_ids: list[list[int]] | None,
    sample_hs_selections: list[Any] | None,
    hs_selection: HSSelectionMode,
    patching_k: int | None,
    patch_from_generation: bool,
    tokenizer,
) -> dict[Any, list[int]]:
    positions_to_select = patching_k or 1
    positions_by_mode: dict[Any, list[int]] = {}

    if sample_hs_selections is not None:
        for item in sample_hs_selections:
            mode = int(item) if isinstance(item, str) and item.isdigit() else item
            positions = select_positions_with_mode(
                candidates,
                positions_to_select,
                mode,
            )
            if not positions:
                fill = int(candidates[-1]) if candidates else 0
                positions = [fill]
            positions_by_mode[item] = positions
        return positions_by_mode

    positions = (
        positions_from_generation_mode(
            hs_selection,
            generated_token_ids,
            tokenizer,
        )
        if patch_from_generation
        else []
    )
    if not positions:
        positions = select_positions_with_mode(
            candidates,
            positions_to_select,
            hs_selection,
        )
    if not positions:
        fill = int(candidates[-1]) if candidates else 0
        positions = [fill]
    positions_by_mode[hs_selection] = positions
    return positions_by_mode


def get_source_hidden_state_at_position(
    *,
    patch_from_generation: bool,
    all_source_hs: torch.Tensor,  # (batch, seq_len, hidden_dim)
    hs_pos: int,
    tokenized_source: dict[str, torch.Tensor],
    generated_token_ids: list[list[int]] | None,
    tokenizer,
) -> tuple[torch.Tensor, int, str]:
    """Extract hidden state at position, returns (hs, safe_pos, token_text)."""
    if patch_from_generation:
        max_gen_pos = all_source_hs.size(1) - 1
        safe_pos = (
            min(hs_pos, max_gen_pos) if hs_pos >= 0 else max(0, all_source_hs.size(1) + hs_pos)
        )
        # (batch, 1, hidden_dim)
        hs_at_pos = all_source_hs[:, safe_pos : safe_pos + 1, :]
        if generated_token_ids and safe_pos < len(generated_token_ids[0]):
            token_text = _safe_token_from_id(tokenizer, generated_token_ids[0][safe_pos])
        else:
            token_text = ""
        return hs_at_pos, safe_pos, token_text

    seq_len = all_source_hs.size(1)
    safe_pos = min(hs_pos, seq_len - 1) if hs_pos >= 0 else max(0, seq_len + hs_pos)
    hs_at_pos = all_source_hs[:, safe_pos : safe_pos + 1, :]
    token_text = _safe_token_from_id(tokenizer, tokenized_source["input_ids"][0, safe_pos].item())
    return hs_at_pos, safe_pos, token_text


def patch_target_logits(
    *,
    model,
    tokenized_tgt: dict[str, torch.Tensor],
    hs_at_pos: torch.Tensor,  # (batch, 1, hidden_dim) - hidden state to inject
    target_layer: int,
    target_position: int,
) -> torch.Tensor:
    """Run target forward with patched hidden state, returns logits."""
    patch_config_dict = {
        "patch_method": "replace",
        "layer_to_patch": target_layer,
        "hs_position": [target_position],
        "hs": hs_at_pos,
    }

    hooks = set_patch(model, [patch_config_dict])
    try:
        with torch.no_grad():
            patched_output = model(**tokenized_tgt, output_hidden_states=False)
    finally:
        remove_hooks(hooks)
    return patched_output.logits


def get_all_hidden_states_from_forward(
    model,
    tokenized_source: dict[str, torch.Tensor],
    source_layer: int,
) -> torch.Tensor:
    """Get hidden states at source_layer from forward pass.

    Returns: tensor of shape (batch, seq_len, hidden_dim)
    """
    source_outputs = model(**tokenized_source, output_hidden_states=True)
    hidden_state_tuple = source_outputs.hidden_states or ()
    if not hidden_state_tuple:
        raise RuntimeError("Model did not return hidden states; cannot perform patching.")

    hs_index = _resolve_hidden_state_index(source_layer, len(hidden_state_tuple))
    return hidden_state_tuple[hs_index]


def get_all_hidden_states_from_generation(
    model,
    tokenizer,
    tokenized_source: dict[str, torch.Tensor],
    source_prompt_texts: list[str],
    source_layer: int,
    max_gen_len: int,
    gen_cache_dir: str | None = None,
    tgt_template_name: str | None = None,
    batch_size: int = 1,
) -> tuple[torch.Tensor, list[list[int]], list[str]]:
    """Generate from source and collect hidden states at each step.

    Returns:
        gen_hs_tensor: (batch, num_gen_steps, hidden_dim)
        generated_token_ids: list of token id lists per batch item
        source_extracted_answers: extracted answer strings
    """
    cached_data = None

    if gen_cache_dir:
        os.makedirs(gen_cache_dir, exist_ok=True)
        prompt_str = "".join(source_prompt_texts)
        model_name = getattr(model, "name_or_path", "unknown_model")
        cache_key_str = f"{prompt_str}_{source_layer}_{max_gen_len}_{model_name}_{batch_size}"
        cache_hash = hashlib.md5(cache_key_str.encode("utf-8")).hexdigest()
        cache_path = os.path.join(gen_cache_dir, f"gen_cache_{cache_hash}.pt")
        if os.path.exists(cache_path):
            try:
                cached_data = torch.load(cache_path, map_location=model.device)
            except Exception as e:
                print(f"Warning: Failed to load cache {cache_path}: {e}")

    if cached_data is not None:
        gen_hs_tensor = cached_data["gen_hs_tensor"]
        generated_token_ids = cached_data["generated_token_ids"]
        source_extracted_answers = cached_data["source_extracted_answers"]
    else:
        source_pad_id = get_pad_id(tokenizer)
        source_eos_ids = get_eos_token_ids(tokenizer)

        generation = model.generate(
            **tokenized_source,
            max_new_tokens=max_gen_len,
            eos_token_id=source_eos_ids,
            pad_token_id=source_pad_id,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        output = decode_tokens(tokenizer, generation.sequences)
        source_extracted_answers = extract_answer_from_generation(
            output,
            tokenizer=tokenizer,
            template_name=tgt_template_name,
        )

        hidden_steps = generation.hidden_states or ()
        if not hidden_steps:
            raise RuntimeError("Model did not return hidden states during generation.")

        per_step_hs = []
        for step_hidden in hidden_steps:
            if step_hidden is None:
                continue
            hs_index = _resolve_hidden_state_index(source_layer, len(step_hidden))
            layer_hidden = step_hidden[hs_index]
            if layer_hidden.dim() == 3 and layer_hidden.size(1) == 1:
                per_step_hs.append(layer_hidden[:, 0, :])
            else:
                per_step_hs.append(layer_hidden[:, -1, :])

        if not per_step_hs:
            raise RuntimeError("Generation produced no hidden states to select from.")

        # Stack per-step hidden states: (batch, num_steps, hidden_dim)
        gen_hs_tensor = torch.stack(per_step_hs, dim=1)
        num_gen_steps = gen_hs_tensor.size(1)
        sequences = generation.sequences
        input_len = tokenized_source["input_ids"].shape[1]

        generated_token_ids = []
        current_eos_ids = [source_eos_ids] if isinstance(source_eos_ids, int) else source_eos_ids
        for i in range(gen_hs_tensor.size(0)):
            gen_seq = sequences[i, input_len:]
            valid_len = len(gen_seq)
            for eos_id in current_eos_ids:
                matches = (gen_seq == eos_id).nonzero(as_tuple=True)[0]
                if len(matches) > 0:
                    valid_len = min(valid_len, matches[0].item() + 1)
            valid_len = min(valid_len, num_gen_steps)
            generated_token_ids.append([int(tok) for tok in gen_seq[:valid_len].tolist()])

        if gen_cache_dir:
            prompt_str = "".join(source_prompt_texts)
            model_name = getattr(model, "name_or_path", "unknown_model")
            cache_key_str = f"{prompt_str}_{source_layer}_{max_gen_len}_{model_name}_{batch_size}"
            cache_hash = hashlib.md5(cache_key_str.encode("utf-8")).hexdigest()
            cache_path = os.path.join(gen_cache_dir, f"gen_cache_{cache_hash}.pt")

            torch.save(
                {
                    "gen_hs_tensor": gen_hs_tensor,
                    "generated_token_ids": generated_token_ids,
                    "source_extracted_answers": source_extracted_answers,
                },
                cache_path,
            )

    source_generated_answer = source_extracted_answers.get("answer_text", [])
    return gen_hs_tensor, generated_token_ids, source_generated_answer


def analyze_patch_positions(
    source_mt,
    target_mt,
    src_prompter,
    tgt_prompter,
    batched_input_source,
    batched_input_tgt,
    *,
    config: PatchConfig,
    gold_step: bool,
    patch_from_generation: bool,
    sample_idx: int,
    batch_size: int = 1,
) -> dict[str, Any]:
    cfg = config
    source_tokenizer = source_mt.tokenizer
    target_tokenizer = target_mt.tokenizer
    target_supports_system_prompt = bool(getattr(target_mt, "is_instruct_model", False))

    # Assuming batch size 1
    source_sample = batched_input_source[0]
    batched_input_tgt[0]

    sample_hs_selections = normalize_hs_selections(source_sample.get("hs_selection"))

    _, rendered_source_prompts, source_prompt_texts, tokenized_source = build_prompt_inputs(
        source_tokenizer,
        src_prompter,
        batched_input_source,
        steps=cfg.steps,
        device=source_mt.device,
        system_prompt=True,
        add_generation_prompt=True,
    )

    # source_pad_id = get_pad_id(source_tokenizer)
    # source_eos_ids = get_eos_token_ids(source_tokenizer)

    generated_token_ids = None

    if patch_from_generation:
        all_source_hs, generated_token_ids, _source_generated_answer = (
            get_all_hidden_states_from_generation(
                source_mt.model,
                source_tokenizer,
                tokenized_source,
                source_prompt_texts,
                cfg.source_layer,
                cfg.max_gen_len,
                gen_cache_dir=getattr(cfg, "gen_cache_dir", None),
                tgt_template_name=getattr(tgt_prompter, "template_name", None),
                batch_size=batch_size,
            )
        )
        candidates = list(range(len(generated_token_ids[0]))) if generated_token_ids else []
    else:
        all_source_hs = get_all_hidden_states_from_forward(
            source_mt.model,
            tokenized_source,
            cfg.source_layer,
        )
        candidates = select_candidates_from_prompt(
            tokenized_source,
            source_prompt_texts,
            source_tokenizer,
            steps=cfg.steps,
            include_all_tokens=bool(getattr(cfg, "include_all_tokens", False)),
        )

    source_hs_positions_by_mode = resolve_source_positions(
        candidates=candidates,
        generated_token_ids=generated_token_ids,
        sample_hs_selections=sample_hs_selections,
        hs_selection=cfg.hs_selection,
        patching_k=cfg.patching_k,
        patch_from_generation=patch_from_generation,
        tokenizer=source_tokenizer,
    )

    tgt_template_name = getattr(tgt_prompter, "template_name", "unknown")
    tgt_steps = resolve_target_steps(
        cfg.steps,
        tgt_template_name,
        gold_step,
    )

    _, rendered_tgt_prompts, _, tokenized_tgt = build_prompt_inputs(
        target_tokenizer,
        tgt_prompter,
        batched_input_tgt,
        steps=tgt_steps,
        device=target_mt.device,
        system_prompt=target_supports_system_prompt,
        add_generation_prompt=target_supports_system_prompt,
    )

    tgt_offsets = left_pad_offsets(tokenized_tgt)
    am = tokenized_tgt["attention_mask"][0]
    valid_len = int(am.sum().item()) if hasattr(am, "sum") else int(sum(am))

    gold_answer = batched_input_tgt[0].get("answer", "")
    gold_numeric = extract_answer([gold_answer])
    gold_label_str = str(gold_numeric[0]).strip() if gold_numeric else ""
    gold_token_ids = target_tokenizer.encode(gold_label_str, add_special_tokens=False)

    source_generated_answer_num = batched_input_source[0].get("Generated Answer_num", None)
    source_answer_num_str, source_answer_num_token_ids = encode_answer_tokens(
        target_tokenizer,
        source_generated_answer_num,
    )

    target_pad_id = get_pad_id(target_tokenizer)
    target_eos_ids = get_eos_token_ids(target_tokenizer)
    with torch.no_grad():
        target_baseline_gen = target_mt.model.generate(
            **tokenized_tgt,
            max_new_tokens=cfg.max_gen_len,
            eos_token_id=target_eos_ids,
            pad_token_id=target_pad_id,
            do_sample=False,
        )
    target_baseline_text = decode_tokens(target_tokenizer, target_baseline_gen)
    target_baseline_extracted = extract_answer_from_generation(
        target_baseline_text,
        tokenizer=target_tokenizer,
        template_name=getattr(tgt_prompter, "template_name", None),
    )
    target_baseline_answer_num = (
        target_baseline_extracted["answer_num"][0]
        if target_baseline_extracted["answer_num"]
        else None
    )
    target_answer_num_str, target_answer_num_token_ids = encode_answer_tokens(
        target_tokenizer,
        target_baseline_answer_num,
    )

    with torch.no_grad():
        baseline_output = target_mt.model(**tokenized_tgt, output_hidden_states=False)
    baseline_logits = baseline_output.logits

    gold_baseline_probs = compute_gold_label_probability(baseline_logits, gold_token_ids)
    source_answer_baseline_probs = (
        compute_gold_label_probability(baseline_logits, source_answer_num_token_ids)
        if source_answer_num_token_ids
        else None
    )
    target_answer_baseline_probs = (
        compute_gold_label_probability(baseline_logits, target_answer_num_token_ids)
        if target_answer_num_token_ids
        else None
    )

    actual_tgt_pos = resolve_patch_position(
        cfg.patch_position,
        tgt_offsets[0],
        valid_len,
    )

    _safe_token_from_id(target_tokenizer, tokenized_tgt["input_ids"][0, actual_tgt_pos].item())

    results_list = []
    for positions in source_hs_positions_by_mode.values():
        for hs_pos in positions:
            hs_at_pos, safe_pos, source_token_text = get_source_hidden_state_at_position(
                patch_from_generation=patch_from_generation,
                all_source_hs=all_source_hs,
                hs_pos=hs_pos,
                tokenized_source=tokenized_source,
                generated_token_ids=generated_token_ids,
                tokenizer=source_tokenizer,
            )
            patched_logits = patch_target_logits(
                model=target_mt.model,
                tokenized_tgt=tokenized_tgt,
                hs_at_pos=hs_at_pos,
                target_layer=cfg.target_layer,
                target_position=actual_tgt_pos,
            )

            gold_patched_probs = compute_gold_label_probability(patched_logits, gold_token_ids)
            source_answer_patched_probs = (
                compute_gold_label_probability(patched_logits, source_answer_num_token_ids)
                if source_answer_num_token_ids
                else None
            )
            target_answer_patched_probs = (
                compute_gold_label_probability(patched_logits, target_answer_num_token_ids)
                if target_answer_num_token_ids
                else None
            )

            gold_prob_change = gold_patched_probs["sum_prob"] - gold_baseline_probs["sum_prob"]

            source_prob_change = None
            if source_answer_patched_probs and source_answer_baseline_probs:
                source_prob_change = (
                    source_answer_patched_probs["sum_prob"]
                    - source_answer_baseline_probs["sum_prob"]
                )

            target_prob_change = None
            if target_answer_patched_probs and target_answer_baseline_probs:
                target_prob_change = (
                    target_answer_patched_probs["sum_prob"]
                    - target_answer_baseline_probs["sum_prob"]
                )

            results_list.append(
                {
                    "position": safe_pos,
                    "patching_token": source_token_text,
                    "gold_prob_change": gold_prob_change,
                    "source_prob_change": source_prob_change,
                    "target_prob_change": target_prob_change,
                }
            )

    final_output = {
        "question": batched_input_tgt[0]["question"],
        "answer": batched_input_tgt[0]["answer"],
        "source_answer_num": source_answer_num_str,
        "target_answer_num": target_answer_num_str,
        "gold_answer": gold_label_str,
        "gold_answer_prob": gold_baseline_probs["sum_prob"],
        "target_answer_prob": target_answer_baseline_probs["sum_prob"]
        if target_answer_baseline_probs
        else None,
        "source_answer_prob": source_answer_baseline_probs["sum_prob"]
        if source_answer_baseline_probs
        else None,
        "source_prompt": rendered_source_prompts[0] if rendered_source_prompts else "",
        "target_prompt": rendered_tgt_prompts[0] if rendered_tgt_prompts else "",
        "results": results_list,
    }

    return final_output
