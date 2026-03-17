from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Tuple

import torch

from ri.core.hooks import remove_hooks, set_patch
from ri.utils.tokenizer import (
    decode_tokens,
    get_eos_token_ids,
    get_pad_id,
    make_inputs,
    render_prompts,
)
from ri.utils.extraction import extract_answer_from_generation
from ri.utils.text import prompt_text_from_rendered
from ri.common.prompts import build_prompt_batch

from .config import PatchConfig
from .selectors import select_positions_with_mode, select_step_positions
from .tensor_ops import (
    build_word_span_map,
    compute_core_token_positions,
    left_pad_offsets,
    mask_to_positions,
)


def _resolve_hidden_state_index(requested_layer: int, total_hidden_states: int) -> int:
    """Map a layer index (matching module numbering) to the hidden_state tuple index.

    Hugging Face models return embeddings at position 0, followed by the output of each
    transformer block. Users, however, specify patch layers using the module index. Positive
    indices therefore need to be shifted by one, while negative indices can be used as-is
    because they already count from the end (and embeddings mean there is an extra entry).
    """

    max_layers = total_hidden_states - 1
    if max_layers <= 0:
        raise ValueError("Model did not produce per-layer hidden states.")

    if requested_layer >= 0:
        if requested_layer >= max_layers:
            raise ValueError(
                f"source_layer {requested_layer} is invalid for a model with {max_layers} transformer layers."
            )
        return requested_layer + 1

    if requested_layer < -max_layers:
        raise ValueError(
            f"source_layer {requested_layer} is invalid for a model with {max_layers} transformer layers."
        )

    return total_hidden_states + requested_layer


def _safe_token_from_id(tokenizer, token_id: int) -> str:
    converter = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(converter):
        tok = converter(int(token_id))
        if tok is not None:
            return str(tok)
    try:
        return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    except Exception:
        return str(int(token_id))


def _tokens_at_positions(tokenizer, ids_row, positions: List[int]) -> List[str]:
    ids_list = ids_row.tolist() if hasattr(ids_row, "tolist") else list(ids_row)
    tokens: List[str] = []
    seq_len = len(ids_list)
    for pos in positions:
        try:
            idx = int(pos)
        except Exception:
            tokens.append("")
            continue
        if idx < 0 or idx >= seq_len:
            tokens.append("")
            continue
        tokens.append(_safe_token_from_id(tokenizer, ids_list[idx]))
    return tokens


def _run_target_generation_with_patch(
    target_mt,
    tokenized_tgt,
    patch_config_list,
    cfg,
    target_eos_ids,
    target_pad_id,
    tgt_prompter,
    target_tokenizer,
    rendered_source_prompts,
    rendered_tgt_prompts,
    selected_source_token_texts,
    target_patch_tokens,
    source_extracted_answers,
    patch_from_generation,
):
    hooks = set_patch(target_mt.model, patch_config_list)
    try:
        output_toks = target_mt.model.generate(
            **tokenized_tgt,
            max_new_tokens=cfg.max_gen_len,
            eos_token_id=target_eos_ids,
            pad_token_id=target_pad_id,
            do_sample=False,
        )
    finally:
        remove_hooks(hooks)

    output = decode_tokens(target_tokenizer, output_toks)

    extracted_answers = extract_answer_from_generation(
        output,
        tokenizer=target_tokenizer,
        template_name=getattr(tgt_prompter, "template_name", None),
    )

    result_dict: Dict[str, Any] = {
        "answer_num": extracted_answers["answer_num"],
        "answer_cot": extracted_answers["answer_text"],
        "source_prompt": rendered_source_prompts,
        "target_prompt": rendered_tgt_prompts,
        "source_selected_tokens": selected_source_token_texts,
        "target_patch_token": target_patch_tokens,
    }
    if patch_from_generation:
        result_dict["source_generated_answer"] = source_extracted_answers.get(
            'answer_text', [])

    return result_dict


def get_source_hidden_states(
    source_mt,
    tokenized_source,
    source_prompt_texts,
    config: PatchConfig,
    tgt_prompter,
    patch_from_generation: bool,
    batch_size: int = 1,
) -> Tuple[torch.Tensor, Optional[List[List[int]]], Dict[str, Any]]:
    """Extract hidden states from source model.

    Returns:
        all_source_hs: (batch, seq_len, hidden_dim) hidden states
        generated_token_ids: token ids if patch_from_generation, else None
        source_extracted_answers: dict with extracted answer info
    """
    cfg = config
    source_tokenizer = source_mt.tokenizer
    source_pad_id = get_pad_id(source_tokenizer)
    source_eos_ids = get_eos_token_ids(source_tokenizer)

    source_extracted_answers = {}
    generated_token_ids = None
    all_source_hs = None

    if patch_from_generation:
        # Check for cache configuration
        cache_dir = getattr(cfg, "gen_cache_dir", None)
        cached_data = None

        if cache_dir:
            import hashlib
            import os
            os.makedirs(cache_dir, exist_ok=True)

            # Create a unique hash based on inputs that affect generation
            prompt_str = "".join(source_prompt_texts)
            model_name = getattr(
                source_mt.model, "name_or_path", "unknown_model")
            # Include layer, max_len, and batch_size in hash to differentiate experiments
            cache_key_str = f"{prompt_str}_{cfg.source_layer}_{cfg.max_gen_len}_{model_name}_{batch_size}"
            cache_hash = hashlib.md5(cache_key_str.encode("utf-8")).hexdigest()
            cache_path = os.path.join(cache_dir, f"gen_cache_{cache_hash}.pt")

            if os.path.exists(cache_path):
                try:
                    cached_data = torch.load(
                        cache_path, map_location=source_mt.device)
                except Exception as e:
                    print(f"Warning: Failed to load cache {cache_path}: {e}")

        if cached_data is not None:
            # Restore from cache
            gen_hs_tensor = cached_data["gen_hs_tensor"]
            generated_token_ids = cached_data["generated_token_ids"]
            source_extracted_answers = cached_data["source_extracted_answers"]
        else:
            gen_tokens = max(cfg.max_gen_len, cfg.patching_k or 1, 1)
            generation = source_mt.model.generate(
                **tokenized_source,
                max_new_tokens=gen_tokens,
                eos_token_id=source_eos_ids,
                pad_token_id=source_pad_id,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output = decode_tokens(source_tokenizer, generation.sequences)

            source_extracted_answers = extract_answer_from_generation(
                output,
                tokenizer=source_mt.tokenizer,  # Use source tokenizer for decoding
                template_name=getattr(tgt_prompter, "template_name", None),
            )

            hidden_steps = generation.hidden_states or ()
            if not hidden_steps:
                raise RuntimeError(
                    "Model did not return hidden states during generation; "
                    "please disable --patch_from_generation or ensure the model supports hidden states."
                )
            # Collect last token hidden state from each generation step
            per_step_hs: List[torch.Tensor] = []  # each: (batch, hidden_dim)
            for step_hidden in hidden_steps:
                if step_hidden is None:
                    continue
                hs_index = _resolve_hidden_state_index(
                    cfg.source_layer, len(step_hidden))
                layer_hidden = step_hidden[hs_index]
                if layer_hidden.dim() == 3 and layer_hidden.size(1) == 1:
                    per_step_hs.append(layer_hidden[:, 0, :])
                else:
                    per_step_hs.append(layer_hidden[:, -1, :])
            if not per_step_hs:
                raise RuntimeError(
                    "Generation produced no hidden states to select from.")

            # (batch, num_gen_steps, hidden_dim)
            gen_hs_tensor = torch.stack(per_step_hs, dim=1)
            num_gen_steps = gen_hs_tensor.size(1)

            sequences = generation.sequences
            input_len = tokenized_source["input_ids"].shape[1]

            generated_token_ids: List[List[int]] = []
            for i in range(gen_hs_tensor.size(0)):
                gen_seq = sequences[i, input_len:]
                valid_len = len(gen_seq)

                current_eos_ids = [source_eos_ids] if isinstance(
                    source_eos_ids, int) else source_eos_ids

                for eos_id in current_eos_ids:
                    matches = (gen_seq == eos_id).nonzero(as_tuple=True)[0]
                    if len(matches) > 0:
                        valid_len = min(valid_len, matches[0].item() + 1)

                valid_len = min(valid_len, num_gen_steps)
                trimmed_seq = gen_seq[:valid_len]
                generated_token_ids.append([int(tok)
                                           for tok in trimmed_seq.tolist()])

            # Save to cache if configured
            if cache_dir:
                torch.save({
                    "gen_hs_tensor": gen_hs_tensor,
                    "generated_token_ids": generated_token_ids,
                    "source_extracted_answers": source_extracted_answers
                }, cache_path)

        all_source_hs = gen_hs_tensor
    else:
        source_outputs = source_mt.model(
            **tokenized_source, output_hidden_states=True)
        hidden_state_tuple = source_outputs.hidden_states or ()
        if not hidden_state_tuple:
            raise RuntimeError(
                "Model did not return hidden states; cannot perform patching.")
        hs_index = _resolve_hidden_state_index(
            cfg.source_layer, len(hidden_state_tuple))
        all_source_hs = hidden_state_tuple[hs_index]

    return all_source_hs, generated_token_ids, source_extracted_answers


def patch_and_generate(
    source_mt,
    target_mt,
    src_prompter,
    tgt_prompter,
    batched_input_source,
    batched_input_tgt,
    *,
    patch_from_generation: bool,
    config: PatchConfig,
    gold_step: bool,
    batch_size: int = 1,
) -> Dict[str, Any]:

    cfg = config
    source_tokenizer = source_mt.tokenizer
    target_tokenizer = target_mt.tokenizer
    target_supports_system_prompt = bool(
        getattr(target_mt, "is_instruct_model", False))
    tgt_template_name = getattr(tgt_prompter, "template_name", "unknown")
    tgt_steps: Optional[int] = None
    if "non_cot" not in tgt_template_name.lower() and gold_step:
        if isinstance(cfg.steps, int) and cfg.steps > 1:
            tgt_steps = cfg.steps - 1
        elif cfg.steps == "all":
            tgt_steps = "all"

    source_convos = build_prompt_batch(
        src_prompter,
        batched_input_source,
        steps=cfg.steps,
    )

    rendered_source_prompts = render_prompts(
        source_tokenizer,
        source_convos,
        system_prompt=True,
        add_generation_prompt=True,
    )

    source_prompt_texts = [prompt_text_from_rendered(
        prompt) for prompt in source_convos]

    tokenized_source = make_inputs(
        source_tokenizer,
        source_convos,
        source_mt.device,
        system_prompt=True,
        add_generation_prompt=True,
        rendered_prompts=rendered_source_prompts,
    )

    source_prompt_positions = [
        mask_to_positions(attn_mask) for attn_mask in tokenized_source["attention_mask"]
    ]

    # selects the core part of the prompt (question + answer)
    source_core_positions, _ = compute_core_token_positions(
        tokenized_source,
        source_prompt_texts,
        source_tokenizer,
    )

    # Find step-specific positions within the core positions
    if cfg.steps in (None, "no_steps"):
        step_positions = [[] for _ in range(len(source_prompt_positions))]
    else:
        step_positions = [
            select_step_positions(
                source_core_positions[i] if i < len(
                    source_core_positions) else [],
                source_prompt_texts[i] if i < len(source_prompt_texts) else "",
                source_tokenizer,
                cfg.steps,
            )
            for i in range(len(source_prompt_positions))
        ]

    # if include_all_tokens is set, use all prompt tokens as selection pool
    use_full_selection_pool = bool(getattr(cfg, "include_all_tokens", False))
    selection_pool: List[List[int]] = []
    for i in range(len(source_prompt_positions)):
        if use_full_selection_pool:
            candidates = list(source_prompt_positions[i])
        else:
            candidates = step_positions[i] if i < len(step_positions) else []
            if not candidates and i < len(source_core_positions):
                candidates = list(source_core_positions[i])
            if not candidates:
                candidates = list(source_prompt_positions[i])
        selection_pool.append(candidates)

    source_pad_id = get_pad_id(source_tokenizer)
    source_eos_ids = get_eos_token_ids(source_tokenizer)
    target_pad_id = get_pad_id(target_tokenizer)
    target_eos_ids = get_eos_token_ids(target_tokenizer)

    all_source_hs, generated_token_ids, source_extracted_answers = get_source_hidden_states(
        source_mt,
        tokenized_source,
        source_prompt_texts,
        cfg,
        tgt_prompter,
        patch_from_generation,
        batch_size=batch_size,
    )

    if patch_from_generation:
        # Reconstruct selection pool (common to both paths)
        selection_pool = [
            list(range(len(ids))) for ids in generated_token_ids]

    # Prepare target inputs
    tgt_convos = build_prompt_batch(
        tgt_prompter,
        batched_input_tgt,
        steps=tgt_steps,
    )

    rendered_tgt_prompts = render_prompts(
        target_tokenizer,
        tgt_convos,
        system_prompt=target_supports_system_prompt,
        add_generation_prompt=target_supports_system_prompt,
    )

    tokenized_tgt = make_inputs(
        target_tokenizer,
        tgt_convos,
        target_mt.device,
        system_prompt=target_supports_system_prompt,
        add_generation_prompt=target_supports_system_prompt,
        rendered_prompts=rendered_tgt_prompts,
    )

    tgt_offsets = left_pad_offsets(tokenized_tgt)
    attn_masks = tokenized_tgt["attention_mask"]
    valid_lens: List[int] = []
    for am in attn_masks:
        if hasattr(am, "sum"):
            vl = int(am.sum().item())
        else:
            vl = int(sum(am))
        valid_lens.append(vl)

    # find where to patch in the target positions
    actual_patch_positions: List[int] = []
    for i, off in enumerate(tgt_offsets):
        vl = valid_lens[i]
        if cfg.patch_position is None:
            idx = off + vl - 1
        elif cfg.patch_position < 0:
            idx = off + vl + int(cfg.patch_position)
            if idx < off:
                idx = off
        else:
            idx = off + int(cfg.patch_position)
            max_valid = off + vl - 1
            if idx > max_valid:
                idx = max_valid
            if idx < off:
                idx = off
        actual_patch_positions.append(idx)

    hs_selection_mode = cfg.hs_selection
    positions_to_select = cfg.patching_k or 1
    selected_source_token_texts: List[List[str]] = []

    # Select token positions to extract hidden states from
    source_selected_tokens = [
        select_positions_with_mode(
            selection_pool[i],
            positions_to_select,
            hs_selection_mode,
        )
        for i in range(len(selection_pool))
    ]

    for i in range(len(source_selected_tokens)):
        if source_selected_tokens[i]:
            continue
        fallback = selection_pool[i] if i < len(selection_pool) else []
        if fallback:
            fill = int(fallback[-1])
            source_selected_tokens[i] = [fill] * positions_to_select
        else:
            source_selected_tokens[i] = [0] * positions_to_select

    if patch_from_generation:
        # We already computed source_selected_tokens using the generation pool earlier
        # Just need to extract the text for logging and the tensors for patching

        selected_source_token_texts = []
        for i, pos_list in enumerate(source_selected_tokens):
            gen_tokens = generated_token_ids[i] if i < len(
                generated_token_ids) else []
            tokens_for_sample: List[str] = []
            for pos in pos_list:
                if 0 <= pos < len(gen_tokens):
                    tokens_for_sample.append(
                        _safe_token_from_id(source_tokenizer, gen_tokens[pos])
                    )
                else:
                    tokens_for_sample.append("")
            selected_source_token_texts.append(tokens_for_sample)

        hs_device = all_source_hs.device
        selected_samples: List[torch.Tensor] = []
        hidden_size = all_source_hs.size(-1)

        for i, pos_list in enumerate(source_selected_tokens):
            if not pos_list:
                sample_vec = all_source_hs.new_zeros((0, hidden_size))
            else:
                idx = torch.tensor(
                    pos_list, device=hs_device, dtype=torch.long)
                sample_vec = all_source_hs[i].index_select(0, idx)
            selected_samples.append(sample_vec)
        selected_source_hs = torch.stack(selected_samples, dim=0)

    else:
        batch_source_hs = all_source_hs
        index_tensor = torch.tensor(
            source_selected_tokens, device=batch_source_hs.device
        ).unsqueeze(-1).expand(-1, -1, batch_source_hs.size(-1))  # (B, K, H)
        selected_source_hs = torch.gather(
            batch_source_hs,
            dim=1,
            index=index_tensor,
        )
        selected_source_token_texts = [
            _tokens_at_positions(
                source_tokenizer,
                tokenized_source["input_ids"][i],
                source_selected_tokens[i],
            )
            for i in range(len(source_selected_tokens))
        ]

    tgt_input_ids = tokenized_tgt["input_ids"]
    target_patch_tokens: List[str] = []
    for i, pos in enumerate(actual_patch_positions):
        tokens = _tokens_at_positions(
            target_tokenizer, tgt_input_ids[i], [pos])
        target_patch_tokens.append(tokens[0] if tokens else "")

    patch_config = {
        "patch_method": "replace",
        "layer_to_patch": cfg.target_layer,
        "hs_position": actual_patch_positions,
        "hs": selected_source_hs,
    }

    return _run_target_generation_with_patch(
        target_mt,
        tokenized_tgt,
        [patch_config],
        cfg,
        target_eos_ids,
        target_pad_id,
        tgt_prompter,
        target_tokenizer,
        rendered_source_prompts,
        rendered_tgt_prompts,
        selected_source_token_texts,
        target_patch_tokens,
        source_extracted_answers,
        patch_from_generation
    )
